"""Bank Reconciliation Agent — Clean LangGraph Implementation.

Follows best practices:
- Single unified state (ResearcherState-style)
- Tools are pure, typed, docstring-compliant @tool functions
- LLM controls flow via `think_tool`
- No hidden index/state mutation in nodes
"""

import os
from typing import List, Dict, Any, Optional, Literal, TypedDict, Sequence
from typing_extensions import Annotated
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph.message import add_messages
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from app.core.agent.tools.prompts import SYSTEM_PROMPT
from operator import add
from dotenv import load_dotenv

# ====== Core Dependencies ======
from app.core.db.workflow.bank_flow import (
    execute_query_tool,
    classify_errors,
    load_ccy_mapping,
)


# ====== Data Models (for type safety) ======


class DiscrepancyRecord(TypedDict):
    org_num: str
    sbj_num: str
    ccy: str
    tot_mint_dif: float
    dt: str


class ValidationResult(TypedDict):
    org_num: str
    sbj_num: str
    ccy: str
    dt: str
    history_total_diff: float
    individual_total_diff: float
    inconsistent_accounts: List[Dict[str, Any]]
    per_account_comparison: List[Dict[str, Any]]


class AuditEntry(TypedDict):
    """
    One audit log entry per discrepancy group (or failed attempt).
    Designed for frontend table rendering.
    """

    # 主键 & 元信息
    org_num: str
    sbj_num: str
    ccy: str
    dt: str
    discrepancy_type: Literal["type1", "type2", "type3", "skipped", "error"]
    processed_at: str  # ISO format

    # 核心指标（用于汇总表格）
    history_total_diff: float
    individual_total_diff: float
    diff_gap: float  # |history - individual|
    inconsistent_account_count: int

    # 存疑明细（用于展开详情）
    inconsistent_accounts: List[Dict[str, Any]]  # from compare_account_differences
    red_blue_cancellations: List[Dict[str, Any]]  # from check_red_blue_cancellation_in_type3
    error_message: Optional[str]  # e.g., Day-1 skip reason

    # 附加上下文
    zero_span: Optional[Dict[str, str]]  # for type3
    change_dates: Optional[List[str]]  # for type2


# ====== Tools (Pure, Typed, Docstring-Compliant) ======


@tool(parse_docstring=True)
def scan_and_classify_discrepancies() -> Dict[Literal["type1", "type2", "type3"], List[DiscrepancyRecord]]:
    """Scan reconciliation table and classify discrepancies by temporal pattern.

    Scans the `tot` table for records where `tot_mint_dif ≠ 0`, then classifies them as:
    - type1: constant non-zero difference across dates
    - type2: difference changes on specific dates (has 'change_dates')
    - type3: difference returns to zero over a date span (has 'zero_span')

    Returns:
        Dictionary with keys 'type1', 'type2', 'type3', each mapping to list of records.
        Each record has: org_num, sbj_num, ccy, tot_mint_dif, dt (+ optional metadata)
    """
    # Step 1: Scan
    sql = """
        SELECT 
            org_num, sbj_num, ccy, tot_mint_dif, dt ,CASE WHEN SUBSTR(dt, 7, 2) = '01' THEN 1 ELSE 0 END AS is_day_one
        FROM tot 
        WHERE CAST(NULLIF(tot_mint_dif, '') AS NUMERIC(18,2)) != 0.00
        ORDER BY org_num, sbj_num, ccy, dt;
    """
    raw_records = execute_query_tool.invoke(sql)
    discrepancies = [
        DiscrepancyRecord(
            org_num=r["org_num"],
            sbj_num=r["sbj_num"],
            ccy=r["ccy"],
            tot_mint_dif=float(r["tot_mint_dif"]),
            dt=r["dt"],
            is_day_one=bool(r["is_day_one"]),  # ← 新增字段
        )
        for r in raw_records
    ]

    # Step 2: Classify
    classified = classify_errors(discrepancies)
    # Ensure typed output
    out: Dict[Literal["type1", "type2", "type3"], List[DiscrepancyRecord]] = {"type1": [], "type2": [], "type3": []}
    for typ in ["type1", "type2", "type3"]:
        out[typ] = [
            DiscrepancyRecord(
                org_num=r["org_num"],
                sbj_num=r["sbj_num"],
                ccy=r["ccy"],
                tot_mint_dif=float(r["tot_mint_dif"]),
                dt=r["dt"],
            )
            for r in classified.get(typ, [])
        ]
    return out


@tool(parse_docstring=True)
def check_red_blue_cancellation_in_type3(
    org_num: str,
    sbj_num: str,
    ccy_symb: str,
    start_dt: str,
    end_dt: str,
) -> List[Dict[str, Any]]:
    """检查type3类型差异期间内是否存在异常冲销凭证（R字），并验证其金额总和是否匹配总差异。

    Args:
        org_num: 组织编号
        sbj_num: 科目编号
        ccy_symb: 货币符号
        start_dt: 差异起始日期 (YYYYMMDD)
        end_dt: 差异结束日期 (YYYYMMDD)

    Returns:
        List[Dict]: 每个元素包含冲销凭证详情及对总差异的贡献度。
        示例：
        [
            {
                "vchr_num": "V001",
                "acg_dt": "20251105",
                "amt": 100.0,
                "rd_flg": "R",
                "contribution_pct": 50.0
            },
            ...
        ]
    """
    # 验证输入
    if not all([org_num, sbj_num, ccy_symb, start_dt, end_dt]):
        raise ValueError("所有参数必须提供")

    try:
        datetime.strptime(start_dt, "%Y%m%d")
        datetime.strptime(end_dt, "%Y%m%d")
    except ValueError:
        raise ValueError("日期格式必须为 YYYYMMDD")

    # Step 1: 查询该期间内所有 R 字冲销凭证
    sql = f"""
        SELECT 
            vchr_num,
            acg_dt,
            amt,
            rd_flg,
            ldin_flg  -- 用于判断方向
        FROM history_total
        WHERE acg_org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy_symb = '{ccy_symb}'
          AND acg_dt BETWEEN '{start_dt}' AND '{end_dt}'
          AND rd_flg = 'R'
        ORDER BY acg_dt, vchr_num;
    """

    raw_records = execute_query_tool.invoke(sql)

    if not raw_records:
        return []

    # Step 2: 计算所有冲销凭证的净影响金额（考虑借贷方向）
    cancellation_amounts = []
    total_impact = 0.0

    for r in raw_records:
        amt = float(r["amt"])
        # 根据贷方/借方标识调整冲销金额符号
        # 红字冲销：如果是贷方，则实际减少贷方 → 相当于增加借方 → 正向影响
        #           如果是借方，则实际减少借方 → 相当于增加贷方 → 负向影响
        if r["ldin_flg"] == "C":  # 贷方红字冲销 → 减少贷方 → 增加净借方 → +amt
            impact = amt
        elif r["ldin_flg"] == "D":  # 借方红字冲销 → 减少借方 → 增加净贷方 → -amt
            impact = -amt
        else:
            impact = 0.0  # 不处理未知方向

        cancellation_amounts.append(
            {
                "vchr_num": r["vchr_num"],
                "acg_dt": r["acg_dt"],
                "amt": amt,
                "rd_flg": r["rd_flg"],
                "ldin_flg": r["ldin_flg"],
                "impact": impact,
            }
        )
        total_impact += impact

    # Step 3: 计算总差异值（通过查询 tot 表）
    sql_tot = f"""
        SELECT SUM(CAST(NULLIF(tot_mint_dif, '') AS NUMERIC(18,2))) AS total_diff
        FROM tot
        WHERE org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_symb}'
          AND dt BETWEEN '{start_dt}' AND '{end_dt}';
    """
    tot_result = execute_query_tool.invoke(sql_tot)
    total_diff = float(tot_result[0]["total_diff"]) if tot_result and tot_result[0]["total_diff"] else 0.0

    # Step 4: 为每条冲销记录添加贡献率
    results = []
    for item in cancellation_amounts:
        contribution_pct = abs(item["impact"] / total_diff * 100) if total_diff != 0 else 0.0
        results.append(
            {
                "vchr_num": item["vchr_num"],
                "acg_dt": item["acg_dt"],
                "amt": item["amt"],
                "rd_flg": item["rd_flg"],
                "ldin_flg": item["ldin_flg"],
                "impact": round(item["impact"], 2),
                "contribution_pct": round(contribution_pct, 2),
            }
        )

    # Step 5: 添加总结性提示
    if abs(total_impact - total_diff) < 0.01:  # 容差 0.01
        results.append({"note": "✅ 冲销凭证总额与期间总差异高度吻合，疑似由冲销操作导致差异。"})
    elif abs(total_impact) > 0:
        results.append({"note": f"⚠️ 冲销凭证总额 ({round(total_impact, 2)}) 与总差异 ({round(total_diff, 2)}) 不一致，需进一步核查。"})

    return results


@tool(parse_docstring=True)
def validate_voucher_and_ledger(
    org_num: str,
    sbj_num: str,
    ccy_symb: str,
    acg_dt: str,
) -> Dict[str, Any]:
    """Validate voucher totals (history_total) vs. ledger day-to-day balances (individual_total).

    For a given (org, subject, currency, date), computes:
    - history: debit/credit/balance_diff from **same-day vouchers**
    - individual: balance changes from **day-to-day ledger snapshots** (requires previous-day balance)

    ⚠️ Note: This tool **cannot be used on the 1st day of any month** (e.g., '20250601'),
    because the previous-day balance (e.g., '20250531') belongs to a different accounting period
    and may be unavailable or require opening-balance adjustments.

    Args:
        org_num: Accounting organization code (e.g., '001')
        sbj_num: Subject/account code (e.g., '1001')
        ccy_symb: Currency symbol (e.g., 'CNY', 'USD')
        acg_dt: Accounting date in YYYYMMDD format (e.g., '20251110')

    Returns:
        Dictionary with:
        - "history": { "records": [...], "total_diff": float }
        - "individual": { "records": [...], "total_diff": float }
        - OR, if acg_dt is Day 1: { "error": "Insufficient data..." }
    """
    # Validate inputs
    if not all([org_num, sbj_num, ccy_symb, acg_dt]):
        raise ValueError("All parameters required: org_num, sbj_num, ccy_symb, acg_dt")
    try:
        dt = datetime.strptime(acg_dt, "%Y%m%d")
    except ValueError:
        raise ValueError("acg_dt must be in YYYYMMDD format")

    # 🔴 新增检查：是否为每月1号？
    if acg_dt.endswith("01"):
        return {
            "error": (
                f"❌ Cannot validate on Day 1 ({acg_dt}).\n"
                "Reason: Previous-day balance belongs to prior month and is structurally unavailable "
                "for same-period reconciliation.\n"
                "Recommendation: Skip Day 1 records during discrepancy analysis."
            )
        }

    # --- History validation (vouchers on acg_dt) ---
    sql_hist = f"""
        SELECT
            t.acct_num,
            SUM(CASE
                    WHEN t.ldin_flg = 'D' AND (t.rd_flg IS NULL OR t.rd_flg = 'B') THEN CAST(t.amt AS DECIMAL(18,2))
                    WHEN t.ldin_flg = 'D' AND t.rd_flg = 'R' THEN -CAST(t.amt AS DECIMAL(18,2))
                    ELSE 0
                END) AS debit_amt,
            SUM(CASE
                    WHEN t.ldin_flg = 'C' AND (t.rd_flg IS NULL OR t.rd_flg = 'B') THEN CAST(t.amt AS DECIMAL(18,2))
                    WHEN t.ldin_flg = 'C' AND t.rd_flg = 'R' THEN -CAST(t.amt AS DECIMAL(18,2))
                    ELSE 0
                END) AS credit_amt,
            SUM(CASE
                    WHEN t.ldin_flg = 'D' AND (t.rd_flg IS NULL OR t.rd_flg = 'B') THEN CAST(t.amt AS DECIMAL(18,2))
                    WHEN t.ldin_flg = 'D' AND t.rd_flg = 'R' THEN -CAST(t.amt AS DECIMAL(18,2))
                    WHEN t.ldin_flg = 'C' AND (t.rd_flg IS NULL OR t.rd_flg = 'B') THEN -CAST(t.amt AS DECIMAL(18,2))
                    WHEN t.ldin_flg = 'C' AND t.rd_flg = 'R' THEN CAST(t.amt AS DECIMAL(18,2))
                    ELSE 0
                END) AS balance_diff
        FROM history_total t
        WHERE t.dt = '{acg_dt}'
          AND t.acg_org_num = '{org_num}'
          AND t.sbj_num = '{sbj_num}'
          AND t.ccy_symb = '{ccy_symb}'
        GROUP BY t.acct_num;
    """
    hist_rows = execute_query_tool.invoke(sql_hist)
    hist_summary = {
        "records": [
            {
                "acct_num": r["acct_num"],
                "debit_amt": float(r["debit_amt"]),
                "credit_amt": float(r["credit_amt"]),
                "balance_diff": float(r["balance_diff"]),
            }
            for r in hist_rows
        ],
        "total_diff": sum(float(r["balance_diff"]) for r in hist_rows),
    }

    # --- Individual ledger (dt-1 → dt) ---
    ccy_map = load_ccy_mapping()
    ccy_int = ccy_map.get(ccy_symb)
    if not ccy_int:
        raise ValueError(f"Unsupported currency symbol: {ccy_symb}")

    dt_prev = (dt - timedelta(days=1)).strftime("%Y%m%d")
    sql_indiv = f"""
        SELECT 
            a.acct_num,
            a.bal_prev_day,
            b.bal_curr_day,
            b.bal_curr_day - a.bal_prev_day AS balance_diff
        FROM (
            SELECT acct_num, CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_prev_day
            FROM individual_total
            WHERE dt = '{dt_prev}' 
              AND org_num = '{org_num}'
              AND sbj_num = '{sbj_num}'
              AND ccy = '{ccy_int}'
        ) a
        JOIN (
            SELECT acct_num, CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_curr_day
            FROM individual_total
            WHERE dt = '{acg_dt}' 
              AND org_num = '{org_num}'
              AND sbj_num = '{sbj_num}'
              AND ccy = '{ccy_int}'
        ) b ON a.acct_num = b.acct_num;
    """
    indiv_rows = execute_query_tool.invoke(sql_indiv)
    indiv_summary = {
        "records": [
            {
                "acct_num": r["acct_num"],
                "bal_prev_day": float(r["bal_prev_day"]),
                "bal_curr_day": float(r["bal_curr_day"]),
                "balance_diff": float(r["balance_diff"]),
            }
            for r in indiv_rows
        ],
        "total_diff": sum(float(r["balance_diff"]) for r in indiv_rows),
    }
    # 🔥【核心新增】内联 compare_account_differences 逻辑
    # Compare per-account balance differences
    hist_map = {r["acct_num"]: abs(r["balance_diff"]) for r in hist_summary["records"]}
    indiv_map = {r["acct_num"]: abs(r["balance_diff"]) for r in indiv_summary["records"]}
    common_accts = sorted(set(hist_map) & set(indiv_map))

    comparison_results = []
    for acct in common_accts:
        h = hist_map[acct]
        i = indiv_map[acct]
        diff = h - i
        error_rate = abs(diff / h * 100) if h != 0 else 0.0
        comparison_results.append(
            {
                "acct_num": acct,
                "history_balance_diff": round(h, 2),
                "individual_balance_diff": round(i, 2),
                "difference": round(diff, 2),
                "is_consistent": abs(diff) < 0.01,
                "error_rate": round(error_rate, 2),
            }
        )
    if comparison_results == []:
        print(
            "❌ No discrepancies found between history and individual ledger.\n"
            "Reason: All accounts have zero balance difference between history and individual ledger.\n"
            "Recommendation: Skip this group and move to next."
        )
        print(f"{hist_summary}")
        print(f"{indiv_summary}")
        return {
            "history": hist_summary,
            "individual": indiv_summary,
            "per_account_comparison": comparison_results,
            "error": (
                "❌ No discrepancies found between history and individual ledger.\n"
                "Reason: All accounts have zero balance difference between history and individual ledger.\n"
                "Recommendation: Skip this group and move to next."
            ),
        }
    print(comparison_results)
    return {
        "history": hist_summary,
        "individual": indiv_summary,
        "per_account_comparison": comparison_results,  # ← 直接返回比对结果
        # "error" is absent unless Day-1 (handled above)
    }


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Strategic reflection tool for reconciliation planning and decision-making.

    Use this tool to:
    - After scanning: "I found X type1, Y type2, Z type3 discrepancies. Next I'll validate group (org, sbj, ccy, dt)."
    - After validation: "History diff = 100, ledger diff = 99.8 → close. Will compare accounts."
    - After comparison: "3/50 accounts inconsistent. Record and move to next group."
    - Before finishing: "All 12 groups processed. Ready to summarize."

    Args:
        reflection: Your reasoning about current findings, next target, and progress

    Returns:
        Acknowledgment string to confirm reflection was recorded.
    """
    return f"✅ Reflection recorded: {reflection}"


# ====== State Definitions (Single Unified State) ======


class ReconciliationState(TypedDict):
    """
    State for the bank reconciliation agent.

    Tracks message history, iteration count (for safety), current processing target,
    and accumulated validation results.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    current_target: Optional[tuple[str, str, str, str]]  # (org, sbj, ccy, dt) if processing
    results: Annotated[List[ValidationResult], add]
    #
    audit_log: Annotated[List[AuditEntry], add]  # ← 关键：用 `add` 实现追加


class ReconciliationOutput(TypedDict):
    """
    Final output of the reconciliation agent.
    """

    summary: Dict[str, Any]
    results: List[ValidationResult]
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ====== LLM Setup ======

tools = [
    scan_and_classify_discrepancies,
    validate_voucher_and_ledger,
    think_tool,
]
tools_by_name = {t.name: t for t in tools}
load_dotenv()
# Use your preferred model
api_key = os.getenv("SILICON_API_KEY")
base_url = os.getenv("SILICON_BASE_URL")
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    temperature=0.7,
    api_key=api_key,
    base_url=base_url,
)

model_with_tools = model.bind_tools(tools)


# ====== Nodes ======


def agent_node(state: ReconciliationState) -> dict:
    """LLM decides next action based on conversation history."""
    sys_msg = SystemMessage(content=SYSTEM_PROMPT)
    messages = [sys_msg] + list(state["messages"])
    response = model_with_tools.invoke(messages)
    return {
        "messages": [response],
        "tool_call_iterations": state.get("tool_call_iterations", 0) + 1,
    }


def tool_node(state: ReconciliationState) -> dict:
    """Execute tool calls from last LLM message."""
    last_msg = state["messages"][-1]
    if not hasattr(last_msg, "tool_calls"):
        return {}

    outputs = []
    for tc in last_msg.tool_calls:
        tool = tools_by_name.get(tc["name"])
        if not tool:
            outputs.append(
                ToolMessage(
                    content=f"Tool '{tc['name']}' not found.",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )
            continue

        try:
            result = tool.invoke(tc["args"])
            outputs.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"],
                    name=tc["name"],
                    additional_kwargs={"structured_result": result},
                )
            )
        except Exception as e:
            outputs.append(
                ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tc["id"],
                    name=tc["name"],
                )
            )

    return {"messages": outputs}


def update_results_node(state: ReconciliationState) -> dict:
    """Extract structured validation results and accumulate into `results` AND `audit_log`."""
    new_results: List[ValidationResult] = []
    new_audit_entries: List[AuditEntry] = []

    # Find latest validation + comparison (+ optional cancellation) messages
    validation_msg = None
    comparison_msg = None
    cancellation_msg = None

    for msg in reversed(state["messages"]):
        if msg.name == "validate_voucher_and_ledger" and hasattr(msg, "additional_kwargs"):
            validation_msg = msg
        elif msg.name == "validate_voucher_and_ledger" and hasattr(msg, "additional_kwargs"):
            comparison_msg = msg
        elif msg.name == "check_red_blue_cancellation_in_type3" and hasattr(msg, "additional_kwargs"):
            cancellation_msg = msg
        if validation_msg and comparison_msg:
            break  # cancellation is optional

    # --- Case 1: 正常验证完成（含 comparison）---
    if validation_msg and comparison_msg:
        val = validation_msg.additional_kwargs["structured_result"]
        cmp = comparison_msg.additional_kwargs["structured_result"]
        cancels = cancellation_msg.additional_kwargs["structured_result"] if cancellation_msg else []

        # Extract args
        args = getattr(validation_msg, "tool_call_args", {}) or {}
        org = args.get("org_num", "")
        sbj = args.get("sbj_num", "")
        ccy = args.get("ccy_symb", "")
        dt = args.get("acg_dt", "")

        # Inconsistent accounts
        inc = [r for r in cmp if not r["is_consistent"]]

        # Build ValidationResult (as before)
        new_results.append(
            ValidationResult(
                org_num=org,
                sbj_num=sbj,
                ccy=ccy,
                dt=dt,
                history_total_diff=val["history"]["total_diff"],
                individual_total_diff=val["individual"]["total_diff"],
                inconsistent_accounts=inc,
                per_account_comparison=cmp,
            )
        )

        # 👇 新增：构建 AuditEntry
        new_audit_entries.append(
            AuditEntry(
                org_num=org,
                sbj_num=sbj,
                ccy=ccy,
                dt=dt,
                discrepancy_type="unknown",  # LLM 应在 think_tool 中指定；可从上下文推断
                processed_at=datetime.now().isoformat(),
                history_total_diff=val["history"]["total_diff"],
                individual_total_diff=val["individual"]["total_diff"],
                diff_gap=abs(val["history"]["total_diff"] - val["individual"]["total_diff"]),
                inconsistent_account_count=len(inc),
                inconsistent_accounts=inc,
                red_blue_cancellations=cancels,
                error_message=None,
                zero_span=None,
                change_dates=None,
            )
        )

    # --- Case 2: validate_voucher_and_ledger 返回 error（如 Day-1 跳过）---
    elif validation_msg and not validation_msg.additional_kwargs.get("structured_result", {}).get("history"):
        val_res = validation_msg.additional_kwargs.get("structured_result", {})
        if "error" in val_res:
            args = getattr(validation_msg, "tool_call_args", {}) or {}
            org = args.get("org_num", "")
            sbj = args.get("sbj_num", "")
            ccy = args.get("ccy_symb", "")
            dt = args.get("acg_dt", "")

            new_audit_entries.append(
                AuditEntry(
                    org_num=org,
                    sbj_num=sbj,
                    ccy=ccy,
                    dt=dt,
                    discrepancy_type="type1",
                    processed_at=datetime.now().isoformat(),
                    history_total_diff=0.0,
                    individual_total_diff=0.0,
                    diff_gap=0.0,
                    inconsistent_account_count=0,
                    inconsistent_accounts=[],
                    red_blue_cancellations=[],
                    error_message=val_res["error"],
                    zero_span=None,
                    change_dates=None,
                )
            )
    # ✅ 返回结果
    return {
        "results": new_results,
        "audit_log": new_audit_entries,
    }


def should_continue(state: ReconciliationState) -> Literal["tool_node", "update_results", "finish", "agent_node"]:
    """Route based on LLM action and progress."""
    last_msg = state["messages"][-1] if state.get("messages") else None
    iterations = state.get("tool_call_iterations", 0)

    # Safety: max 50 iterations
    if iterations > 50:
        return "finish"

    # If LLM called tools → execute them
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tool_node"

    # If last action was comparison or think, update results
    if (last_msg and last_msg.name in ["validate_voucher_and_ledger"]) or (
        hasattr(last_msg, "content") and "ready to summarize" in str(last_msg.content).lower()
    ):
        return "update_results"

    # Default: let LLM re-respond (e.g., after tool results)
    # 修改这里：返回一个已定义的节点名称
    return "agent_node"


def finish_node(state: ReconciliationState) -> ReconciliationOutput:
    """Produce final structured output."""
    results = state.get("results", [])
    scan_msg = next((m for m in reversed(state["messages"]) if m.name == "scan_and_classify_discrepancies"), None)
    classes = scan_msg.additional_kwargs.get("structured_result", {}) if scan_msg else {}

    summary = {
        "total_groups_processed": len(results),
        "type1_count": len(classes.get("type1", [])),
        "type2_count": len(classes.get("type2", [])),
        "type3_count": len(classes.get("type3", [])),
        "inconsistent_account_count": sum(len(r["inconsistent_accounts"]) for r in results),
    }

    return {
        "summary": summary,
        "results": results,
        "messages": list(state["messages"]),
    }


# ====== Build Graph ======

builder = StateGraph(ReconciliationState, output_schema=ReconciliationOutput)

builder.add_node("agent_node", agent_node)
builder.add_node("tool_node", tool_node)
builder.add_node("update_results", update_results_node)
builder.add_node("finish", finish_node)

builder.set_entry_point("agent_node")

builder.add_conditional_edges(
    "agent_node",
    should_continue,
    {
        "tool_node": "tool_node",
        "update_results": "update_results",
        "finish": "finish",
    },
)

builder.add_edge("tool_node", "agent_node")
builder.add_edge("update_results", "agent_node")
builder.add_edge("finish", END)

reconciliation_agent = builder.compile()


# ====== Public API ======


def run_reconciliation(topic: str = "Bank Reconciliation Analysis") -> ReconciliationOutput:
    """Run the reconciliation agent to completion."""
    initial_state = {
        "messages": [HumanMessage(content=f"Analyze discrepancies for: {topic}")],
        "tool_call_iterations": 0,
        "current_target": None,
        "results": [],
    }

    config = RunnableConfig(recursion_limit=100, configurable={"thread_id": f"recon_{int(datetime.now().timestamp())}"})

    return reconciliation_agent.invoke(initial_state, config=config)


if __name__ == "__main__":
    import json

    try:
        output = run_reconciliation()
        print(json.dumps(output, ensure_ascii=False, indent=2, default=str))
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
