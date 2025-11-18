from collections import defaultdict
from typing import Dict, Any, List, Tuple, TypedDict, Optional
import pandas as pd
from langgraph.graph import StateGraph
from sqlalchemy.engine.create import create_engine
from sqlalchemy.pool.impl import QueuePool
from datetime import datetime, timedelta
from app.core.agent.graph.sql_graph import db_uri
import logging

# 配置日志
logging.basicConfig(level=logging.DEBUG)

engine = create_engine(
    db_uri,
    poolclass=QueuePool,  # 使用队列池（默认）
    pool_size=10,  # 连接池大小
    max_overflow=20,  # 最大溢出连接
    pool_timeout=30,  # 获取连接超时时间
    pool_pre_ping=True,  # 预先检查连接有效性
    pool_recycle=3600,  # 连接回收时间（避免数据库断开）
    echo=False  # 设为True可查看SQL日志（调试用）
)
DATE_FMT = "%Y%m%d"
START_DT = "20250601"
END_DT = "20250610"


class PandasSQLQueryTool:
    def __init__(self, engine):
        self.engine = engine

    def invoke(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        执行查询并返回字典列表
        支持参数化查询，防止SQL注入
        """
        try:
            with self.engine.connect() as conn:
                result = pd.read_sql(query, conn, params=params)  # 直接执行SQL，无参数
                return result.to_dict('records')
        except Exception as e:
            logging.error(f"查询执行失败: {e}")
            logging.error(f"SQL: {query}")
            return []


def load_ccy_mapping():
    """加载币种映射表到内存"""

    sql = "SELECT ccy_int, ccy_symb FROM ccy_mapping"
    results = execute_query_tool.invoke(sql)
    _CCY_MAPPING = {
        row['ccy_symb']: row['ccy_int']  # symb -> int
        for row in results
    }
    return _CCY_MAPPING


def parse_dt(dt: str) -> datetime:
    try:
        return datetime.strptime(dt, DATE_FMT)
    except ValueError as e:
        logging.error(f"Error: 日期格式错误，无法将 '{dt}' 转换为日期。")
        raise e  # 继续抛出异常


# 使用方式相同
execute_query_tool = PandasSQLQueryTool(engine)


def classify_errors(records: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    from datetime import timedelta

    date_set = set()
    dt_start = parse_dt(START_DT)
    dt_end = parse_dt(END_DT)
    for d in range((dt_end - dt_start).days + 1):
        date_set.add((dt_start + timedelta(days=d)).strftime(DATE_FMT))

    bucket = defaultdict(list)
    for r in records:
        key = (r['org_num'], r['sbj_num'], r['ccy'])
        bucket[key].append(r)

    type1, type2, type3 = [], [], []

    for key, rows in bucket.items():
        # ---------- 预处理 ----------
        rows.sort(key=lambda x: x['dt'])
        exist_dates = {r['dt'] for r in rows}
        full_period = (exist_dates == date_set)  # 是否 10 天全量
        diffs = [float(r['tot_mint_dif']) for r in rows]
        non_zero_count = sum(1 for d in diffs if d != 0)  # 不平记录条数

        # ---------- Type1：十天全在 + 差额恒定 ----------
        if full_period and len(set(diffs)) == 1:
            rows[0]['is_first'] = True
            type1.append(rows[1])
            continue

        # ---------- Type3：且非全量 ----------
        if (
                not full_period
                and non_zero_count < 10
                and non_zero_count > 0
        ):
            first_nz = next(i for i, d in enumerate(diffs) if d != 0)
            last_nz = len(diffs) - 1 - next(i for i, d in enumerate(reversed(diffs)) if d != 0)
            rows[0]['zero_span'] = {'start': rows[first_nz]['dt'],
                                    'end': rows[last_nz]['dt']}
            type3.append(rows[0])
            continue

        # ---------- Type2：其余出现≥2种差额的情况 ----------
        change_list, change_dates = [], []
        for i, d in enumerate(diffs):
            if i == 0 or d != diffs[i - 1]:
                change_list.append(d)
                change_dates.append(rows[i]['dt'])
        if len(change_list) >= 2:
            rows[0]['change_list'] = change_list
            rows[0]['change_dates'] = change_dates
            type2.append(rows[0])

    return {'type1': type1, 'type2': type2, 'type3': type3}


class OutputState(TypedDict):
    discrepancies: List[Dict[str, Any]]
    classes: Dict[str, List[Dict[str, Any]]]
    results: List[Dict[str, Any]]
    summary: Dict[str, Any]  # 建议保留 summary，对主智能体有用


# -------- State --------
class AgentState(TypedDict, total=False):
    discrepancies: List[Dict[str, Any]]  # 原始不平明细
    classes: Dict[str, List[Dict[str, Any]]]  # type1/type2/type3
    current_type_index: Dict[str, int]  # {"type1": 0, "type2": 0, "type3": 0}
    current_date_index: int  # 对于 type2，当前处理到 change_dates 的第几个日期
    current_record: Dict[str, Any]  # 当前处理的分类记录
    current_target: Tuple[str, str, str, str]  # 当前处理组 (org, sbj, ccy, dt)
    current_type: str  # 当前处理的类型
    has_more: bool
    history: Dict[str, Any]
    individual: Dict[str, Any]
    per_account: List[Dict[str, Any]]
    results: List[Dict[str, Any]]  # 累计各组结果
    summary: Dict[str, Any]
    red_blue_cancellations: List[Dict[str, Any]]  # 新增：用于存储冲销凭证检查结果


# -------- Helpers (参数化版本 SQL，避免硬编码) --------
def _print_classification_analysis(classes: Dict[str, List[Dict[str, Any]]], discrepancies: List[Dict[str, Any]]):
    """
    打印三类错误的分类结果和分析原因
    """
    logging.info("\n" + "=" * 80)
    logging.info("【错误分类分析报告】")
    logging.info("=" * 80)

    total_records = len(discrepancies)
    type1_records = classes.get("type1", [])
    type2_records = classes.get("type2", [])
    type3_records = classes.get("type3", [])

    logging.info(f"\n总计发现 {total_records} 条不平记录，分类如下：")
    logging.info(f"  - Type1 (恒定差额): {len(type1_records)} 组")
    logging.info(f"  - Type2 (差额变化): {len(type2_records)} 组")
    logging.info(f"  - Type3 (差额归零): {len(type3_records)} 组")

    # 分析 Type1
    if type1_records:
        logging.info("\n【Type1 - 恒定差额错误】")
        logging.info(
            "分析原因：6月1日起总账户与分户合计差额恒定，业务期间分户/总账同步变动。该总分不平发生在6月1日之前，建议您往6月1日前追溯原因。")
        logging.info("判断标准：")
        logging.info("  1. 该组(org_num, sbj_num, ccy)在查询期间内所有日期都有记录")
        logging.info("  2. 所有日期的 tot_mint_dif 值完全相同（恒定差额）")
        logging.info("  3. 说明：可能存在系统性的余额计算错误或初始余额设置问题")
        logging.info(f"\n共 {len(type1_records)} 组，详情：")
        for idx, record in enumerate(type1_records, 1):
            logging.info(f"  [{idx}] 机构: {record.get('org_num')}, 科目: {record.get('sbj_num')}, "
                         f"币种: {record.get('ccy')}, 差额: {record.get('tot_mint_dif')}")

    # 分析 Type2
    if type2_records:
        logging.info("\n【Type2 - 差额变化错误】")
        logging.info(
            "分析原因：6月1日起总账户与分户合计产生差额不固定，业务期间分户/总账不同步变动。该总分不平发生在6月1日之前，同时中间又发生了新的错误，建议您对该账户的相关情况进行具体分析。")
        logging.info("判断标准：")
        logging.info("  1. 在查询期间内，该组的 tot_mint_dif 值发生了至少一次变化")
        logging.info("  2. 存在多个不同的差额值（change_list 长度 ≥ 2）")
        logging.info("  3. 说明：可能在特定日期发生了交易或调整，导致差额发生变化")
        logging.info(f"\n共 {len(type2_records)} 组，详情：")
        for idx, record in enumerate(type2_records, 1):
            change_list = record.get('change_list', [])
            change_dates = record.get('change_dates', [])
            logging.info(f"  [{idx}] 机构: {record.get('org_num')}, 科目: {record.get('sbj_num')}, "
                         f"币种: {record.get('ccy')}")
            logging.info(f"      变化点: {len(change_list)} 个，差额值: {change_list}")
            logging.info(f"      变化日期: {change_dates}")

    # 分析 Type3
    if type3_records:
        logging.info("\n【Type3 - 差额归零错误】")
        logging.info(
            "分析原因：账户部分天数总分平衡，部分天数总分不平。建议借助平衡法则“当天余额=上一天余额±借方发生额±贷方发生额”进行计算找到错误")
        logging.info("判断标准：")
        logging.info("  1. 该组在查询期间内不是所有日期都有记录（非全量）")
        logging.info("  2. 不平记录数少于总天数，但大于0")
        logging.info("  3. 存在一个日期范围（zero_span），在这个范围内差额从非零变为零")
        logging.info("  4. 说明：可能在某段时间内发生了错误，之后被纠正或自动归零")
        logging.info(f"\n共 {len(type3_records)} 组，详情：")
        for idx, record in enumerate(type3_records, 1):
            zero_span = record.get('zero_span', {})
            logging.info(f"  [{idx}] 机构: {record.get('org_num')}, 科目: {record.get('sbj_num')}, "
                         f"币种: {record.get('ccy')}")
            if zero_span:
                logging.info(f"      异常日期范围: {zero_span.get('start')} 至 {zero_span.get('end')}")

    logging.info("\n" + "=" * 80)
    logging.info("开始逐组验证...")


def _print_account_result(state: AgentState):
    """
    打印每个账户处理完成后的结果，包括错误原因和可疑记录
    """
    org, sbj, ccy, acg_dt = state.get("current_target", ("", "", "", ""))
    current_type = state.get("current_type", "unknown")
    record = state.get("current_record", {})
    history = state.get("history", {})
    individual = state.get("individual", {})
    per_account = state.get("per_account", [])

    logging.info("\n" + "-" * 80)
    logging.info(f"【处理完成 - {current_type.upper()}】")
    logging.info("-" * 80)
    logging.info(f"机构: {org}, 科目: {sbj}, 币种: {ccy}, 日期: {acg_dt}")

    if current_type == "type1":
        logging.info("\n【错误原因分析】")
        logging.info("Type1 - 恒定差额错误：")
        logging.info("  6月1日起总账户与分户合计差额恒定，业务期间分户/总账同步变动。")
        logging.info("  可能原因：")
        logging.info("    1. 系统性的余额计算错误")
        logging.info("    2. 初始余额设置问题")
        logging.info("    3. 科目余额与分户余额之间存在固定偏差")
        if record:
            logging.info(f"  恒定差额值: {record.get('tot_mint_dif', 'N/A')}")

    elif current_type == "type2":
        logging.info("\n【错误原因分析】")
        logging.info("Type2 - 差额变化错误：")
        logging.info(
            " 6月1日起总账户与分户合计产生差额不固定，业务期间分户/总账不同步变动。该总分不平发生在6月1日之前，同时中间又发生了新的错误.")
        logging.info("  可能原因：")
        logging.info("    1. 在特定日期发生了交易或调整")
        logging.info("    2. 传票数据与分户余额数据在变化点日期不一致")
        logging.info("    3. 可能存在数据录入错误或冲正操作")
        change_list = record.get('change_list', [])
        change_dates = record.get('change_dates', [])
        if change_list:
            logging.info(f"  差额变化序列: {change_list}")
            logging.info(f"  变化日期: {change_dates}")

    elif current_type == "type3":
        logging.info("\n【错误原因分析】")
        logging.info("Type3 - 差额归零错误：")
        logging.info("  账户部分天数总分平衡，部分天数总分不平。")
        logging.info("  可能原因：")
        logging.info("    1. 在某段时间内发生了错误，之后被纠正")
        logging.info("    2. 可能存在红蓝字冲销操作")
        logging.info("    3. 数据在异常期间后自动归零")
        zero_span = record.get('zero_span', {})
        if zero_span:
            logging.info(f"  异常日期范围: {zero_span.get('start')} 至 {zero_span.get('end')}")
            red_blue_result = state.get("red_blue_cancellations", {})
            if current_type == "type3" and red_blue_result:
                summary = red_blue_result.get("summary", {})
                vouchers = red_blue_result.get("raw_vouchers", [])
                tot_records = red_blue_result.get("tot_records", [])
                match_result = red_blue_result.get("matches", [])
                logging.info("\n【冲销凭证分析】")
                logging.info(f"  {summary.get('note', '')}")
                logging.info(f"  → {summary.get('conclusion', '')}")
                logging.info("\n【冲销嫌疑匹配详情】")
                if not match_result:
                    logging.info("→ 未发现凭证金额与总差异高度吻合的记录。")
                else:
                    for i, item in enumerate(match_result, 1):
                        v = item["voucher"]
                        t = item["tot_record"]
                        diff = item["abs_diff"]
                        rd_flag = "🔴 R" if v.get("rd_flg") == "R" else "🔵 B"
                        logging.info(f"{i:2d}. {rd_flag} 凭证 {v['vchr_num']} | 日期 {v['dt']} | 金额 {v['amt']:+.2f} "
                                     f"≈ 差异 {t['dif']:+.2f} (差值 {diff:.4f})")

    logging.info("\n【验证结果汇总】")
    logging.info("  History表(传票发生额):")
    logging.info(f"    - 账户数: {history.get('count', 0)}")
    logging.info(f"    - 总借方: {history.get('total_debit', 0):.2f}")
    logging.info(f"    - 总贷方: {history.get('total_credit', 0):.2f}")
    logging.info(f"    - 总差额: {history.get('total_diff', 0):.2f}")
    # logging.info(f"   - 可疑的账号: {per_account:.2f}")
    logging.info("  Individual表(分户余额差):")
    logging.info(f"    - 账户数: {individual.get('count', 0)}")
    logging.info(f"    - 总差额: {individual.get('total_diff', 0):.2f}")
    # 添加前30个不一致的账号信息
    inconsistent_accounts = [r for r in per_account if not r["is_consistent"]]
    logging.info("  传票历史跟分户差额不一致的账号 (前30个):")
    if per_account == []:
        logging.info(f"→ 传票历史表跟分户余额表其中一个表存在对应的{org}, {sbj}, {ccy}, {acg_dt}丢失，请检查。")
    for i, account in enumerate(inconsistent_accounts[:30], start=1):
        logging.info(
            f"    [{i}] 账号: {account['acct_num']}, 差异: {account['difference']:.4f}, 错误率: {account['error_rate']:.6f}%,借贷发生额: {account['history_balance_diff']},分户差额: {account['individual_balance_diff']}")

    logging.info("-" * 80 + "\n")


def _validate_voucher_today(acg_dt: str, org_num: str, sbj_num: str, ccy_symb: str) -> Dict[str, Any]:
    sql = f"""
        SELECT
            t.acct_num,
            t.acg_org_num,
            t.sbj_num,
            t.ccy_symb,
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
        GROUP BY t.acct_num, t.acg_org_num, t.sbj_num, t.ccy_symb;
    """
    rows = execute_query_tool.invoke(sql)
    return {
        "count": len(rows),
        "total_debit": sum(r['debit_amt'] for r in rows),
        "total_credit": sum(r['credit_amt'] for r in rows),
        "total_diff": sum(r['balance_diff'] for r in rows),
        "records": rows,
        "summary_diff": sum(r['debit_amt'] for r in rows) - sum(r['credit_amt'] for r in rows),
    }


def _validate_ledger_day(acg_dt: str, org_num: str, sbj_num: str, ccy_int: str) -> Dict[str, Any]:
    # 需要 acg_dt+1
    acg_dt_after = (datetime.strptime(acg_dt, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")
    sql = f"""
        SELECT 
            a.acct_num,
            a.sbj_num,
            a.ccy,
            a.bal_prev_day,
            b.bal_curr_day,
            b.bal_curr_day - a.bal_prev_day AS balance_diff
        FROM (
            SELECT acct_num, sbj_num, ccy, CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_prev_day
            FROM individual_total
            WHERE dt = '{acg_dt}' 
              AND org_num = '{org_num}'
              AND sbj_num = '{sbj_num}'
              AND ccy = '{ccy_int}'
        ) a
        JOIN (
            SELECT acct_num, sbj_num, ccy, CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_curr_day
            FROM individual_total
            WHERE dt = '{acg_dt_after}' 
              AND org_num = '{org_num}'
              AND sbj_num = '{sbj_num}'
              AND ccy = '{ccy_int}'
        ) b ON a.acct_num = b.acct_num 
           AND a.sbj_num = b.sbj_num 
           AND a.ccy = b.ccy;
    """
    rows = execute_query_tool.invoke(sql)
    return {
        "count": len(rows),
        "records": rows,
        "total_diff": sum(r['balance_diff'] for r in rows),
    }


def _compare_account_diffs(history_rows: List[Dict[str, Any]], individual_rows: List[Dict[str, Any]]) -> List[
    Dict[str, Any]]:
    history = {r['acct_num']: float(r['balance_diff']) for r in history_rows}
    individual = {r['acct_num']: float(r['balance_diff']) for r in individual_rows}
    common = sorted(set(history) & set(individual))
    out = []
    for acct in common:
        h = abs(history[acct])
        i = abs(individual[acct])
        diff = h - i
        out.append({
            "acct_num": acct,
            "history_balance_diff": h,
            "individual_balance_diff": i,
            "difference": diff,
            "is_consistent": abs(diff) < 0.01,
            "error_rate": abs(diff / h * 100) if h != 0 else 0,
        })
    return out


from datetime import datetime
from typing import List, Dict, Any


def _check_red_blue_cancellation_in_type3(
        org_num: str,
        sbj_num: str,
        ccy_symb: str,
        start_dt: str,
        end_dt: str,
) -> dict[str, dict[str, str | int | float] | list[Any] | int | Any]:
    """精准匹配模式：仅比对 tot.dif 与 voucher.amt 是否相等（容差 ±0.001），返回所有匹配项"""
    if not all([org_num, sbj_num, ccy_symb, start_dt, end_dt]):
        raise ValueError("所有参数必须提供")

    try:
        datetime.strptime(start_dt, "%Y%m%d")
        datetime.strptime(end_dt, "%Y%m%d")
    except ValueError:
        raise ValueError("日期格式必须为 YYYYMMDD")

    # === Step 1: 查询 zero_span 期间内所有凭证（仅需 amt + 基础字段）===
    sql_vchr = f"""
        SELECT 
            vchr_num,
            dt,
            ldin_flg,
            rd_flg,
            CAST(amt AS DECIMAL(18,2)) AS amt
        FROM history_total
        WHERE acg_org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy_symb = '{ccy_symb}'
          AND dt BETWEEN '{start_dt}' AND '{end_dt}'
          AND vchr_num IS NOT NULL
        ORDER BY acg_dt, vchr_num;
    """
    raw_vouchers = execute_query_tool.invoke(sql_vchr)

    # === Step 2: 查询 tot 表 dif 记录（仅需 dt + dif）===
    sql_tot = f"""
        SELECT 
            dt,
            CAST(tot_mint_dif AS DECIMAL(18,2)) AS dif
        FROM tot
        WHERE org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_symb}'
          AND dt BETWEEN '{start_dt}' AND '{end_dt}'
        ORDER BY dt;
    """
    tot_records = execute_query_tool.invoke(sql_tot)

    # === Step 3: 两两比对 amt 与 dif，误差 < 0.001 视为匹配 ===
    matches = []
    TOLERANCE = 0.001

    for v in raw_vouchers:
        v_amt = float(v["amt"])
        for t in tot_records:
            t_dif = float(t["dif"])
            if abs(v_amt - t_dif) < TOLERANCE:
                matches.append({
                    "voucher": v,
                    "tot_record": t,
                    "abs_diff": abs(v_amt - t_dif)
                })

    # === Step 4: 构建返回结果 ===
    summary = {
        "note": f"【冲销嫌疑匹配分析】期间 {start_dt}–{end_dt}："
                f"共 {len(raw_vouchers)} 笔凭证，{len(tot_records)} 条差异记录；"
                f"发现 {len(matches)} 组凭证金额与当日总差异高度吻合（误差 < {TOLERANCE}）。",
        "match_count": len(matches),
        "tolerance_used": TOLERANCE,
        "interpretation": (
            "⚠️ 注意：此类精确匹配常见于红字冲销（R）或蓝字反向凭证操作，"
            "可能导致单日凭证金额直接体现为 tot_mint_dif。"
            "建议人工核查匹配项中的 rd_flg='R' 或异常借贷方向凭证。"
        )
    }

    return {
        "summary": summary,
        "matches": matches,  # 按 amt ≈ dif 匹配成功的可疑冲销候选
        "raw_vouchers": raw_vouchers,
        "tot_records": tot_records,
        "suspicious_candidates": len([m for m in matches if m["voucher"].get("rd_flg") == "R"]),
    }


# -------- Nodes --------
def node_scan(state: AgentState) -> AgentState:
    sql = """
        SELECT 
                org_num, 
                sbj_num, 
                ccy, 
                sbact_acct_bal,
                gnl_ldgr_bal,
                tot_mint_dif,
                dt 
        FROM tot 
        WHERE CAST(NULLIF(tot_mint_dif, '') AS NUMERIC(18,2)) != 0.00
        ORDER BY  org_num, sbj_num, ccy,dt;
    """
    records = execute_query_tool.invoke(sql)
    state["discrepancies"] = records
    state["classes"] = classify_errors(records)

    # 打印分类分析结果
    _print_classification_analysis(state["classes"], records)

    # 初始化索引：按优先级 type1 -> type2 -> type3
    state["current_type_index"] = {"type1": 0, "type2": 0, "type3": 0}
    state["current_date_index"] = 0
    state["results"] = []
    state["has_more"] = True  # 初始化为 True，表示有记录需要处理
    return state


def node_pick_next(state: AgentState) -> AgentState:
    """
    从 classes 中按优先级选择下一个要处理的记录和日期
    默认优先级：type3 -> type1 -> type2
    对于 type2，需要遍历 change_dates 中的所有日期
    """
    classes = state.get("classes", {})
    type_index = state.get("current_type_index", {"type1": 0, "type2": 0, "type3": 0})

    # 优先级顺序
    type_order = ["type3", "type1", "type2"]

    for type_name in type_order:
        type_records = classes.get(type_name, [])
        if not type_records:
            continue

        idx = type_index.get(type_name, 0)
        if idx >= len(type_records):
            continue  # 这个类型已经处理完，继续下一个类型

        record = type_records[idx]
        org = record["org_num"]
        sbj = record["sbj_num"]
        ccy = record["ccy"]

        # 根据类型决定处理哪些日期
        if type_name == "type1":
            # type1: 恒定差额，处理第一个日期即可
            dt = record["dt"]
            state["current_record"] = record
            state["current_target"] = (org, sbj, ccy, dt)
            state["current_type"] = type_name
            # 处理完这条记录，移动到下一条
            type_index[type_name] = idx + 1
            state["current_type_index"] = type_index
            state["has_more"] = True  # 标记还有更多记录
            return state

        elif type_name == "type2":
            # type2: 有多个变化点，需要处理 change_dates 中的每个日期
            change_dates = record.get("change_dates", [])
            if not change_dates:
                # 如果没有 change_dates，使用 dt
                dt = record["dt"]
                state["current_record"] = record
                state["current_target"] = (org, sbj, ccy, dt)
                state["current_type"] = type_name
                type_index[type_name] = idx + 1
                state["current_date_index"] = 0
                state["current_type_index"] = type_index
                state["has_more"] = True
                return state

            # 获取当前记录的日期索引（如果当前记录不是这条，重置为0）
            current_record_key = f"{org}|{sbj}|{ccy}"
            last_record = state.get("current_record", {})
            last_record_key = f"{last_record.get('org_num', '')}|{last_record.get('sbj_num', '')}|{last_record.get('ccy', '')}"

            if current_record_key != last_record_key:
                # 切换到新记录，重置日期索引
                state["current_date_index"] = 0

            date_idx = state.get("current_date_index", 0)
            if date_idx < len(change_dates):
                # 还有日期未处理
                dt = change_dates[date_idx]
                state["current_record"] = record
                state["current_target"] = (org, sbj, ccy, dt)
                state["current_type"] = type_name
                state["current_date_index"] = date_idx + 1
                state["current_type_index"] = type_index
                state["has_more"] = True
                return state
            else:
                # 这个记录的所有日期都处理完了，移动到下一条记录
                type_index[type_name] = idx + 1
                state["current_date_index"] = 0
                state["current_type_index"] = type_index
                # 继续循环，处理下一条记录
                continue

        elif type_name == "type3":
            # type3: 处理 zero_span 中的日期范围
            zero_span = record.get("zero_span", {})
            if zero_span:
                # 可以处理 span 的 start 和 end，或者整个范围
                # 这里先处理 start 日期
                dt = zero_span.get("start", record["dt"])
            else:
                dt = record["dt"]

            state["current_record"] = record
            state["current_target"] = (org, sbj, ccy, dt)
            state["current_type"] = type_name
            type_index[type_name] = idx + 1
            state["current_type_index"] = type_index
            state["has_more"] = True
            return state

    # 所有类型都处理完了
    state["has_more"] = False
    # 如果没有更多记录，也要确保 current_target 存在（避免 validate 节点报错）
    if "current_target" not in state or state.get("current_target") is None:
        # 如果没有任何记录，设置一个默认值（虽然不应该发生）
        state["current_target"] = ("", "", "", "")
    return state


def node_decide(state: AgentState) -> str:
    """
    判断是否还有需要处理的记录
    """
    # 直接检查 has_more 标志
    if not state.get("has_more", False):
        return "finish"
    return "next"


def node_validate(state: AgentState) -> AgentState:
    org, sbj, ccy_symb, acg_dt = state["current_target"]

    ccy_mapping = load_ccy_mapping()
    ccy_int = ccy_mapping.get(ccy_symb)
    if not ccy_int:
        raise ValueError(f"无效的币种符号: {ccy_symb}")

    history = _validate_voucher_today(acg_dt, org, sbj, ccy_symb)
    individual = _validate_ledger_day(acg_dt, org, sbj, ccy_int)
    state["history"] = history
    state["individual"] = individual
    return state


def node_compare(state: AgentState) -> AgentState:
    per = _compare_account_diffs(state["history"]["records"], state["individual"]["records"])
    state["per_account"] = per
    inc = [r for r in per if not r["is_consistent"]]
    org, sbj, ccy, acg_dt = state["current_target"]

    # 添加类型信息和分类记录中的额外信息
    result = {
        "org_num": org,
        "sbj_num": sbj,
        "ccy": ccy,
        "acg_dt": acg_dt,
        "type": state.get("current_type", "unknown"),
        "history_total_diff": state["history"]["total_diff"],
        "individual_total_diff": state["individual"]["total_diff"],
        "account_inconsistent_count": len(inc),
        "inconsistent_accounts": inc[:50],
    }

    # 根据类型添加额外信息
    record = state.get("current_record", {})
    if state.get("current_type") == "type2":
        result["change_list"] = record.get("change_list", [])
        result["change_dates"] = record.get("change_dates", [])
    elif state.get("current_type") == "type3":
        result["zero_span"] = record.get("zero_span", {})
        # 对于type3类型，执行冲销凭证检查
        zero_span = record.get("zero_span", {})
        if zero_span:
            start_dt = zero_span.get("start", acg_dt)
            end_dt = zero_span.get("end", acg_dt)
            red_blue_cancellations = _check_red_blue_cancellation_in_type3(
                org, sbj, ccy, start_dt, end_dt
            )
            state["red_blue_cancellations"] = red_blue_cancellations
            result["red_blue_cancellations"] = red_blue_cancellations

    state["results"].append(result)

    # 打印每个账户处理完成后的结果
    _print_account_result(state)

    return state


def node_finish(state: AgentState) -> AgentState:
    total_discrepancies = len(state.get("discrepancies", []))
    state["summary"] = {
        "total_discrepancies": total_discrepancies,
        "group_count": len(state.get("results", [])),
        "type1": len(state.get("classes", {}).get("type1", [])),
        "type2": len(state.get("classes", {}).get("type2", [])),
        "type3": len(state.get("classes", {}).get("type3", [])),
    }
    return state


# -------- Graph builder --------
def build_graph():
    g = StateGraph(AgentState)
    g.add_node("scan", node_scan)
    g.add_node("pick_next", node_pick_next)
    g.add_node("validate", node_validate)
    g.add_node("compare", node_compare)
    g.add_node("finish", node_finish)

    g.set_entry_point("scan")
    g.add_edge("scan", "pick_next")
    g.add_edge("pick_next", "validate")
    g.add_edge("validate", "compare")
    g.add_conditional_edges("compare", node_decide, {
        "finish": "finish",
        "next": "pick_next",
    })

    return g.compile()


# -------- Public API --------
def run_react() -> Dict[str, Any]:
    app = build_graph()
    final = app.invoke({}, config={"recursion_limit": 100})
    # ✅ 投影：仅保留主智能体需要的字段
    output: OutputState = {
        "discrepancies": final.get("discrepancies", []),
        "classes": final.get("classes", {}),
        "results": final.get("results", []),
        "summary": final.get("summary", {}),
    }
    return output


if __name__ == "__main__":
    import json

    try:
        result = run_react()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"执行出错: {e}")
        import traceback

        traceback.print_exc()