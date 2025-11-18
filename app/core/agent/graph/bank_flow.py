#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/11/1 21:25
# @Author  : 周启航-开发
# @File    : bank_flow.py
# bank_flow.py
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd

from app.core.db.workflow.config import engine  # 数据库连接
import json
# ---------- 配置 ----------
DATE_FMT = "%Y%m%d"
START_DT = "20250601"
END_DT   = "20250610"

# ---------- 工具 ----------
def parse_dt(dt: str) -> datetime:
    return datetime.strptime(dt, DATE_FMT)

def dt_between(dt: str) -> bool:
    return START_DT <= dt <= END_DT
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
                result = pd.read_sql(query, conn)  # 直接执行SQL，无参数
                return result.to_dict('records')
        except Exception as e:
            print(f"[ERROR] 查询执行失败: {e}")
            print(f"SQL: {query}")
            return []
_CCY_MAPPING = None

def load_ccy_mapping():
    """加载币种映射表到内存"""
    global _CCY_MAPPING
    if _CCY_MAPPING is None:
        sql = "SELECT ccy_int, ccy_symb FROM ccy_mapping"
        results = execute_query_tool.invoke(sql)
        _CCY_MAPPING = {
            row['ccy_symb']: row['ccy_int']  # symb -> int
            for row in results
        }
    return _CCY_MAPPING

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
def calculate_per_account_diff(history_results: List[Dict], individual_results: List[Dict]) -> List[Dict[str, Any]]:
    """
    计算每个共同账户的balance_diff差异
    返回：每个账户的差异详情
    """
    # 转换为字典，key为acct_num
    history_dict = {r['acct_num']: r['balance_diff'] for r in history_results}
    individual_dict = {r['acct_num']: r['balance_diff'] for r in individual_results}

    # 获取共同账户
    common_accts = set(history_dict.keys()) & set(individual_dict.keys())

    # 计算每个账户的差异
    diff_details = []
    for acct_num in sorted(common_accts):
        history_diff = abs(history_dict[acct_num])
        individual_diff = abs(individual_dict[acct_num])
        diff_between_tables = history_diff - individual_diff  # 关键：计算两个diff的差异

        diff_details.append({
            "acct_num": acct_num,
            "history_balance_diff": history_diff,
            "individual_balance_diff": individual_diff,
            "difference": diff_between_tables,  # 差异值（应该为0）
            "is_consistent": abs(diff_between_tables) < 0.01,  # 允许微小浮点误差
            "error_rate": abs(diff_between_tables / history_diff * 100) if history_diff != 0 else 0
        })

    return diff_details
# 3. 简化后的主函数（无需复杂解析）
def scan_and_validate_discrepancies_optimized() -> List[Dict[str, Any]]:
    """
    优化版：直接返回字典，省去所有解析逻辑
    """
    try:
        # 提取差异记录（直接得到字典列表）
        discrepancy_sql = """
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
             AND SUBSTR(dt, 7, 2) != '01' 
            ORDER BY  org_num, sbj_num, ccy,dt;
        """

        # 直接返回字典列表！
        discrepancy_records = execute_query_tool.invoke(discrepancy_sql)
        result = classify_errors(discrepancy_records)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not discrepancy_records:
            print("[INFO] 未找到差异记录")
            return []

        print(f"[INFO] 找到 {len(discrepancy_records)} 条差异记录，开始验证...")

    except Exception as e:
        print(f"[ERROR] 查询失败: {e}")
        return []

    # 遍历验证（直接使用字典）
    results = []
    acg_dt = '20250601'
    org_num = '220070667'
    sbj_num = '01228107'
    ccy = 'YCN'
    validation_invidual = validate_ledger_day(acg_dt,org_num, sbj_num, ccy)
    validation_history = validate_voucher_today(acg_dt,org_num, sbj_num, ccy)
    per_account_diff = calculate_per_account_diff(validation_history['records'], validation_invidual['records'])

    # 筛选有差异的账户
    inconsistent_accounts = [r for r in per_account_diff if not r['is_consistent']]

    # 打印结果
    print(f"发现 {len(inconsistent_accounts)} 个账户存在差异：")
    for acc in inconsistent_accounts:
        print(f"账户 {acc['acct_num']}: "
              f"history={acc['history_balance_diff']}, "
              f"individual={acc['individual_balance_diff']}, "
              f"差异={acc['difference']:.2f}")
    results.append(inconsistent_accounts)
    # for idx, record in enumerate(discrepancy_records, 1):
    #     try:
    #         # 直接访问字典字段
    #         org_num = record['org_num']
    #         sbj_num = record['sbj_num']
    #         ccy = record['ccy']
    #         acg_dt = record['acg_dt']
    #
    #         print(f"[{idx}/{len(discrepancy_records)}] 验证: {org_num}/{sbj_num}/{ccy}")
    #
    #         # 调用验证函数
    #         #validation_invidual = validate_ledger_day(acg_dt,org_num, sbj_num, ccy)
    #         validation_history = validate_voucher_today(acg_dt,org_num, sbj_num, ccy)
    #
    #     except Exception as e:
    #         error_info = {
    #             "record": record,
    #             "error": f"验证异常: {str(e)}",
    #             "is_consistent": False
    #         }
    #         results.append(error_info)
    #         print(f"    [ERROR] {error_info['error']}")
    return discrepancy_records


def validate_ledger_day(acg_dt: str, org_num: str, sbj_num: str, ccy: str) -> Dict[str, Any]:
    """
    基于分户余额差汇总查询
    """
    acg_dt = '20250601'
    org_num = '220070667'
    sbj_num = '01228107'
    ccy = 'YCN'
    # 加载币种映射
    ccy_mapping = load_ccy_mapping()
    ccy_int = ccy_mapping.get(ccy)
    if not ccy_int:
        raise ValueError(f"无效的币种符号: {ccy}")

    date_before = datetime.strptime(acg_dt, '%Y%m%d')
    date_after = date_before + timedelta(days=1)
    acg_dt_after = date_after.strftime('%Y%m%d')

    # ⚠️ 直接拼接参数到SQL字符串中
    sql = f"""
            SELECT 
        a.acct_num,
        a.sbj_num,
        a.ccy,
        a.bal_prev_day,
        b.bal_curr_day,
        b.bal_curr_day - a.bal_prev_day AS balance_diff
    FROM (
        SELECT acct_num, sbj_num, ccy, 
               CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_prev_day
        FROM individual_total
        WHERE dt = '{acg_dt}' 
          AND org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_int}'
    ) a
    JOIN (
        SELECT acct_num, sbj_num, ccy,
               CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_curr_day
        FROM individual_total
        WHERE dt = '{acg_dt_after}' 
          AND org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_int}'
    ) b ON a.acct_num = b.acct_num 
       AND a.sbj_num = b.sbj_num 
       AND a.ccy = b.ccy;
       """
    # 执行查询（无参数列表）
    results = execute_query_tool.invoke(sql)
    total_diff = sum(r['balance_diff'] for r in results)

    return {
        "count": len(results),
        "records": results,
        "total_diff": total_diff
    }


def validate_voucher_today(acg_dt: str, org_num: str, sbj_num: str, ccy: str) -> Dict[str, Any]:
    """
    传票发生额汇总（优化版）
    - 增加借贷差额计算
    - 修复SQL注入漏洞
    - 消除重复计算
    """
    acg_dt = '20250601'
    org_num= '220070667'
    sbj_num = '01228107'
    ccy = 'YCN'
    # 使用参数化查询，防止SQL注入
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
          AND t.ccy_symb = '{ccy}'
        GROUP BY t.acct_num, t.acg_org_num, t.sbj_num, t.ccy_symb;
    """

    # 使用参数化查询
    results = execute_query_tool.invoke(sql)

    # 计算汇总（从Python移到SQL更优）
    total_debit = sum(r['debit_amt'] for r in results)
    total_credit = sum(r['credit_amt'] for r in results)
    total_diff = sum(r['balance_diff'] for r in results)  # 新增：直接汇总差额

    return {
        "count": len(results),
        "total_debit": total_debit,
        "total_credit": total_credit,
        "total_diff": total_diff,  # 新增
        "records": results,
        "summary_diff": total_debit - total_credit  # 保留兼容字段
    }

def index_test():
    acg_dt = '20250601'
    acg_dt_after = '20250602'
    org_num = '220070667'
    sbj_num = '01228107'
    ccy_int = '615'
    sql = f"""
            SELECT 
        a.acct_num,
        a.sbj_num,
        a.ccy,
        a.bal_prev_day,
        b.bal_curr_day,
        b.bal_curr_day - a.bal_prev_day AS balance_diff
    FROM (
        SELECT acct_num, sbj_num, ccy, 
               CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_prev_day
        FROM individual_total
        WHERE dt = '{acg_dt}' 
          AND org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_int}'
    ) a
    JOIN (
        SELECT acct_num, sbj_num, ccy,
               CAST(sbact_acct_bal AS DECIMAL(18,2)) AS bal_curr_day
        FROM individual_total
        WHERE dt = '{acg_dt_after}' 
          AND org_num = '{org_num}'
          AND sbj_num = '{sbj_num}'
          AND ccy = '{ccy_int}'
    ) b ON a.acct_num = b.acct_num 
       AND a.sbj_num = b.sbj_num 
       AND a.ccy = b.ccy;
       """
    # 使用参数化查询
    results = execute_query_tool.invoke(sql)
    return {
        "count": len(results),
        "records": results
    }

if __name__ == "__main__":
    # question = "机构220070667科目01228107币种YCN总分不平，总分差额为1016.25，"
    try:
        #result = scan_and_validate_discrepancies_optimized()#第一类错误编写
        result=index_test()#第二类错误编写
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"执行出错: {e}")
