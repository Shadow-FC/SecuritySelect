# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import range_T
from utility.MidDataUtility import func_Structured_reversal


def tradeSpecial1_worker(data: pd.DataFrame,
                         date: str,
                         **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d['amtPerTrade'] = d['amount'] / d['tradenum']
        d_sub = d[range_T]
        d_sub['ret'] = d_sub['close'].pct_change(fill_method=None)

        d_sub1 = d_sub[d_sub['amtPerTrade'] >= d_sub['amtPerTrade'].quantile(0.8)]
        d_sub_inflow, d_sub_outflow = d_sub1[d_sub1['ret'] > 0], d_sub1[d_sub1['ret'] < 0]

        specialData = {
            "corCloseVol": d_sub[['close', 'volume']].corr().iloc[0, 1],  # 收盘价和成交量pearson相关系数
            "corRetVol": d_sub[['ret', 'volume']].corr().iloc[0, 1],  # 收益率和成交量pearson相关系数

            "closeVolWeightSkew": (pow((d_sub['close'] - d_sub['close'].mean()) / d_sub['close'].std(), 3) * (
                    d_sub['volume'] / d_sub['volume'].sum())).sum(),  # 加权收盘价偏度

            "AMTInFlowBigOrder": d_sub_inflow['amount'].sum(),  # 单笔成交量在前20%的成交量收益率大于零的和(大单流入)
            "AMTOutFlowBigOrder": d_sub_outflow['amount'].sum(),  # 单笔成交量在前20%的成交量收益率小于零的和(大单流出)

            "CashFlow": (np.sign(d_sub['close'].diff(1)) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 资金流向(成交量加权收盘价差分和)

            "MOMBigOrder": (d_sub[d_sub['amtPerTrade'] >= d_sub['amtPerTrade'].quantile(0.8)]['ret'] + 1).prod(
                min_count=1),  # 大单驱动涨幅

            "retD": np.log(1 + abs(np.log(d_sub['close'] / d_sub['close'].shift(1)))).sum(),
            # 轨迹非流动因子分母
            "RevStruct": func_Structured_reversal(d_sub, 0.1),  # 结构化反转因子

            "date": date  # 日期
        }

        return pd.Series(specialData)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
