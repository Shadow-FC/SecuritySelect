# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
import scipy.stats as st
from mapping import range_T
from utility.MidDataUtility import func_M_sqrt, func_M_ln, entropy


def tradeSpecial2_worker(data: pd.DataFrame,
                         date: str,
                         **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d_sub = d[range_T]
        d_sub['ret'] = d_sub['close'].pct_change(fill_method=None)

        specialData = {
            "volEntropy": entropy(d_sub['close'] * d_sub['volume']),  # 单位一成交量占比熵
            "amtEntropy": entropy(d_sub['amount']),  # 成交额占比熵

            "naiveAmtR": (st.t.cdf(d_sub['close'].diff(1) / d_sub['close'].diff(1).std(), len(d_sub) - 1) * d_sub[
                'amount']).sum() / d_sub['amount'].sum(),  # 朴素主动占比因子
            "TAmtR": (st.t.cdf(d_sub['ret'] / d_sub['ret'].std(), len(d_sub) - 1) * d_sub['amount']).sum() / d_sub[
                'amount'].sum(),  # T分布主动占比因子
            "NAmtR": (st.norm.cdf(d_sub['ret'] / d_sub['ret'].std()) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 正态分布主动占比因子
            "CNAmtR": (st.norm.cdf(d_sub['ret'] / 0.1 * 1.96) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 置信正态分布主动占比因子
            "EventAmtR": ((d_sub["ret"] - 0.1) / 0.2 * d_sub['amount']).sum() / d_sub['amount'].sum(),  # 均匀分布主动占比因子

            "SmartQ": func_M_sqrt(d_sub),  # 聪明钱因子
            "SmartQln": func_M_ln(d_sub),  # 聪明钱因子改进

            "date": date  # 日期
        }

        return pd.Series(specialData)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
