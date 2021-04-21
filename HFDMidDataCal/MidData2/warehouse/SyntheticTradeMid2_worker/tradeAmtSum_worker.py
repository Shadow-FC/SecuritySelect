# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import time_PM, time_AM, range_T


def tradeAmtSum_worker(data: pd.DataFrame,
                       date: str,
                       **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d_sub = d[range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        amtSumAM = {f"amtAM_{time_}": d[d['time'] < right_time]['amount'].sum()
                    for time_, right_time in time_AM.items()}  # 开盘不同时间段成交量和

        amtSumPM = {f"amtPM_{time_}": d[d['time'] >= left_time]['amount'].sum()
                    for time_, left_time in time_PM.items()}  # 尾盘不同时间段成交量和

        amtSumSp = {
            "amtRetUpSum_0": d_sub[d_sub['ret'] > 0]['amount'].sum(),  # 收益率大于0的成交额和
            "amtRetDownSum_0": d_sub[d_sub['ret'] < 0]['amount'].sum(),  # 收益率小于0的成交额和
            "amtRetEqualSum_0": d_sub[np.isnan(d_sub['ret']) | (d_sub['ret'] == 0)]['amount'].sum(),  # # 收益率等于0的成交额和
            "date": date  # 日期
        }

        amtSum = {**amtSumAM, **amtSumPM, **amtSumSp}

        return pd.Series(amtSum)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
