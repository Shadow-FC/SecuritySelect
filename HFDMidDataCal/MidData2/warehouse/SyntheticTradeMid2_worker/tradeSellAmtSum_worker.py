# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import time_AM, time_PM


def tradeSellAmtSum_worker(data: pd.DataFrame,
                           date: str,
                           **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d['sellAmount'] = d['amount'] - d['buyamount']

        sellAmtSumAM = {f"sellAmtSumAM_{t_}": d[d['time'] < T_r]['sellAmount'].sum()
                        for t_, T_r in time_AM.items()}  # 开盘不同时间段主卖额和

        sellAmtSumPM = {f"sellAmtSumPM_{t_}": d[d['time'] >= T_l]['sellAmount'].sum()
                        for t_, T_l in time_PM.items()}  # 尾盘不同时间段主卖额和

        sellAmtSum = {**sellAmtSumAM, **sellAmtSumPM, **{"date": date}}

        return pd.Series(sellAmtSum)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
