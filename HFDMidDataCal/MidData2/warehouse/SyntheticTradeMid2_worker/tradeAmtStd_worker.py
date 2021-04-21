# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import time_std


def tradeAmtStd_worker(data: pd.DataFrame,
                       date: str,
                       **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d['sellAmount'] = d['amount'] - d['buyamount']
        d['netAmount'] = d['buyamount'] - d['sellAmount']

        buyAmtStd = {f"buyAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['buyamount'].std()
                     for t_, T_ in time_std.items()}  # 不同时间段主买额标准差
        sellAmtStd = {f"sellAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['sellAmount'].std()
                      for t_, T_ in time_std.items()}  # 不同时间段主卖额标准差
        netAmtStd = {f"netAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['netAmount'].std()
                     for t_, T_ in time_std.items()}  # 不同时间段净主买额标准差

        allAmtAtd = {f"amtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['amount'].std()
                     for t_, T_ in time_std.items()}  # 不同时间段成交额标准差

        amtStd = {**buyAmtStd, **sellAmtStd, **allAmtAtd, **netAmtStd, **{"date": date}}

        return pd.Series(amtStd)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
