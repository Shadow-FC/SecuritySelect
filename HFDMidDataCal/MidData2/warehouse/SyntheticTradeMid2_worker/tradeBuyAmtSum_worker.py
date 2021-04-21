# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import time_AM, time_PM


def tradeBuyAmtSum_worker(data: pd.DataFrame,
                          date: str,
                          **kwargs) -> pd.DataFrame:

    def cal(data_: pd.DataFrame) -> pd.Series:
        buyAmtSumAM = {f"buyAmtSumAM_{t_}": data_[data_['time'] < T_r]['buyamount'].sum()
                       for t_, T_r in time_AM.items()}  # 开盘不同时间段主买额和

        buyAmtSumPM = {f"buyAmtSumPM_{t_}": data_[data_['time'] >= T_l]['buyamount'].sum()
                       for t_, T_l in time_PM.items()}  # 尾盘不同时间段主买额和

        buyAmtSum = {**buyAmtSumAM, **buyAmtSumPM, **{"date": date}}

        return pd.Series(buyAmtSum)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
