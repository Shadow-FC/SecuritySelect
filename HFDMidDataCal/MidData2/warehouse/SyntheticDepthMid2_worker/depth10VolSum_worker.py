# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import close_price


# 10档盘口委买委卖量和
def depth10VolSum_worker(data: pd.DataFrame,
                         date: str,
                         **kwargs) -> pd.DataFrame:
    def cal(data_: pd.DataFrame) -> pd.Series:
        dataSub = data_[((data['time'] < '14:57:00') | (data_['time'] == '15:00:00')) & (
                (data_['bidvolume1'] != 0) | (data_['askvolume1'] != 0))]

        bid10VolSum = {f"bid10VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['bidvolume10sum'].tail(1)).sum()
                       for t_, T_r in close_price.items()}  # 不同时间点10挡委买量和
        ask10VolSum = {f"ask10VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['askvolume10sum'].tail(1)).sum()
                       for t_, T_r in close_price.items()}  # 不同时间点10挡委卖量和

        depth10Sum = {**bid10VolSum, **ask10VolSum, **{"date": date}}
        return pd.Series(depth10Sum)

    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)

    return res
