# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import close_price


# 5档盘口委买委卖量和
def depth5VolSum_worker(data: pd.DataFrame,
                        date: str,
                        **kwargs) -> pd.DataFrame:

    def cal(data_: pd.DataFrame) -> pd.Series:

        dataSub = data_[((data['time'] < '14:57:00') | (data_['time'] == '15:00:00')) & (
                (data_['bidvolume1'] != 0) | (data_['askvolume1'] != 0))]

        bid5VolSum = {f"bid5VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['bidvolume5sum'].tail(1)).sum()
                      for t_, T_r in close_price.items()}  # 不同时间点5挡委买量和
        ask5VolSum = {f"ask5VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['askvolume5sum'].tail(1)).sum()
                      for t_, T_r in close_price.items()}  # 不同时间点5挡委卖量和

        depth5Sum = {**bid5VolSum, **ask5VolSum, **{"date": date}}

        return pd.Series(depth5Sum)

    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
