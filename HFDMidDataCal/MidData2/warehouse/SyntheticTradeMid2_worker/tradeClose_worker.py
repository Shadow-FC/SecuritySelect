# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import close_price, range_T


def tradeClose_worker(data: pd.DataFrame,
                      date: str,
                      **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d_sub = d[range_T]

        # 收盘价相关中间过程
        closeData = {"close" + t_: 0 if d[d['time'] <= T_r].tail(1)['close'].empty
        else d[d['time'] <= T_r].tail(1)['close'].values[0] for t_, T_r in close_price.items()}  # 不同时间截面收盘价

        closeData.update({
            "closeMean": d_sub['close'].mean(),  # 收盘价均值
            "closeStd": d_sub['close'].std(),  # 收盘价标准差
            "closeAmtWeight": (d_sub['close'] * d_sub['amount']).sum() / d_sub['amount'].sum(),  # 成交量加权收盘价
            "date": date  # 日期
        })

        return pd.Series(closeData)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
