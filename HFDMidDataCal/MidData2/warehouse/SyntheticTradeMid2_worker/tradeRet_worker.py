# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import range_T


def tradeRet_worker(data: pd.DataFrame,
                    date: str,
                    **kwargs) -> pd.DataFrame:

    def cal(data_: pd.DataFrame) -> pd.Series:
        d_sub = data_[range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        retDiff = d_sub['ret'].diff(1)
        retData = {
            "retDiffMean": retDiff.mean(),  # 收益率差分均值
            "retDiffAbsMean": abs(retDiff).mean(),  # 收益率差分绝对值均值

            "retMean": d_sub['ret'].mean(),  # 收益率均值
            "retAbsMean": abs(d_sub['ret']).mean(),  # 收益率绝对值均值

            "ret2Up_0": pow(d_sub['ret'][d_sub['ret'] > 0], 2).sum(),  # 收益率大于0的平方和
            "ret2Down_0": pow(d_sub['ret'][d_sub['ret'] < 0], 2).sum(),  # 收益率小于0的平方和

            "ret3Up_0": pow(d_sub['ret'][d_sub['ret'] > 0], 3).sum(),  # 收益率大于0的三次方和
            "ret3Down_0": pow(d_sub['ret'][d_sub['ret'] < 0], 3).sum(),  # 收益率小于0的三次方和

            "retVolWeight": (d_sub['ret'] * d_sub['volume']).sum() / d_sub['volume'].sum(),  # 成交量加权收益

            "retVar": d_sub['ret'].var(),  # 收益率方差
            "retSkew": d_sub['ret'].skew(),  # 收益率偏度
            "retKurt": d_sub['ret'].kurt(),  # 收益率峰度

            "date": date
        }
        return pd.Series(retData)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
