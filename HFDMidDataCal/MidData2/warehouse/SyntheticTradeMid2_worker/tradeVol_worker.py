# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import range_T


def tradeVol_worker(data: pd.DataFrame,
                    date: str,
                    **kwargs) -> pd.DataFrame:

    def cal(d: pd.DataFrame) -> pd.Series:
        d['volPerTrade'] = d['volume'] / d['tradenum']
        d_sub = d[range_T]

        volDiff = d_sub['volume'].diff(1)
        volPerDiff = d_sub['volPerTrade'].diff(1)

        vol_data = pd.Series({

            "volDiffMean": volDiff.mean(),  # 成交量差分均值
            "volDiffStd": volDiff.std(),  # 成交量差分标准差

            "volDiffAbsMean": abs(volDiff).mean(),  # 成交量差分绝对值均值
            "volDiffAbsStd": abs(volDiff).std(),  # 成交量差分绝对值标准差

            "volPerMean": d_sub['volPerTrade'].mean(),  # 每笔成交量均值
            "volPerStd": d_sub['volPerTrade'].std(),  # 每笔成交量标准差

            "volPerDiffMean": volPerDiff.mean(),  # 每笔成交量差分均值
            "volPerDiffStd": volPerDiff.std(),  # 每笔成交量差分标准差

            "volPerDiffAbsMean": abs(volPerDiff).mean(),  # 每笔成交量差分绝对值均值
            "volPerDiffAbsStd": abs(volPerDiff).std(),  # 每笔成交量差分绝对值标准差

            "date": date
        })

        return pd.Series(vol_data)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
