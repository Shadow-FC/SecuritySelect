import os
import sys
import time
import numpy as np
import pandas as pd
import pickle5 as pickle
from typing import Union, Dict, Any

from DataAPI.FactorAPI.FactorBase import FactorBase
from utility.utility import (
    Process
)
from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN
)

"""
高频数据
高频数据需要合成中间过程
然后利用中间过程计算因子
数据命名以合成的数据名为主，没有统一定义

1分钟频率收益率计算采用收盘价比上开盘价(分钟数据存在缺失，采用开盘或者收盘直接计算容易发生跳空现象)
2h 数据存在异常，在原数据中进行剔除
若不做特殊说明， 分钟级别数据运算会包含集合竞价信息
"""


class HighFrequencyStrengthFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(HighFrequencyStrengthFactor, self).__init__()

    @classmethod
    def Strength001(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """高频波动(HFD_std_ret)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Strength001_data_raw(cls,
                             n: int = 5,
                             **kwargs) -> pd.Series(float):
        """买单强弱()"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"
        with open(r"D:\DataBase\MidData2\tradeMarket.pkl", 'rb') as f:
            data1 = pickle.load(f)

        with open(r"D:\DataBase\AStockData.pkl", 'rb') as f:
            data2 = pickle.load(f)
        data2['ret'] = data2.groupby('code')['close'].pct_change(fill_method=None)
        data = pd.merge(data1, data2[['date', 'code', 'liqMv', 'ret']], on=['date', 'code'], how='left')

        for i in facts:
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])[i.split('.')[0]]
            res = cls().reindex(data)
            res = res.reset_index()
            res.to_pickle(r'D:\DataBase\HighFrequencyStrengthFactor\{}'.format(i))
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = pow(data['ret2Up_0'] + data['ret2Down_0'], 0.5)

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res


if __name__ == '__main__':
    pass
