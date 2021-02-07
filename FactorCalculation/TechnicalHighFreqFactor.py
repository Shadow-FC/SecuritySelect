import pandas as pd
import os
import time
import scipy.stats as st
import datetime as dt
import numpy as np
import sys
from pyfinance.ols import PandasRollingOLS

from FactorCalculation.FactorBase import FactorBase
from Object import FactorInfo
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN,
    FilePathName as FPN
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


class TechnicalHighFrequencyFactor(FactorBase):
    """
    高频因子
    """

    def __init__(self):
        super(TechnicalHighFrequencyFactor, self).__init__()

    ####################################################################################################################


if __name__ == '__main__':
    # path = 'A:\\数据'
    # file_name = 'factor.csv'
    # file_path = os.path.join(path, file_name)
    # Initiative_col = ['BuyAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_AM_120min', 'SaleAll_PM_120min', 'code', 'date']
    # df_stock = pd.read_csv(file_path, usecols=Initiative_col)

    A = TechnicalHighFrequencyFactor()
    # A.HighFreq056_data_raw()
    #
    # # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock.set_index('date', inplace=True)
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor()
    # A.momentum_in_day(df_stock)
    pass
