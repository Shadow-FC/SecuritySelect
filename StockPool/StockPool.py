import pandas as pd
import datetime as dt
import numpy as np
import os
import sys
import time
from functools import reduce

from SecuritySelect.constant import (
    KeyName as KN,
    FilePathName as FPN,
    PriceVolumeName as PVN,
)


class StockPool(object):
    """
    股票池属性名称最好与文件名名称保持一致
    股票池返回数据类型为pandas.Index
    """

    def __init__(self):
        self.index_list = []
        pass

    # ST股标识
    def ST(self,
           data: pd.DataFrame,
           st: str = 'ST') -> pd.Series(bool):
        """
        对ST,*ST股进行标记
        合格股票标记为True
        """
        res = data[st]
        # 没有标记的记为正常股
        res.fillna(False, inplace=True)
        self.index_list.append(res.index)
        return ~ res

    # 成立年限
    def established(self,
                    data: pd.DataFrame,
                    listdate: str = 'listdate',
                    days: int = 90) -> pd.Series(bool):
        """
        以真实成立时间进行筛选
        合格股票标记为True
        """

        list_date = data[listdate].apply(lambda x: dt.datetime(year=int(str(x)[0:4]),
                                                               month=int(str(x)[4:6]),
                                                               day=int(str(x)[6:8])))

        res = (dt.datetime.now() - list_date) > np.timedelta64(days)

        self.index_list.append(res.index)
        return res

    # 交易规模
    def liquidity(self,
                  data: pd.DataFrame,
                  amount_name: str = PVN.AMOUNT.value,
                  days: int = 5,
                  proportion: float = 0.05) -> pd.Series(bool):
        """
        默认标记过去5个交易日日均成交额占比在后5%的股票
        合格股票标记为True"""
        amount_mean = data[amount_name].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(days).mean())

        # 空值不参与计算
        res = amount_mean.groupby(KN.TRADE_DATE.value).apply(lambda x: x.gt(x.quantile(proportion)))

        self.index_list.append(res.index)
        return res

    # 停牌
    def suspension(self,
                   data: pd.DataFrame,
                   amount_name: str = PVN.AMOUNT.value,
                   days: int = 5,
                   frequency: int = 3) -> pd.Series(bool):
        """
        以成交额为空作为停牌标识
        如果当天下午发生停牌，该标识有误
        前days天以第days天标识为主：若第days天为5，则前days天都为5
        合格股票标记为True
        """

        trade_days = data[amount_name].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(days, min_periods=days).count().fillna(method='bfill'))

        res = trade_days > days - frequency

        self.index_list.append(res.index)
        return res

    def StockPool1(self) -> pd.Index:
        """
        1.剔除ST股：是ST为True
        2.剔除成立年限小于6个月的股票：成立年限小于6个月为False
        3.过去5天成交额占比排名最后5%：成交额占比在最后5%为False
        4.过去5天停牌天数超过3天:停牌数超过阈值为False

        注意：函数名需要与数据源文件名对应，保持一致防止出错，可自行修改
        :return:
        """
        result_path = os.path.join(FPN.stock_pool_path.value, sys._getframe().f_code.co_name + '_result.csv')
        if os.path.exists(result_path):
            index_effect_stock = pd.read_csv(result_path, index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value]).index
        else:
            # get data file path
            data_address = os.path.join(FPN.stock_pool_path.value, sys._getframe().f_code.co_name + '.csv')

            # read data
            print(f"{dt.datetime.now().strftime('%X')}: Read the data of stock pool")
            data_input = pd.read_csv(data_address)
            data_input.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value], inplace=True)

            # get filter condition
            print(f"{dt.datetime.now().strftime('%X')}: Weed out ST stock")
            self.ST(data_input)

            print(f"{dt.datetime.now().strftime('%X')}: Weed out stock established in less than 3 months")
            self.established(data=data_input, days=90)

            print(f"{dt.datetime.now().strftime('%X')}: Weed out stock illiquidity")
            self.liquidity(data_input)

            print(f"{dt.datetime.now().strftime('%X')}: Weed out suspension stock")
            self.suspension(data_input)

            # Filter
            index_effect_stock = reduce(lambda x, y: x.intersection(y), self.index_list)

            # Sort
            index_effect_stock = index_effect_stock.sort_values()
            # to_csv
            index_effect_stock.to_frame().to_csv(result_path, index=False)
        return index_effect_stock


if __name__ == '__main__':
    path = "A:\\数据\\StockPool"
    # stock_pool = pd.read_csv("A:\\数据\\StockPool.csv")

    # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # stock_pool[price_columns] = stock_pool[price_columns].multiply(stock_pool['adjfactor'], axis=0)
    # df_stock.set_index('date', inplace=True)
    A = StockPool()
    A.StockPool1()
