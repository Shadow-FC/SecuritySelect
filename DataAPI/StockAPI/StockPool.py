import os
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt
import pickle5 as pickle

from Object import (
    DataInfo
)

from constant import (
    KeyName as KN,
    FilePathName as FPN,
    PriceVolumeName as PVN,
    BroadName as BN
)


class StockAPI(object):

    # ST股标识
    def ST(self,
           data: pd.Series) -> pd.Series(bool):
        """
        对ST,*ST股进行标记
        合格股票标记为True
        """
        return ~ data

    # 剔除某些板块
    def list_board(self,
                   data: pd.Series,
                   bord_name: str = BN.STIB.value) -> pd.Series(bool):
        """
        剔除科创板股
        """
        res = data != bord_name
        return res

    # 成立年限
    def established(self,
                    data: pd.Series,
                    days: int = 180) -> pd.Series(bool):
        """
        以真实成立时间进行筛选
        合格股票标记为True
        """
        res = data >= days
        return res

    # 交易规模
    def liquidity(self,
                  data: pd.Series,
                  days: int = 20,
                  amount_th: float = 1) -> pd.Series(bool):
        """
        默认标记过去20个交易日日均成交额小于1000万的股票
        合格股票标记为True
        amount_th: 单位为千
        """
        amount_mean = data.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(days, min_periods=1).mean())
        # 空值不参与计算
        res = amount_mean >= amount_th
        return res

    # 停牌
    def suspension(self,
                   data: pd.Series,
                   days: int = 20,
                   frequency: int = 5) -> pd.Series(bool):
        """
        1.当天停牌，下一天不交易
        2.连续20天发生5天停牌不交易
        以成交额为空作为停牌标识
        如果当天下午发生停牌，该标识有误
        前days天以第days天标识为主：若第days天为5，则前days天都为5
        合格股票标记为True
        """

        J = data.isnull()
        suspension_days = J.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(days, min_periods=days).sum())
        suspension_days = suspension_days.groupby(KN.STOCK_ID.value).bfill()
        res = pd.DataFrame(dict(J1=suspension_days <= frequency, J2=~ J))
        return res['J1'] & res['J2']

    # 流通市值
    def liq_mv(self,
               data: pd.Series,
               proportion: float = 0.02) -> pd.Series(bool):
        """
        剔除流通市值最小的2%股票
        """
        res = data.groupby(KN.TRADE_DATE.value).apply(lambda x: x.gt(x.quantile(proportion)))
        return res

    def price_limit(self,
                    data: pd.Series):
        """
        当天涨跌停，下一天停止交易
        若标识为空，则默认为涨跌停股票（该类股票一般为退市股或ST股等异常股）
        合格股票标记为True
        :param data:
        :return:
        """
        res = data.fillna(1) == 0
        return res


class StockPool(object):
    """
    股票池属性名称最好与文件名名称保持一致
    股票池返回数据类型为pandas.Index
    """
    Mapping = {"StockPoolZD": {"file": 'AStockData.csv',
                               "columns": [PVN.ISST.value, PVN.LIST_DAYS_NUM.value,
                                           PVN.AMOUNT.value, PVN.LIST_BOARD.value,
                                           PVN.LIQ_MV.value]},
               }

    def __init__(self):
        self.api = StockAPI()  # 股票池API接口
        self.path = FPN.Server_inputData.value  # 原始数据路径
        self.local_path = FPN.Local_inputData.value  # 历史股票池数据路径

    def read_data(self, pool_name: str) -> pd.DataFrame:
        data = pd.read_csv(os.path.join(self.path, self.Mapping[pool_name]['file']),
                           usecols=[KN.TRADE_DATE.value, KN.STOCK_ID.value] + self.Mapping[pool_name]['columns'],
                           index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        return data

    def StockPoolZD(self) -> DataInfo:
        """
        1.剔除ST股：是ST为True
        2.剔除成立年限小于6个月的股票：成立年限小于6个月为False
        3.过去5天成交额占比排名最后5%：成交额占比在最后5%为False
        4.过去5天停牌天数超过3天:停牌数超过阈值为False

        注意：函数名需要与数据源文件名对应，保持一致防止出错，可自行修改
        :return:
        """
        func_name = sys._getframe().f_code.co_name
        result_path = os.path.join(self.local_path, func_name + '.pkl')

        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                stock_judge = pickle.load(f)
        else:
            # read data
            print(f"{dt.datetime.now().strftime('%X')}: Read the data of stock pool")
            data_input = self.read_data(func_name)

            # get filter condition
            print(f"{dt.datetime.now().strftime('%X')}: Weed out ST stock")
            data_input['judge1'] = self.api.ST(data_input[PVN.ISST.value])

            # print(f"{dt.datetime.now().strftime('%X')}: Weed out Price Up_Down limit stock")
            # data_input['judge2'] = self.price_limit(data_input[PVN.Up_Down.value])

            print(f"{dt.datetime.now().strftime('%X')}: Weed out stock Established in less than 6 months")
            data_input['judge3'] = self.api.established(data=data_input[PVN.LIST_DAYS_NUM.value])

            print(f"{dt.datetime.now().strftime('%X')}: Weed out stock illiquidity")
            data_input['judge4'] = self.api.liquidity(data_input[PVN.AMOUNT.value])

            print(f"{dt.datetime.now().strftime('%X')}: Weed out Suspension stock")
            data_input['judge5'] = self.api.suspension(data_input[PVN.AMOUNT.value])

            print(f"{dt.datetime.now().strftime('%X')}: Weed out liq_mv less than 0.02 stock")
            data_input['judge6'] = self.api.liq_mv(data_input[PVN.LIQ_MV.value])

            print(f"{dt.datetime.now().strftime('%X')}: Weed out science and technology innovation board stock")
            data_input['judge7'] = self.api.list_board(data_input[PVN.LIST_BOARD.value])

            Judge = [column for column in data_input.columns if 'judge' in column]

            # Filter
            stock_judge = data_input[Judge].all(axis=1).sort_index()
            stock_judge.name = func_name
            # to_csv
            stock_judge.to_pickle(result_path)

        # switch judge to label
        stock_pool = stock_judge.groupby(KN.STOCK_ID.value).shift(1).fillna(True)
        dataClass = DataInfo(data=stock_pool[stock_pool],
                             data_category=self.__class__.__name__,
                             data_name=func_name)
        return dataClass


if __name__ == '__main__':
    path = "A:\\数据\\StockPool"
    # stock_pool = pd.read_csv("A:\\数据\\StockPool.csv")

    # Data cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # stock_pool[price_columns] = stock_pool[price_columns].multiply(stock_pool['adjfactor'], axis=0)
    # df_stock.set_index('date', inplace=True)
    A = StockPool()
    A.StockPoolZD()
