# -*-coding:utf-8-*-
# @Time:   2020/9/9 10:49
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import os
from typing import Callable, Dict, Any, Union

from DataAPI.DataInput.GetData import SQL, CSV
from constant import (
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    ExchangeName as EN
)


class FactorBase(object):

    def __init__(self):
        self.Q = SQL()
        self.CSV = CSV()
        self.list_date = self.CSV.list_date_csv()

    # 财务数据转换，需要考虑未来数据
    def _switch_freq(self,
                     data_: pd.DataFrame,
                     name: str,
                     limit: int = 120,
                     date_sta: str = '20130101',
                     date_end: str = '20200401',
                     exchange: str = EN.SSE.value) -> pd.Series:
        """

        :param data_:
        :param name: 需要转换的财务指标
        :param limit: 最大填充时期，默认二个季度
        :param date_sta:
        :param date_end:
        :param exchange:
        :return:
        """

        def _reindex(data: pd.DataFrame) -> pd.DataFrame:
            """填充有风险哦"""

            data_re = pd.merge(data, trade_date, on=KN.TRADE_DATE.value, how='outer')
            data_re[KN.STOCK_ID.value] = data_re[KN.STOCK_ID.value].ffill()
            return data_re

        sql_trade_date = self.Q.trade_date_SQL(date_sta=date_sta,
                                               date_end=date_end,
                                               exchange=exchange)
        trade_date = self.Q.query(sql_trade_date)

        # 保留最新数据
        data_sub = data_.groupby(KN.STOCK_ID.value,
                                 group_keys=False).apply(
            lambda x: x.sort_values(
                by=[KN.TRADE_DATE.value, SN.REPORT_DATE.value]).drop_duplicates(subset=[KN.TRADE_DATE.value],
                                                                                keep='last'))
        data_sub = data_sub.reset_index()

        # 交易日填充
        data_trade_date = data_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(_reindex)
        res = data_trade_date.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value]).sort_index()

        # 历史数据有限填充因子值
        res[name] = res[name].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: x.ffill(limit=limit))

        # if 'index' in res.columns:
        #     res = res.drop(columns='index')
        return res

    # 读取因子计算所需常用数据
    def _csv_data(self,
                  data_name: list,
                  file_path: str = FPN.Server_inputData.value,
                  file_name: str = "AStockData",
                  date: str = KN.TRADE_DATE.value,
                  stock_id: str = KN.STOCK_ID.value):
        res = pd.read_csv(os.path.join(file_path, file_name + '.csv'),
                          usecols=[date, stock_id] + data_name)
        return res

    # 读取指数数据
    def csv_index(self,
                  data_name: list,
                  file_path: str = '',
                  file_name: str = '',
                  index_name: str = '',
                  date: str = KN.TRADE_DATE.value, ):
        index_data = pd.read_csv(os.path.join(file_path, file_name + '.csv'), usecols=[date, 'code'] + data_name)
        res = index_data[index_data['code'] == index_name]
        return res

    # 读取分钟数据(数据不在一个文件夹中)，返回回调函数结果
    def csv_HFD_data(self,
                     data_name: list,
                     func: Callable = None,
                     fun_kwargs: dict = {},
                     file_path: str = '',
                     sub_file: str = '') -> Dict[str, Any]:
        if sub_file == '':
            Path = file_path
        elif sub_file == '1minute':
            Path = FPN.HFD_trade1min.value
        else:
            Path = os.path.join(file_path, sub_file)
        data_dict = {}
        file_names = os.listdir(Path)

        i = 1
        for file_name in file_names:
            i += 1
            if file_name[-3:] == 'csv':
                try:
                    # file_name = np.random.choice(file_names, 1)[0]  # 随机抽样
                    # file_name = '2017-03-30.csv'
                    data_df = pd.read_csv(os.path.join(Path, file_name), usecols=['code', 'time'] + data_name)
                except Exception as e:
                    continue
                data_df['date'] = file_name[:-4]
                # data_df.rename(columns={'code': 'stock_id'}, inplace=True)
                res = func(data_df, **fun_kwargs)
                data_dict[file_name[:-4]] = res
            # if i == 2:
            #     break

        return data_dict

    def _switch_ttm(self, data: pd.DataFrame, name: str) -> pd.Series:
        """
        计算TTM，groupby后要排序
        """

        data['M'] = data[SN.REPORT_DATE.value].apply(lambda x: x[5:7])

        dataNew = data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value]).sort_index()

        dataNew[name + '_TTM'] = dataNew.groupby(KN.STOCK_ID.value)[name].diff(1)
        dataNew[name + '_TTM'] = np.where(dataNew['M'] == '03', dataNew[name], dataNew[name + '_TTM'])
        res = dataNew.groupby(KN.STOCK_ID.value)[name + '_TTM'].apply(lambda x: x.rolling(4, min_periods=1).mean())

        res.name = name
        return res

    def reindex(self, data: Union[pd.DataFrame, pd.Series]) -> pd.Series:
        date_series = self.CSV.trade_date_csv()[KN.TRADE_DATE.value]
        date_index, code_index = data.index.levels[0], data.index.levels[1]
        date_sta, date_end = min(date_index), max(date_index)

        date_effect = date_series[(date_series >= date_sta) & (date_series <= date_end)]

        new_index = pd.MultiIndex.from_product([date_effect, code_index],
                                               names=[KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data.reindex(new_index.sort_values())

        return res

    def switchForm(self, data: pd.Series, n: int) -> pd.DataFrame:
        algo = {}
        for i_ in range(n):
            algo[f"col_{i_}"] = data.shift(i_)
        df_new = pd.DataFrame(algo).iloc[n - 1:]
        return df_new


if __name__ == '__main__':
    A = FactorBase()
    # A.csv_HFD_data()
