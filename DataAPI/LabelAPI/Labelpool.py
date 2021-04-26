import os
import sys
import time
import numpy as np
import pandas as pd
import datetime as dt
from typing import Dict
import pickle5 as pickle
from collections import defaultdict

from Object import (
    DataInfo
)

from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


class LabelAPI(object):

    def stock_ret(self,
                  stock_price: pd.DataFrame,
                  return_type: str = PVN.OPEN.value,
                  label: bool = True) -> pd.Series:
        """
        收益率作为预测标签需放置到前一天, 默认每个交易日至少存在一只股票价格，否则会出现收益率跳空计算现象
        :param stock_price: 股票价格表
        :param return_type: 计算收益率用到的股票价格
        :param label: 是否作为标签
        :return:
        """
        stock_price = stock_price.sort_index()
        result = stock_price[return_type].groupby(KN.STOCK_ID.value).pct_change(fill_method=None)
        if label:
            if return_type == PVN.OPEN.value:
                result = result.groupby(KN.STOCK_ID.value).shift(-2)
            else:
                result = result.groupby(KN.STOCK_ID.value).shift(-1)
        else:
            if return_type == PVN.OPEN.value:
                result = result.groupby(KN.STOCK_ID.value).shift(-1)

        result.name = KN.RETURN.value + return_type.capitalize()
        return result

    def industry_w(self,
                   index_weight: pd.Series,
                   industry_exposure: pd.Series) -> pd.Series:
        """
        生成行业权重
        如果某个行业权重为零则舍弃掉
        """
        data_ = pd.concat([index_weight, industry_exposure], axis=1).dropna()
        data_[SN.INDUSTRY_WEIGHT.value] = data_.groupby(KN.TRADE_DATE.value, group_keys=False).apply(
            lambda x: x[SN.STOCK_WEIGHT.value] / x[SN.STOCK_WEIGHT.value].sum())
        # industry weight
        ind_weight = data_.groupby([KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value])[SN.INDUSTRY_WEIGHT.value].sum()
        index_ = industry_exposure.index.get_level_values(KN.TRADE_DATE.value).drop_duplicates()
        ind_weight_new = ind_weight.unstack().reindex(index_).fillna(method='ffill').stack(dropna=False)
        ind_weight_new.name = SN.INDUSTRY_WEIGHT.value
        # fill weight and industry
        res_ = pd.merge(ind_weight_new.reset_index(), industry_exposure.reset_index(),
                        on=[KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value], how='right')
        res_ = res_.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value]).sort_index()

        return res_[SN.INDUSTRY_WEIGHT.value]

    def industry_mv(self,
                    index_weight: pd.Series,
                    industry_exposure: pd.Series,
                    mv: pd.Series) -> pd.Series:

        data_ = pd.concat([index_weight, mv, industry_exposure], axis=1)

        data_[SN.INDUSTRY_MV.value] = data_.groupby(KN.TRADE_DATE.value, group_keys=False).apply(
            lambda x: x[PVN.LIQ_MV.value] * x[SN.STOCK_WEIGHT.value] / x[SN.STOCK_WEIGHT.value].sum())
        # industry weight
        ind_mv = data_.groupby([KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value])[SN.INDUSTRY_MV.value].sum()
        index_ = industry_exposure.index.get_level_values(KN.TRADE_DATE.value).drop_duplicates()
        ind_weight_new = ind_mv.unstack().reindex(index_).fillna(method='ffill').stack(dropna=False)
        ind_weight_new.name = SN.INDUSTRY_MV.value
        # fill weight and industry
        res_ = pd.merge(ind_weight_new.reset_index(), industry_exposure.reset_index(),
                        on=[KN.TRADE_DATE.value, SN.INDUSTRY_FLAG.value], how='right')
        res_ = res_.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value]).sort_index()
        # 去除无效市值
        res_ = res_[res_[SN.INDUSTRY_MV.value] != 0]

        return res_[SN.INDUSTRY_MV.value]


class LabelPool(object):
    Mapping = {"price": {"file": 'AStockData.csv',
                         "columns": [PVN.CLOSE_ADJ.value, PVN.OPEN_ADJ.value, PVN.HIGH_ADJ.value, PVN.LOW_ADJ.value]},
               "industry": {"file": 'AStockData.csv',
                            "columns": [SN.INDUSTRY_FLAG.value]},
               "mv": {"file": 'AStockData.csv',
                      "columns": [PVN.LIQ_MV.value]},
               # "composition": {"file": 'IndexMember.csv',
               #                 "columns": [SN.CSI_300.value, SN.CSI_500.value, SN.CSI_800.value]},
               "stock_w1": {"file": "StockPool.csv",
                            "columns": [SN.STOCK_WEIGHT.value]},
               "stock_w2": {"file": "StockPool.csv",
                            "columns": ["stockPool", SN.STOCK_WEIGHT.value]},
               "priceLimit": {"file": "AStockData.csv",
                              "columns": [PVN.Up_Down.value]},
               "index_w": {"file": "StockPool.csv",
                           "columns": ["stockPool", "stockWeight"]},
               }

    def __init__(self):
        self.api = LabelAPI()
        self.path = FPN.Input_data_server.value
        self.local_path = FPN.Input_data_local.value

    def read_data(self) -> Dict[str, pd.DataFrame]:
        file = defaultdict(list)
        for label_name, label_info in self.Mapping.items():
            file[label_info['file']] += label_info['columns']

        data = {}
        for file_name, columns_list in file.items():
            data[file_name] = pd.read_csv(os.path.join(self.path, file_name),
                                          usecols=[KN.TRADE_DATE.value, KN.STOCK_ID.value] + columns_list,
                                          index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
        return data

    def merge_labels(self, **kwargs) -> pd.DataFrame:
        """
        :param kwargs: 标签数据
        :return:
        """
        res = pd.concat(kwargs.values(), axis=1)
        return res

    def strategyLabel(self) -> DataInfo:
        func_name = sys._getframe().f_code.co_name
        result_path = os.path.join(self.local_path, func_name + '.pkl')

        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                category_label = pickle.load(f)
        else:
            # read data
            print(f"{dt.datetime.now().strftime('%X')}: Construction the label pool")

            data_dict = self.read_data()
            price_data = data_dict['AStockData.csv'][self.Mapping["price"]['columns']]
            ind_exp = data_dict['AStockData.csv'][self.Mapping["industry"]['columns']]
            stock_mv_data = data_dict['AStockData.csv'][self.Mapping["mv"]['columns']] * 10000  # wan yuan->yuan
            # composition_data = data_dict['IndexMember.csv'][self.Mapping['composition']['columns']]
            stock_w_data = data_dict['StockPool.csv'][self.Mapping['stock_w1']['columns']]
            up_down_limit = data_dict['AStockData.csv'][self.Mapping['priceLimit']['columns']]

            price_data = price_data.rename(columns={PVN.CLOSE_ADJ.value: PVN.CLOSE.value,
                                                    PVN.OPEN_ADJ.value: PVN.OPEN.value,
                                                    PVN.HIGH_ADJ.value: PVN.HIGH.value,
                                                    PVN.LOW_ADJ.value: PVN.LOW.value})

            print(f"{dt.datetime.now().strftime('%X')}: Calculate stock daily return label")
            stock_ret_c = self.api.stock_ret(price_data, return_type=PVN.CLOSE.value)
            stock_ret_o = self.api.stock_ret(price_data, return_type=PVN.OPEN.value)

            print(f"{dt.datetime.now().strftime('%X')}: Set price limit label")
            up_down_limit = up_down_limit.fillna(1) == 0
            ############################################################################################################
            # merge labels
            print(f"{dt.datetime.now().strftime('%X')}: Merge labels")
            category_label = self.merge_labels(
                data_ret_close=stock_ret_c,
                data_ret_open=stock_ret_o,
                # composition=composition_data,
                ind_exp=ind_exp,
                mv=stock_mv_data[PVN.LIQ_MV.value],
                stock_w=stock_w_data,
                price_limit=up_down_limit
            )

            # sort
            category_label = category_label.sort_index()

            category_label.to_pickle(result_path)

        dataClass = DataInfo(data=category_label,
                             data_category=self.__class__.__name__,
                             data_name=func_name)
        return dataClass


if __name__ == '__main__':
    A = LabelPool()
    A.strategyLabel()
