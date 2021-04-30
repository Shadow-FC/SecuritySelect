# -*-coding:utf-8-*-
# @Time:   2020/9/4 14:11
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import statsmodels.api as sm
import time
from statsmodels.tsa.arima_model import ARMA
from utility.FactorUtility import MethodSets

from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN
)

from utility.utility import (
    timer
)


class ReturnForecast(MethodSets):

    def __init__(self):
        super(ReturnForecast, self).__init__()

    # 收益预测1
    @timer
    def Return_Forecast1(self,
                         factor: pd.DataFrame = None,
                         **kwargs):
        """
        当期因子暴露与下期个股收益流通市值加权最小二乘法回归得到下期因子收益预测值
        下期因子收益预测值与下期因子暴露相乘得到个股收益作为当天对下期的预测值
        对于当期存在因子缺失不做完全删除，以当期有效因子进行计算
        :return:
        """
        data_sub = pd.concat([self.holding_ret, self.ind_exp, self.mv], axis=1, join='inner')
        data_input = pd.merge(data_sub, factor, left_index=True, right_index=True, how='left')

        # 因子收益预测
        reg_res = data_input.groupby(KN.TRADE_DATE.value).apply(self.WLS, 100)

        fact_ret_fore_ = pd.DataFrame(map(lambda x: x.params, reg_res), index=reg_res.index)  # 因子收益预测值

        # 当天预测值，收盘价需要+1, 若当期因子收益率和因子暴露全为空则去除, 否则计算会设置默认值0！
        fact_ret_fore = fact_ret_fore_.shift(self.hp + 1).dropna(how='all').fillna(0)

        fact_exp_c = self.fac_exp.dropna(how='all').fillna(0).copy(deep=True)
        fact_ = pd.concat([fact_exp_c, pd.get_dummies(self.ind_exp)], axis=1, join='inner')

        # 个股收益预测
        asset_ret_fore = fact_.groupby(KN.TRADE_DATE.value,
                                       group_keys=False).apply(
            lambda x: (x * fact_ret_fore.loc[x.index[0][0], x.columns]).sum(axis=1)
            if x.index[0][0] in fact_ret_fore.index else pd.Series(index=x.index))

        asset_ret_fore.dropna(inplace=True)

        self.OPT_params['ASSET_RET_FORECAST'] = asset_ret_fore.unstack().T.to_dict('series')

    # 收益预测2
    @timer
    def Return_Forecast2(self,
                         method: str = 'EWMA',
                         **kwargs):

        # 因子收益预测
        fact_ret_fore_ = getattr(self.RET, method)(self.df_input['FACT_RET'], **kwargs)

        # 当天预测值， 收盘价需要+1, 若当期因子收益率和因子暴露全为空则去除，否则nan填充为0！
        fact_ret_fore = fact_ret_fore_.shift(self.hp + 1).dropna(how='all').fillna(0)
        fact_exp_c = self.fac_exp.dropna(how='all').fillna(0).copy(deep=True)
        fact_ = pd.concat([fact_exp_c, pd.get_dummies(self.ind_exp)], axis=1, join='inner')

        # 个股收益预测
        asset_ret_fore = fact_.groupby(KN.TRADE_DATE.value,
                                       group_keys=False).apply(
            lambda x: (x * fact_ret_fore.loc[x.index[0][0], x.columns]).sum(axis=1)
            if x.index[0][0] in fact_ret_fore.index else pd.Series(index=x.columns))

        asset_ret_fore.dropna(inplace=True)
        try:
            self.OPT_params['ASSET_RET_FORECAST'] = asset_ret_fore.unstack().T.to_dict('series')
        except Exception as e:
            print(e)

    pass

