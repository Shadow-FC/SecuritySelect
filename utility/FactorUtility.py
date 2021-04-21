# -*-coding:utf-8-*-
# @Time:   2021/4/14 13:43
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import numpy as np
import os
import json
import statsmodels.api as sm
import copy
import time
import datetime as dt
from typing import Dict, Any
from sklearn.decomposition import PCA
from sklearn import linear_model

from Optimization import MaxOptModel
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


# *去极值*
class RemoveOutlier(object):
    dataName = ''

    def process(self,
                data: pd.Series(float),
                method='before_after_3%',
                **kwargs) -> pd.Series(float):

        method_dict = {
            "before_after_3%": self.before_after_n,
            "before_after_3sigma": self.before_after_3sigma,
            "mad": self.mad
        }
        if method is None:
            return data
        else:
            res = method_dict[method](data, **kwargs)
            return res

    """去极值"""

    # 前后3%
    def before_after_n(self,
                       data: pd.DataFrame,
                       n: int = 3,
                       **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        threshold_down, threshold_up = data_df.quantile(n / 100, axis=1), data_df.quantile(1 - n / 100, axis=1)
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res

    # 3倍标准差外
    def before_after_3sigma(self,
                            data: pd.DataFrame,
                            **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        miu, sigma = data_df.mean(axis=1), data_df.std(axis=1)
        threshold_down, threshold_up = miu - 3 * sigma, miu + 3 * sigma
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res

    # 绝对中位偏差法
    def mad(self,
            data: pd.DataFrame,
            **kwargs) -> pd.Series(float):
        data_df = data[self.dataName].unstack()
        median = data_df.median(axis=1)
        MAD = data_df.sub(median, axis=0).abs().median(axis=1)
        threshold_up, threshold_down = median + 3 * 1.483 * MAD, median - 3 * 1.483 * MAD
        res = data_df.clip(threshold_down, threshold_up, axis=0).stack()
        return res


# *中性化*
class Neutralization(object):
    dataName = ''

    def process(self,
                data: pd.Series,
                mvName: str = PVN.LIQ_MV.value,
                indName: str = SN.INDUSTRY_FLAG.value,
                method: str = 'industry+mv',
                **kwargs) -> pd.Series(float):
        """
        若同时纳入行业因子和市值因子需要加上截距项，若仅纳入行业因子则回归方程不带截距项！
        :param data: 因子数据
        :param mvName: 市值名称
        :param indName: 行业指数名称
        :param method: 中心化方法
        :return: 剔除行业因素和市值因素后的因子

        Args:
            indName ():
            mvName ():
        """

        colName = [self.dataName]
        # read mv and industry data
        if 'mv' in method:
            colName.append(mvName)

        if 'industry' in method:
            colName.append(indName)

        # remove Nan
        dataNew = data[colName].dropna(how='any').copy()
        # neutralization
        res = dataNew.groupby(KN.TRADE_DATE.value, group_keys=False).apply(self.reg)
        return res

    # regression
    def reg(self, data: pd.DataFrame) -> pd.Series(float):
        """！！！不排序回归结果会不一样！！！"""
        dataSub = data.sort_index()

        X = pd.get_dummies(dataSub.loc[:, dataSub.columns != self.dataName], columns=[SN.INDUSTRY_FLAG.value])
        Y = dataSub[self.dataName]
        reg = np.linalg.lstsq(X, Y)
        factNeu = Y - (reg[0] * X).sum(axis=1)
        return factNeu


# *标准化*
class Standardization(object):
    dataName = ''

    def process(self,
                data: pd.DataFrame,
                method='z_score',
                **kwargs) -> pd.Series(float):

        method_dict = {"range01": self.range01,
                       "z_score": self.z_score,
                       "mv": self.market_value_weighting
                       }

        if method is None:
            return data

        res = method_dict[method](data, **kwargs)

        return res

    """标准化"""

    # 标准分数法
    def z_score(self,
                data: pd.DataFrame,
                **kwargs):
        """
        :param data:
        :return:
        """
        data_df = data[self.dataName].unstack()
        miu, sigma = data_df.mean(axis=1), data_df.std(axis=1)
        stand = data_df.sub(miu, axis=0).div(sigma, axis=0).stack()
        return stand

    def range01(self,
                data: pd.DataFrame,
                **kwargs):
        data_df = data[self.dataName]
        denominator, numerator = data_df.max(axis=1) - data_df.min(axis=1), data_df.div(data_df.min(axis=1), axis=0)
        result = numerator.div(denominator, axis=0).stack()
        return result

    # 市值加权标准化
    def market_value_weighting(self,
                               data: pd.DataFrame,
                               mvName: str = PVN.LIQ_MV.value,
                               **kwargs) -> pd.Series(float):
        data_df = data[[self.dataName, mvName]].dropna(how='any')
        dataFact = data_df[self.dataName].unstack()
        dataMv = data_df[mvName].unstack()

        miu, std = (dataFact * dataMv).div(dataMv.sum(axis=1), axis=0).sum(axis=1), dataFact.std(axis=1)
        res = dataFact.sub(miu, axis=0).div(std, axis=0).stack()

        return res

#  因子预处理
# class FactorProcess(object):
#     """
#     去极值，标准化，中性化，分组
#     """
#     data_name = {'mv': 'AStockData.csv',
#                  'industry': 'AStockData.csv'}
#
#     def __init__(self):
#         self.fact_name = ''
#         self.raw_data = {}
#
#     # *去极值*
#     def remove_outliers(self,
#                         data: pd.Series,
#                         method='before_after_3%') -> pd.Series:
#
#         self.fact_name = data.name
#
#         method_dict = {
#             "before_after_3%": self.before_after_n,
#             "before_after_3sigma": self.before_after_3sigma,
#             "mad": self.mad
#         }
#         if method is None:
#             return data
#         else:
#             res = data.groupby(KN.TRADE_DATE.value).apply(method_dict[method])
#             return res
#
#     # *中性化*
#     def neutralization(self,
#                        data: pd.Series,
#                        method: str = 'industry+mv') -> pd.Series:
#         """
#         若同时纳入行业因子和市值因子需要加上截距项，若仅纳入行业因子则回归方程不带截距项！
#         :param data: 因子数据
#         :param method: 中心化方法
#         :return: 剔除行业因素和市值因素后的因子
#         """
#
#         self.fact_name = data.name
#
#         # regression TODO 判断空值即可
#         def _reg(data_: pd.DataFrame) -> pd.Series:
#             """！！！不排序回归结果会不一样！！！"""
#             data_sub_ = data_.sort_index()
#
#             if data_sub_.shape[0] < data_sub_.shape[1]:
#                 fact_neu = pd.Series(data=np.nan, index=data_.index)
#             else:
#                 X = pd.get_dummies(data_sub_.loc[:, data_sub_.columns != self.fact_name],
#                                    columns=[SN.INDUSTRY_FLAG.value])
#                 Y = data_sub_[self.fact_name]
#                 reg = np.linalg.lstsq(X, Y)
#                 fact_neu = Y - (reg[0] * X).sum(axis=1)
#             fact_neu.name = self.fact_name
#             return fact_neu
#
#         # read mv and industry data
#         if 'mv' in method:
#             if self.raw_data.get('mv', None) is None:
#                 mv_data = pd.read_csv(os.path.join(FPN.Input_data_server.value, self.data_name['mv']),
#                                       index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value],
#                                       usecols=[KN.TRADE_DATE.value, KN.STOCK_ID.value, PVN.LIQ_MV.value])
#                 self.raw_data['mv'] = mv_data
#             else:
#                 mv_data = self.raw_data['mv']
#
#         else:
#             mv_data = pd.DataFrame()
#
#         if 'industry' in method:
#             if self.raw_data.get('industry', None) is None:
#                 industry_data = pd.read_csv(os.path.join(FPN.Input_data_server.value, self.data_name['industry']),
#                                             index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
#                 self.raw_data['industry'] = industry_data
#             else:
#                 industry_data = self.raw_data['industry']
#         else:
#             industry_data = pd.DataFrame()
#
#         # merge data
#         neu_factor = pd.concat([data, mv_data, industry_data], axis=1).dropna()
#
#         # neutralization
#
#         res = neu_factor.groupby(KN.TRADE_DATE.value, group_keys=False).apply(_reg)
#         return res
#
#     # *标准化*
#     def standardization(self,
#                         data: pd.Series,
#                         method='z_score') -> pd.Series:
#
#         method_dict = {"range01": self.range01,
#                        "z_score": self.z_score,
#                        "mv": self.market_value_weighting
#                        }
#         self.fact_name = data.name
#
#         if method is None:
#             return data
#         elif method == 'mv':
#             if self.raw_data.get('mv', None) is None:
#                 mv_data = pd.read_csv(os.path.join(FPN.Input_data_server.value, self.data_name['mv']),
#                                       index_col=['date', 'stock_id'],
#                                       usecols=['date', 'stock_id', 'liq_mv'])
#                 self.raw_data['mv'] = mv_data
#             else:
#                 mv_data = self.raw_data['mv']
#
#             stand_data = pd.concat([data, mv_data], axis=1, join='inner')
#         else:
#             stand_data = data
#
#         res = stand_data.groupby(KN.TRADE_DATE.value, group_keys=False).apply(method_dict[method])
#         return res
#
#     # # *正交化*
#     # @staticmethod
#     # def orthogonal(factor_df, method='schimidt'):
#     #     # 固定顺序的施密特正交化
#     #     def schimidt():
#     #
#     #         col_name = factor_df.columns
#     #         factors1 = factor_df.values
#     #
#     #         R = np.zeros((factors1.shape[1], factors1.shape[1]))
#     #         Q = np.zeros(factors1.shape)
#     #         for k in range(0, factors1.shape[1]):
#     #             R[k, k] = np.sqrt(np.dot(factors1[:, k], factors1[:, k]))
#     #             Q[:, k] = factors1[:, k] / R[k, k]
#     #             for j in range(k + 1, factors1.shape[1]):
#     #                 R[k, j] = np.dot(Q[:, k], factors1[:, j])
#     #                 factors1[:, j] = factors1[:, j] - R[k, j] * Q[:, k]
#     #
#     #         Q = pd.DataFrame(Q, columns=col_name, index=factor_df.index)
#     #         return Q
#     #
#     #     # 规范正交
#     #     def canonial():
#     #         factors1 = factor_df.values
#     #         col_name = factor_df.columns
#     #         D, U = np.linalg.eig(np.dot(factors1.T, factors1))
#     #         S = np.dot(U, np.diag(D ** (-0.5)))
#     #
#     #         Fhat = np.dot(factors1, S)
#     #         Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)
#     #
#     #         return Fhat
#     #
#     #     # 对称正交
#     #     def symmetry():
#     #         col_name = factor_df.columns
#     #         factors1 = factor_df.values
#     #         D, U = np.linalg.eig(np.dot(factors1.T, factors1))
#     #         S = np.dot(U, np.diag(D ** (-0.5)))
#     #
#     #         Fhat = np.dot(factors1, S)
#     #         Fhat = np.dot(Fhat, U.T)
#     #         Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)
#     #
#     #         return Fhat
#     #
#     #     method_dict = {
#     #         "schimidt": schimidt(),
#     #         "canonial": canonial(),
#     #         "symmetry": symmetry()
#     #     }
#     #
#     #     return method_dict[method]
#     #
#
#     """去极值"""
#
#     # 前后3%
#     @staticmethod
#     def before_after_n(data: pd.Series, n: int = 3):
#         length = len(data)
#         sort_values = data.sort_values()
#         threshold_top = sort_values.iloc[int(length * n / 100)]
#         threshold_down = sort_values.iloc[-(int(length * n / 100) + 1)]
#         data[data <= threshold_top] = threshold_top
#         data[data >= threshold_down] = threshold_down
#         return data
#
#     # 3倍标准差外
#     @staticmethod
#     def before_after_3sigma(data: pd.Series) -> pd.Series:
#         miu = data.mean()
#         sigma = data.std()
#         threshold_down = miu - 3 * sigma
#         threshold_up = miu + 3 * sigma
#         data[data.ge(threshold_up)] = threshold_up
#         data[data.le(threshold_down)] = threshold_down
#         return data
#
#     # 绝对中位偏差法
#     @staticmethod
#     def mad(data):
#         median = data.median()
#         MAD = (data - median).abs().median()
#         threshold_up = median + 3 * 1.483 * MAD
#         threshold_down = median - 3 * 1.483 * MAD
#         data[data >= threshold_up] = threshold_up
#         data[data <= threshold_down] = threshold_down
#         return data
#
#     """标准化"""
#
#     # 标准分数法
#     @staticmethod
#     def z_score(data: pd.Series):
#         """
#         :param data:
#         :return:
#         """
#         miu = data.mean()
#         sigma = data.std()
#         stand = (data - miu) / sigma
#         return stand
#
#     @staticmethod
#     def range01(data: pd.Series):
#         result = (data - data.min()) / (data.max() - data.min())
#         return result
#
#     # 市值加权标准化
#     def market_value_weighting(self, data) -> pd.Series:
#         data_sub = data.dropna(how='any')
#
#         if data_sub.empty:
#             stand = pd.Series(data=np.nan, index=data.index)
#         else:
#
#             factor = data_sub[self.fact_name]
#             mv = data_sub[PVN.LIQ_MV.value]
#
#             sum_mv = sum(mv)
#             miu = sum(data_sub.prod(axis=1, skipna=False)) / sum_mv
#
#             sigma = factor.std()
#             stand = (factor - miu) / sigma
#         stand.name = self.fact_name
#         return stand
#
#     # 分组
#     @staticmethod
#     def grouping(data: pd.DataFrame, n):
#         """
#         1.假设样本量为M,将因子分成N组，前N-1组有效样本量为int(M/N),最后一组有效样本量为M-(N-1)*int(M/*N);
#         2.无效样本不参与计算;
#         3.相同排序定义为同一组;
#         4.相同排序后下一元素连续不跳级
#         5.升序排列
#         :param data:
#         :param n:分组个数
#         :return:
#         """
#         rank_data = data.rank(axis=1, ascending=True, method='dense')
#         effect_data = rank_data.max(axis=1)
#         amount_each_group = effect_data // n
#         data_group = rank_data.floordiv(amount_each_group, axis=0) + np.sign(rank_data.mod(amount_each_group, axis=0))
#         data_group[data_group > n] = n
#         return data_group
