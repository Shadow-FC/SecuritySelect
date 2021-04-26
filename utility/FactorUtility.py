# -*-coding:utf-8-*-
# @Time:   2021/4/14 13:43
# @Author: FC
# @Email:  18817289038@163.com

import time
import datetime as dt
import pandas as pd
import numpy as np

from scipy import stats
from typing import Dict, Any, List
from itertools import combinations_with_replacement
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score as nor_mi

from utility.Optimization import MaxOptModel
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN
)


# *去极值*
class RemoveOutlier(object):
    dataName = ''

    def process(self,
                data: pd.DataFrame,
                dataName: str,
                method='before_after_3%',
                **kwargs) -> pd.Series(float):

        self.dataName = dataName

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
                dataName: str,
                mvName: str = PVN.LIQ_MV.value,
                indName: str = SN.INDUSTRY_FLAG.value,
                method: str = 'industry+mv',
                **kwargs) -> pd.Series(float):
        """
        若同时纳入行业因子和市值因子需要加上截距项，若仅纳入行业因子则回归方程不带截距项！
        :param data: 因子数据
        :param dataName: 因子数据
        :param mvName: 市值名称
        :param indName: 行业指数名称
        :param method: 中心化方法
        :return: 剔除行业因素和市值因素后的因子

        Args:
            indName ():
            mvName ():

        Parameters
        ----------
        factName :
        """
        self.dataName = dataName

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
                dataName: str,
                method='z_score',
                **kwargs) -> pd.Series(float):
        self.dataName = dataName

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


# *相关性检验*
class Correlation(object):

    def process(self,
                data: pd.DataFrame,
                method='LinCor',
                **kwargs) -> pd.Series(float):

        method_dict = {"LinCor": self.correlation,
                       "MI": self.MI,
                       }

        if method is None:
            return data

        res = method_dict[method](data, **kwargs)

        return res

    # 线性相关性检验
    def correlation(self,
                    data: pd.DataFrame,
                    corName: str,
                    **kwargs) -> Dict[str, Any]:
        """
        每期有效数据过少不计算相关性
        Parameters
        ----------
        data :
        corName :
        kwargs :

        Returns
        -------

        """
        dataSub = data.dropna()

        df_cor = dataSub.groupby(KN.TRADE_DATE.value).corr(method=corName)

        cor_GroupBy = df_cor.groupby(pd.Grouper(level=-1))
        cor_dict = {
            "corMean": cor_GroupBy.apply(lambda x: x.abs().mean().round(4)),
            "corMedian": cor_GroupBy.apply(lambda x: x.abs().median().round(4)),
            "corStd": cor_GroupBy.apply(lambda x: x.abs().std().round(4)),
            "corTtest_0_6": cor_GroupBy.apply(lambda x: pd.Series(stats.ttest_1samp(abs(x), 0.6)[0],
                                                                  index=x.columns).round(4)),
        }

        for corValue in cor_dict.values():
            corValue.replace(np.inf, np.nan, inplace=True)

        return cor_dict

    # 非线性相关性检验
    def MI(self, data: pd.DataFrame, **kwargs):
        dataSub = data.dropna()
        dataNames = dataSub.columns
        df_mi = pd.DataFrame(columns=dataNames, index=dataNames)

        iters = combinations_with_replacement(dataNames, 2)
        for ite_ in iters:
            df_mi.loc[ite_[0], ite_[1]] = nor_mi(dataSub[ite_[0]], dataSub[ite_[1]])


# 因子合成方法
class DataSynthesis(object):

    hp = 0
    rp = 0

    # class OPT(MaxOptModel):
    #
    #     """
    #     默认:
    #     1.目标函数为最大化收益比上波动
    #     2.权重介于0到1之间
    #     3.权重和为1
    #     4.最大迭代次数为300
    #     5.容忍度为1e-7
    #     """
    #
    #     def __init__(self):
    #         super(OPT, self).__init__()
    #
    #     # 目标函数
    #     def object_func(self, w):
    #         """
    #         目标函数默认为夏普比最大化模型，通过前面加上负号转化为最小化模型
    #         :param w:
    #         :return:
    #         """
    #         mean = np.array(self.data.mean())
    #         cov = np.array(self.data.cov())  # 协方差
    #         func = - np.dot(w, mean) / np.sqrt(np.dot(w, np.dot(w, cov)))
    #         return func
    #
    #     # 约束条件
    #     def _constraint1(self, w, **kwargs):
    #         return sum(w) - 1
    #
    #     # 约束条件函数集
    #     def _constraints(self, **kwargs):
    #         limit = {'type': 'eq', 'fun': self._constraint1}
    #         return limit

    def __init__(self):
        self.opt = MaxOptModel()

    # 因子复合
    def process(self,
                factData: pd.DataFrame,
                factWeight: pd.DataFrame,
                method: str = 'Equal',
                rp: int = 60,
                hp: int = 5,
                **kwargs) -> pd.DataFrame:
        """
        部分权重会用到未来数据，所以需要对权重进行平移与相应的因子值进行匹配
        Parameters
        ----------
        hp :
        rp :
        factData :
        factWeight :
        method :
        kwargs :

        Returns
        -------

        """
        self.rp, self.hp = rp, hp

        factDir = np.sign(factWeight.rolling(rp, min_periods=1).mean())
        factDir = factDir.shift(hp + 1)  # 收益率为标签(预测值), 历史收益数据加权需要+ 1

        # 因子转为正向因子，同时因子收益等指标调整为单调状态
        factNew = factData * factDir
        factWeightNew = factWeight.abs()

        method_dict = {"RetWeight": self.retWeighted,
                       "OPT": self.MAX_IC_IR
                       }

        if method is None:
            return data

        res = method_dict[method](fact=factNew, factWeight=factWeightNew, **kwargs)
        return res

    """因子合成"""

    # 等权法
    def retWeighted(self,
                    fact: pd.DataFrame,
                    factWeight: pd.DataFrame,
                    algorithm: str = 'RetMean',
                    **kwargs) -> pd.Series(float):
        """

        Parameters
        ----------
        factWeight :
        fact :
        algorithm : RetMean: 历史收益率均值， HalfTime: 历史收益率半衰加权
        kwargs :

        Returns
        -------

        """

        if algorithm != 'equal':
            # 生成权重
            factWeightNew = abs(self._weight(factWeight, self.rp, algorithm))
            # 权重归一化
            factWeightStand = factWeightNew.div(factWeightNew.sum(axis=1), axis=0)
            # 权重与因子值匹配
            factWeightStand = factWeightStand.shift(self.hp + 1)
            # 复合因子
            fact_comp = fact.mul(factWeightStand).sum(axis=1)
        else:
            fact_comp = fact.groupby(KN.TRADE_DATE.value, group_keys=False).apply(lambda x: x.mean(axis=1))
        return fact_comp

    def MAX_IC_IR(self,
                  fact: pd.DataFrame,
                  factWeight: pd.DataFrame,
                  retType='IC_IR') -> pd.Series(float):

        # 设置优化方程组
        self.opt.obj_func = self.opt.object_func3
        self.opt.limit.append(self.opt.constraint())
        self.opt.bonds = ((0, 1),) * fact.shape[1]

        # 对收益率进行调整
        factWeightNew = factWeight.shift(self.hp + 1).dropna(how='all')

        weightDict = {}
        for sub in range(self.rp, factWeightNew.shape[0] + 1):
            print(dt.datetime.now(), sub)
            df_ = factWeightNew.iloc[sub - self.rp: sub, :]
            data_mean = np.array(df_.mean())

            if retType == 'IC':
                data_cov = np.array(fact.loc[df_.index].cov())
            else:
                data_cov = np.array(df_.cov())

            optParams = {
                "data_mean": data_mean,
                "data_cov": data_cov,
            }
            self.opt.set_params(**optParams)

            res_ = self.opt.solve()
            weightDict[df_.index[-1]] = res_.x

        w_df = pd.DataFrame(weightDict, index=fact.columns).T
        fact_comp = fact.mul(w_df, level=0).dropna(how='all').sum(axis=1).reindex(fact.index)
        return fact_comp

    def PCA(self,
            fact: pd.DataFrame,
            **kwargs):

        w_list = []
        for i in range(rp, fact.shape[0] + 1):
            df_ = fact.iloc[i - self.rp: i, :]

            pca = PCA(n_components=1)
            pca.fit(np.array(df_))
            weight = pca.components_[0]
            w_s = pd.Series(data=weight, index=df_.columns, name=df_.index[-1])
            w_list.append(w_s)
        w_df = pd.DataFrame(w_list)

        fact_comp = fact.mul(w_df).sum(axis=1)
        fact_comp.name = fact_comp

        return fact_comp

    # *正交化*
    @staticmethod
    def orthogonal(factor_df, method='schimidt'):
        # 固定顺序的施密特正交化
        def schimidt():

            col_name = factor_df.columns
            factors1 = factor_df.values

            R = np.zeros((factors1.shape[1], factors1.shape[1]))
            Q = np.zeros(factors1.shape)
            for k in range(0, factors1.shape[1]):
                R[k, k] = np.sqrt(np.dot(factors1[:, k], factors1[:, k]))
                Q[:, k] = factors1[:, k] / R[k, k]
                for j in range(k + 1, factors1.shape[1]):
                    R[k, j] = np.dot(Q[:, k], factors1[:, j])
                    factors1[:, j] = factors1[:, j] - R[k, j] * Q[:, k]

            Q = pd.DataFrame(Q, columns=col_name, index=factor_df.index)
            return Q

        # 规范正交
        def canonial():
            factors1 = factor_df.values
            col_name = factor_df.columns
            D, U = np.linalg.eig(np.dot(factors1.T, factors1))
            S = np.dot(U, np.diag(D ** (-0.5)))

            Fhat = np.dot(factors1, S)
            Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)

            return Fhat

        # 对称正交
        def symmetry():
            col_name = factor_df.columns
            factors1 = factor_df.values
            D, U = np.linalg.eig(np.dot(factors1.T, factors1))
            S = np.dot(U, np.diag(D ** (-0.5)))

            Fhat = np.dot(factors1, S)
            Fhat = np.dot(Fhat, U.T)
            Fhat = pd.DataFrame(Fhat, columns=col_name, index=factor_df.index)

            return Fhat

        method_dict = {
            "schimidt": schimidt(),
            "canonial": canonial(),
            "symmetry": symmetry()
        }

        return method_dict[method]

    def _weight(self,
                data: pd.DataFrame = None,
                rp: int = 60,
                algorithm: str = 'RetMean') -> [pd.DataFrame, None]:

        if algorithm == 'RetMean':
            data_weight = data.rolling(rp, min_periods=1).mean()
        elif algorithm == 'HalfTime':
            data_weight = data.rolling(rp, min_periods=1).apply(lambda x: np.dot(x, self._Half_time(len(x))))
        else:
            return

        return data_weight

    # 半衰权重
    @staticmethod
    def _Half_time(period: int, decay: int = 2) -> List[str]:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1
