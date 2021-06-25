# -*-coding:utf-8-*-
# @Time:   2021/4/14 13:43
# @Author: FC
# @Email:  18817289038@163.com

import time
import numpy as np
import pandas as pd
import datetime as dt
import statsmodels.api as sm

from scipy import stats
from sklearn.decomposition import PCA
from typing import Dict, Any, List, Union
from statsmodels.tsa.arima_model import ARMA
from itertools import combinations_with_replacement
from sklearn.metrics import normalized_mutual_info_score as nor_mi

from utility.Optimization import MaxOptModel
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    SpecialName as SN
)


# 因子处理方法合集
class MethodSets(object):
    # 方法参数必须被继承复写

    methodProcess = {
        "RO": {"method": "", "p": {}},  # 异常值处理
        "Neu": {"method": "", "p": {}},  # 中性化处理
        "Sta": {"method": "", "p": {}},  # 标准化处理

        "Cor": {"method": "", "p": {}},  # 相关性计算
        "Syn": {"method": "", "p": {}},  # 因子合成

        "Ret": {"method": "", "p": {}},  # 收益预测方法
        "Risk": {"method": "", "p": {}},  # 风险分析方法
    }

    def __init__(self):
        self.Cor = Correlation()
        self.Syn = DataSynthesis()
        self.RO = RemoveOutlier()
        self.Neu = Neutralization()
        self.Sta = Standardization()

    # 更新参数
    def set_params(self, **kwargs):
        """

        Parameters
        ----------
        Returns
        -------
        对于因子处理方法设置因子参数
        """
        for paramName, paramValue in kwargs.items():
            setattr(self, paramName, paramValue)

    def processSingle(self,
                      data: Union[pd.DataFrame, pd.Series],
                      methodN: str,
                      **kwargs) -> Any:
        """
        单一处理方法
        Parameters
        ----------
        data :
        methodN : 方法名
        kwargs :

        Returns
        -------

        """
        value = getattr(self, methodN).process(data=data,
                                               method=self.methodProcess[methodN]['method'],
                                               **self.methodProcess[methodN]['p'],
                                               **kwargs)
        return value

    def processSeq(self,
                   data: pd.DataFrame,
                   methodN: List[str],
                   dataName: str,
                   **kwargs):
        """
        连续处理
        Parameters
        ----------
        data :
        methodN : 方法名list，按顺序执行
        dataName :
        kwargs :

        Returns
        -------

        """
        for M in methodN:
            if self.methodProcess[M]['method'] != "":
                value = getattr(self, M).process(data=data,
                                                 method=self.methodProcess[M]['method'],
                                                 dataName=dataName,
                                                 **self.methodProcess[M]['p'],
                                                 **kwargs)
                data[dataName] = value


# 去极值
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


# 中性化
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


# 标准化
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
        denominator, numerator = data_df.max(axis=1) - data_df.min(axis=1), data_df.sub(data_df.min(axis=1), axis=0)
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


# 相关性检验
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

    def __init__(self):
        self.opt = MaxOptModel()

    # 因子复合
    def process(self,
                data: pd.DataFrame,
                weight: pd.DataFrame,
                method: str = 'RetWeight',
                **kwargs) -> pd.DataFrame:
        """
        部分权重会用到未来数据，所以需要对权重进行平移与相应的因子值进行匹配
        Parameters
        ----------
        data : 因子集
        weight :因子权重
        method : 因子合成方法
        kwargs :

        Returns
        -------

        """

        # factor direction
        factDir = np.sign(weight.mean())

        # switch the factor direction，and turn the weight to positive
        factNew = data.mul(factDir, level=0).dropna(how='all')
        weightNew = weight.abs()

        method_dict = {"RetWeight": self.ret_weighted,
                       "OPT": self.MAX_IC_IR,
                       "PCA": self.PCA,
                       "Cluster": self.cluster
                       }

        if method is None:
            return data

        res = method_dict[method](fact=factNew, weight=weightNew, **kwargs)
        return res

    """因子合成"""

    # 加权法
    def ret_weighted(self,
                     fact: pd.DataFrame,
                     weight: pd.DataFrame,
                     weightAlgo: str = 'IC',
                     meanType: str = 'Equal',
                     **kwargs) -> pd.Series(float):
        """
        某股票因子值为Nan，则将剩余因子值进行加权
        默认采用等权加权
        Parameters
        ----------
        weight :
        fact :
        weightAlgo : RetMean: 历史收益率均值， HalfTime: 历史收益率半衰加权
        meanType :
        kwargs :

        Returns
        -------

        """

        # TODO 测试
        if weightAlgo in ['IC', 'Ret']:
            # 生成权重
            factW = self.mean_weight(weight, meanType)
            # 对非空部分进行加权
            fact_comp = self.weighted(fact, factW.values)
        elif weightAlgo == 'IC_IR':
            IC_Mean = self.mean_weight(weight, meanType)
            IC_Std = self.std_weight(weight, meanType)
            factW = IC_Mean / IC_Std

            fact_comp = self.weighted(fact, factW.values)
        else:
            fact_comp = fact.mean(axis=1)
        return fact_comp

    def MAX_IC_IR(self,
                  fact: pd.DataFrame,
                  weight: pd.DataFrame,
                  weightAlgo='IC_IR') -> pd.Series(float):

        # TODO  压缩矩阵估计协方差
        self.opt.clear()
        # 设置优化方程组
        self.opt.obj_func = self.opt.object_func3
        self.opt.limit.append(self.opt.constraint())  # 权重和为1
        self.opt.bonds = ((0, 1),) * fact.shape[1]  # 权重大于0

        weight_mean = np.array(weight.mean())
        if weightAlgo == 'IC':
            weight_cov = np.array(fact.cov())
        else:
            weight_cov = np.array(weight.cov())
        optParams = {
            "data_mean": weight_mean,
            "data_cov": weight_cov,
        }
        self.opt.set_params(**optParams)

        weightOpt = self.opt.solve()

        fact_comp = self.weighted(fact, weightOpt.x)
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

    def cluster(self,
                fact: pd.DataFrame):
        pass

    # 正交化
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

    # 加权均值
    def mean_weight(self,
                    data: pd.DataFrame = None,
                    meanType: str = 'Equal') -> [pd.Series, None]:

        if meanType == 'HalfTime':
            meanW = data.apply(lambda x: np.dot(x, self.half_time(len(x))))
        else:
            meanW = data.mean()
        return meanW

    # 加权标准差
    def std_weight(self,
                   data: pd.DataFrame = None,
                   meanType: str = 'Equal') -> [pd.Series, None]:
        dataSub = data.dropna()

        if meanType == 'HalfTime':
            stdW = np.diag(pow(np.cov(dataSub.T, aweights=self.half_time(len(dataSub))), 0.5))
            stdW = pd.Series(stdW, name='std')
        else:
            stdW = dataSub.std()
        return stdW

    # 半衰权重
    @staticmethod
    def half_time(period: int, decay: int = 2) -> List[str]:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1

    # 考虑缺失值加权法
    def weighted(self, fact: pd.DataFrame, weight: np.array) -> pd.Series:
        """
        对非空部分进行加权
        """
        weight_df = pd.DataFrame(np.repeat(weight.reshape(1, len(weight)), len(fact.index), axis=0),
                                 index=fact.index,
                                 columns=fact.columns)
        weight_df = pd.DataFrame(np.where(fact.isna(), np.nan, weight_df),
                                 index=fact.index,
                                 columns=fact.columns)
        # 复合因子
        factWeighted = (fact * weight_df).div(weight_df.sum(axis=1), axis=0).sum(axis=1)
        return factWeighted


# 收益预测模型
class ReturnModel(object):

    def process(self,
                data: pd.DataFrame,
                method='EWMA',
                **kwargs) -> pd.Series(float):

        method_dict = {
            "Equal": self.equal_weight,
            "EWMA": self.EWMA,
            "TS": self.TimeSeries,
        }

        if method is None:
            return data

        res = method_dict[method](data, **kwargs)

        return res

    # 等权
    def equalWeight(self,
                    data: pd.DataFrame,
                    rp: int = 20,
                    **kwargs):
        """
        因子收益预测--等权法：过去一段时间收益的等权平均作为下一期因子收益的预测
        :param data: 因子收益序列
        :param rp: 滚动周期
        :return:
        """
        fore_ret = data.rolling(rp, min_periods=1).mean().dropna()
        return fore_ret

    # 指数加权移动平均法
    def EWMA(self,
             data: pd.DataFrame,
             alpha: float = 0.5,
             **kwargs):
        """
        pd.ewm中com与alpha的关系为 1 / alpha - 1 = com
        pd.ewm中adjust参数需要设置为False
        :param data:
        :param alpha: 当期权重，前一期权重为1-alpha
        :return:
        """
        fore_ret = data.ewm(com=1 / alpha - 1, adjust=False).mean()
        return fore_ret

    # 时间序列模型
    def TimeSeries(self,
                   data: pd.DataFrame,
                   rp: int = 20,
                   AR_q: int = 1,
                   MA_p: int = 1,
                   **kwargs):
        fore_ret = data.rolling(rp).apply(lambda x: self._ARMA(x, AR_q, MA_p))
        return fore_ret

    # TODO 待研究
    def _ARMA(self, data: pd.Series, AR_q: int = 1, MA_p: int = 1):
        try:
            ar_ma = ARMA(data, order=(AR_q, MA_p)).fit(disp=0)
        except Exception as e:
            print(e)
            print("尝试采用其他滞后阶数")
            forecast = np.nan
        else:
            forecast = ar_ma.predict()[-1]

        return forecast

    def KML(self, data: pd.DataFrame):
        pass


# 风险预测模型
class RiskModel(object):

    def __init__(self):
        pass

    # 因子协方差矩阵估计
    def forecast_cov_fact(self,
                          fact_ret: pd.DataFrame,
                          decay: int = 2,
                          order: int = 2,
                          annual: int = 1):
        """

        :param fact_ret: 因子收益序列
        :param decay: 指数加权衰减系数
        :param order: 自相关之后阶数
        :param annual: "年化"参数
        :return:
        """
        # 指数加权协方差矩阵
        F_Raw = self.exp_weight_cov(fact_ret, decay=decay)

        #  Newey-West adjustment
        matrix_orders = np.zeros(shape=(fact_ret.shape[1], fact_ret.shape[1]))
        for order_ in range(1, order + 1):
            w = 1 - order_ / (order + 1)
            # 滞后order阶的自相关协方差矩阵
            matrix_order = self.auto_cor_cov(fact_ret, order=order, decay=decay)
            matrix_orders += w * (matrix_order + matrix_order.T)

        #  Eigenvalue adjustment
        F_NW = annual * (F_Raw + matrix_orders)

        # 特征值调整
        F_Eigen = self.eigenvalue_adj(F_NW, period=120, M=100)

        # Volatility bias adjustment  TODO
        # F = self.vol_bias_adj(F_Eigen)
        F = F_Eigen
        return F

    # 特异性收益协方差矩阵预测
    def forecast_cov_spec(self,
                          spec_ret: pd.DataFrame,
                          fact_exp: pd.DataFrame,
                          liq_mv: pd.DataFrame,
                          liq_mv_name: str = PVN.LIQ_MV.value,
                          decay: int = 2,
                          order: int = 5,
                          annual: int = 1):
        """

        :param spec_ret: 个股特异性收益
        :param fact_exp: 因子暴露
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param decay: 指数加权衰减周期
        :param order: Newey-West调整最大滞后阶数
        :param annual: 调仓期：对协方差矩阵进行"年化"调整
        :return:
        """
        # 删除无效资产
        eff_asset = spec_ret.iloc[-1, :].dropna().index
        spec_ret_eff = spec_ret[eff_asset]

        # Calculate the weighted covariance of the specific return index
        F_Raw = self.exp_weight_cov(spec_ret_eff, decay=decay)

        #  Newey-West adjustment: 自由度设为n-1
        matrix_orders = np.zeros(shape=(spec_ret_eff.shape[1], spec_ret_eff.shape[1]))
        for order_ in range(1, order + 1):
            w = 1 - order_ / (order + 1)
            matrix_order = self.auto_cor_cov(spec_ret_eff, order=order_, decay=decay)
            matrix_orders += w * (matrix_order + matrix_order.T)

        #  Eigenvalue adjustment
        F_NW = annual * (F_Raw + matrix_orders)

        #  Structural adjustment
        F_STR = self.structural_adj(F_NW, spec_ret_eff, fact_exp, liq_mv.iloc[:, 0], liq_mv_name)

        # Bayesian compression adjustment
        F_SH = self.Bayesian_compression(F_STR, liq_mv.iloc[:, 0], liq_mv_name)

        # 波动率偏误调整  TODO

        # 非对角矩阵替换为0

        return F_SH

    # 指数加权协方差矩阵计算
    def exp_weight_cov(self,
                       data: pd.DataFrame,
                       decay: int = 2) -> pd.DataFrame:
        # Exponentially weighted index volatility: Half-Life attenuation

        w_list = self.Half_time(period=data.shape[0], decay=decay)
        w_list = sorted(w_list, reverse=False)  # 升序排列

        cov_w = pd.DataFrame(np.cov(data.T, aweights=w_list), index=data.columns, columns=data.columns)

        return cov_w

    # 自相关协方差矩阵
    def auto_cor_cov(self,
                     data: pd.DataFrame,
                     order: int = 2,
                     decay: int = 2) -> pd.DataFrame:
        """
        矩阵与矩阵相关性计算：
        A = np.array([[a11,a21],[a12,a22]])
        B = np.array([[b11,b21],[b12,b22]])

        matrix = [[cov([a11,a21], [a11,a21]), cov([a11,a21], [a12,a22]), cov([a11,a21], [b11,b21]), cov([a11,a21], [b12,b22])],
                  [cov([a12,a22], [a11,a21]), cov([a12,a22], [a12,a22]), cov([a12,a22], [b11,b21]), cov([a12,a22], [b12,b22])],
                  [cov([b11,b21], [a11,a21]), cov([b11,b21], [a12,a22]), cov([b11,b21], [b11,b21]), cov([b11,b21], [b12,b22])],
                  [cov([b12,b22], [a11,a21]), cov([b12,b22], [a12,a22]), cov([b12,b22], [b11,b21]), cov([b12,b22], [b12,b22])]]

        自相关协方差矩阵为:
        matrix_at_cor_cov = [[cov([a11,a21], [b11,b21]), cov([a11,a21], [b12,b22])],
                             [cov([a12,a22], [b11,b21]), cov([a12,a22], [b12,b22])]

        注：
        输入pd.DataFrame格式的数据计算协方差会以行为单位向量进行计算
        计算出来的协方差矩阵中右上角order*order矩阵才是自相关矩阵
        协方差矩阵：横向为当期与各因子滞后阶数的协方差；纵向为滞后阶数与当期各因子的协方差
        :param data:
        :param order:
        :param decay:
        :return:
        """

        # order matrix
        matrix_order = data.shift(order).dropna(axis=0, how='all')
        matrix = data.iloc[order:, :].copy(deep=True)

        w_list = self.Half_time(period=matrix.shape[0], decay=decay)
        w_list = sorted(w_list, reverse=False)  # 升序排列

        covs = np.cov(matrix.T, matrix_order.T, aweights=w_list)  # 需要再测试
        cov_order = pd.DataFrame(covs[: -matrix.shape[1], -matrix.shape[1]:],
                                 index=matrix.columns,
                                 columns=matrix.columns)

        return cov_order

    # 特征值调整
    def eigenvalue_adj(self,
                       data: np.array,
                       period: int = 120,
                       M: int = 3000,
                       alpha: float = 1.5):
        """

        :param data:Newey-West调整后的协方差矩阵
        :param period: 蒙特卡洛模拟收益期数
        :param M: 蒙特卡洛模拟次数
        :param alpha:
        :return:
        """

        # 矩阵奇异值分解
        e_vals, U0 = np.linalg.eig(data)

        # 对角矩阵
        D0 = np.diag(e_vals)

        # 蒙特卡洛模拟
        eigenvalue_bias = []
        for i in range(M):
            S = np.random.randn(len(e_vals), period)  # 模拟的特征组合收益率矩阵, 收益期数怎么定 TODO
            f = np.dot(U0, S)  # 模拟的收益率矩阵
            F = np.cov(f)  # 模拟的收益率协方差矩阵
            e_vas_S, U1 = np.linalg.eig(F)  # 对模拟的协方差矩阵进行奇异值分解
            D1 = np.diag(e_vas_S)  # 生成模拟协方差矩阵特征值的对角矩阵
            D1_real = np.dot(np.dot(U1.T, data), U1)

            D1_real = np.diag(np.diag(D1_real))  # 转化为对角矩阵

            lam = D1_real / D1  # 特征值偏误
            eigenvalue_bias.append(lam)

        gam_ = reduce(lambda x, y: x + y, eigenvalue_bias)
        gam = (np.sqrt(gam_ / M) - 1) * alpha + 1
        gam[np.isnan(gam)] = 0

        F_Eigen = pd.DataFrame(np.dot(np.dot(U0, np.dot(gam ** 2, D0)), np.linalg.inv(U0)),
                               index=data.columns,
                               columns=data.columns)

        return F_Eigen

    # 结构化调整
    def structural_adj(self,
                       cov: pd.DataFrame,
                       spec_ret: pd.DataFrame,
                       fact_exp: pd.DataFrame,
                       liq_mv: pd.DataFrame,
                       liq_mv_name: PVN.LIQ_MV.value,
                       time_window: int = 120):
        """

        :param cov: 经Newey-West调整的个股特异收益矩阵
        :param spec_ret: 个股特异收益序列
        :param fact_exp: 因子暴露
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param time_window: 个股特异收益的时间窗口（后面考虑改为特异收益序列的长度）
        :return:
        """
        # 计算协调参数
        h_n = spec_ret.count()  # 非空数量
        V_n = (h_n - 20 / 4) / 20 * 2  # 数据缺失程度（先用20测试）

        sigma_n = spec_ret.std().fillna(1)  # 样本等权标准差（无法计算的标准差记为1）  TODO

        sigma_n_steady = (spec_ret.quantile(.75) - spec_ret.quantile(0.25)) / 1.35  # 样本稳健估计标准差

        Z_n = abs((sigma_n - sigma_n_steady) / sigma_n_steady)  # 数据肥尾程度

        # 将无限大值替换为0
        Z_n[np.isinf(Z_n)] = 0
        Z_n.fillna(0, inplace=True)

        left_, right_ = V_n.where(V_n > 0, 0), np.exp(1 - Z_n)

        left_, right_ = left_.where(left_ < 1, 1), right_.where(right_ < 1, 1)
        gam_n = left_ * right_  # 个股协调参数[0,1]

        reg_data = pd.concat([np.log(sigma_n), liq_mv, gam_n, fact_exp], axis=1)
        reg_data.columns = ['sigma', liq_mv_name, 'gam_n'] + fact_exp.columns.tolist()

        ref_data_com = reg_data[reg_data['gam_n'] == 1]

        # 加权（流通市值）最小二乘法用优质股票估计因子对特异波动的贡献值
        model = sm.WLS(ref_data_com['sigma'], ref_data_com[fact_exp.columns], weights=ref_data_com['gam_n']).fit()

        # 个股结构化特异波动预测值
        sigma_STR = pd.DataFrame(np.diag(np.exp(np.dot(fact_exp, model.params)) * 1.05),
                                 index=fact_exp.index,
                                 columns=fact_exp.index)

        # 对特异收益矩阵进行结构化调整
        F_STR = sigma_STR.mul((1 - gam_n), axis=0) + cov.mul(gam_n, axis=0)

        return F_STR

    # 贝叶斯压缩
    def Bayesian_compression(self,
                             cov: pd.DataFrame,
                             liq_mv: pd.DataFrame,
                             liq_mv_name: PVN.LIQ_MV.value,
                             group_num: int = 10,
                             q: int = 1
                             ):
        """
        𝜎_𝑛_𝑆𝐻 = 𝑣_𝑛*𝜎_𝑛 + (1 − 𝑣_𝑛)*𝜎_𝑛^

        :param cov: 经结构化调整的特异收益矩阵
        :param liq_mv: 流通市值
        :param liq_mv_name: 流通市值名称
        :param group_num: 分组个数
        :param q: 压缩系数，该系数越大，先验风险矩阵所占权重越大
        :return:
        """
        df_ = pd.DataFrame({"sigma_n": np.diag(cov), liq_mv_name: liq_mv})
        # 按流通市值分组
        df_['Group'] = pd.cut(df_['sigma_n'], group_num, labels=[f'Group_{i}' for i in range(1, group_num + 1)])

        # 各组特异风险市值加权均值
        df_['weight'] = df_.groupby('Group', group_keys=False).apply(lambda x: x[liq_mv_name] / x[liq_mv_name].sum())
        sigma_n_weight = df_.groupby('Group').apply(lambda x: x['weight'] @ x['sigma_n'])
        sigma_n_weight.name = 'sigma_n_weight'

        df_N1 = pd.merge(df_, sigma_n_weight, left_on=['Group'], right_index=True, how='left')

        # 个股所属分组特异波动的标准差

        try:
            delta_n = df_N1.groupby('Group').apply(
                lambda x: np.nan if x.empty else pow(sum((x['sigma_n'] - x['sigma_n_weight']) ** 2) / x.shape[0], 0.5))
        except Exception as e:
            delta_n = df_N1.groupby('Group').apply(
                lambda x: np.nan if x.empty else pow(sum((x['sigma_n'] - x['sigma_n_weight']) ** 2) / x.shape[0], 0.5))
            print(e)

        delta_n.name = 'delta'

        df_N2 = pd.merge(df_N1, delta_n, left_on=['Group'], right_index=True, how='left')

        # 压缩系数
        df_N2['V_n'] = q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']) / (
                df_N2['delta'] + q * abs(df_N2['sigma_n'] - df_N2['sigma_n_weight']))

        # 调整后的特异波动
        sigma_SH = df_N2['V_n'] * df_N2['sigma_n_weight'] + (1 - df_N2['V_n']) * df_N2['sigma_n']
        F_SH = pd.DataFrame(np.diag(np.array(sigma_SH)), index=sigma_SH.index, columns=sigma_SH.index)

        return F_SH

    # 半衰权重
    @staticmethod
    def Half_time(period: int, decay: int = 2) -> list:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1
