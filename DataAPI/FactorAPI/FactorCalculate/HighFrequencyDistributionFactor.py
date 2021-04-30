import sys
import numpy as np
import pandas as pd
import scipy.stats as st
from pyfinance.ols import PandasRollingOLS

from DataAPI.FactorAPI.FactorBase import FactorBase
from Object import DataInfo
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN
)

"""
高频数据
高频数据需要合成中间过程
然后利用中间过程计算因子
数据命名以合成的数据名为主，没有统一定义

1分钟频率收益率计算采用收盘价比上开盘价(分钟数据存在缺失，采用开盘或者收盘直接计算容易发生跳空现象)
2h 数据存在异常，在原数据中进行剔除
若不做特殊说明， 分钟级别数据运算不包含集合竞价信息
高频数据默认当天连续成交分钟数为240根，缺失分钟数认为未交易
"""


class HighFrequencyDistributionFactor(FactorBase):
    """
    高频因子
    因子逐文件读取计算，数据合并后需要进行实践重拍操作，避免在rolling时发生时间跳跃
    """
    callAM = '09:30:00'
    callPM = '15:00:00'

    def __init__(self):
        super(HighFrequencyDistributionFactor, self).__init__()
        self.range = lambda x: (x[KN.TRADE_TIME.value] >= self.callAM) & (x[KN.TRADE_TIME.value] < self.callPM)

    @classmethod
    def Distribution001(cls,
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
    def Distribution004(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频上行波动(HFD_std_up)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution005(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频下行波动(HFD_std_down)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution006(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频上行波动占比(HFD_std_up_occupy)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution007(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频下行波动占比(HFD_std_down_occupy)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution008(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频量价相关性(HFD_Corr_Vol_P)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution009(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """高频收益偏度(HFD_ret_skew)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    # @classmethod
    # def Distribution010(cls,
    #                     data: pd.DataFrame,
    #                     **kwargs):
    #     """
    #     量价相关性(Cor_Vol_Price)
    #     默认一分钟频率
    #     """
    #
    #     F = DataInfo()
    #     F.data = data
    #     F.data_type = 'HFD'
    #     F.data_category = cls().__class__.__name__
    #     F.data_name = data.name
    #
    #     return F

    @classmethod
    def Distribution011(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        量收益率相关性(Cor_Vol_Ret)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution012(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        收益率方差(Var_Ret)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution013(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        收益率偏度(Var_Skew)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution014(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        收益率峰度(Var_kurt)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution015(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        加权收盘价偏度(Close_Weight_Skew)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution016(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        单位一成交量占比熵(Vol_Entropy)
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution017(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """
        成交额占比熵(Amt_Entropy)
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution018(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """成交量差分均值(Vol_diff_mean)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution019(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """成交量差分绝对值均值(Vol_diff_abs_mean)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution020(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """朴素主动占比因子(Naïve_Amt_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution021(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """T分布主动占比因子(T_Amt_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution022(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """正态分布主动占比因子(N_Amt_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution023(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """置信正态分布主动占比因子(C_N_Amt_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution024(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """均匀分布主动占比因子(Event_Amt_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution025(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """N日分钟成交量波动稳定性(HFD_vol_std)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution026(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """N日分钟成交笔数波动稳定性(HFD_num_std)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution027(cls,
                        data: pd.DataFrame,
                        **kwargs):
        """N日分钟振幅波动稳定性(HFD_ret_std)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def Distribution028(cls,
                        data: pd.DataFrame,
                        n: int = 20,
                        **kwargs):
        """上下午残差收益差的稳定性(APM)"""
        factor_name = sys._getframe().f_code.co_name + f"_{n}days"
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        # Calculate AM and PM returns on stocks
        data['amRetStock'] = data['2hPrice'] / data[PVN.OPEN.value] - 1
        data['pmRetStock'] = data['4hPrice'] / data['2hPrice'] - 1

        # Calculate AM and PM returns on index
        data['amRetIndex'] = data['2hPrice_index'] / data['open_index'] - 1
        data['pmRetIndex'] = data['4hPrice_index'] / data['2hPrice_index'] - 1

        data['stat'] = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
            lambda x: cls._reg_rolling_APM(x, 'amRetIndex', 'amRetStock', 'pmRetIndex', 'pmRetStock', False, True, n))

        # # Calculate 20-day momentum
        data['ret20'] = data.groupby(KN.STOCK_ID.value, group_keys=False)['4hPrice'].pct_change(periods=n, fill_method=None)

        # filter momentum
        data[factor_name] = data.groupby(KN.TRADE_DATE.value,
                                         group_keys=False).apply(lambda x: cls._reg(x, 'ret20', 'stat'))

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def Distribution029(cls,
                        data: pd.DataFrame,
                        n: int = 20,
                        **kwargs):
        """
        隔夜和下午残差收益差的稳定性(APM_new)
        """
        factor_name = sys._getframe().f_code.co_name + f"_{n}days"
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        # Calculate AM and PM returns on stocks
        data['OvernightStock'] = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
            lambda x: x[PVN.OPEN.value] / x['4hPrice'].shift(1) - 1)
        data['pmRetStock'] = data['4hPrice'] / data['2hPrice'] - 1

        # Calculate AM and PM returns on index
        data['OvernightIndex'] = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
            lambda x: x['open_index'] / x['4hPrice_index'].shift(1) - 1)
        data['pmRetIndex'] = data['4hPrice_index'] / data['2hPrice_index'] - 1

        data['stat'] = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
            lambda x: cls._reg_rolling_APM(x, 'OvernightIndex', 'OvernightStock', 'pmRetIndex', 'pmRetStock', False,
                                           True, n))
        # # Calculate 20-day momentum
        data['ret20'] = data.groupby(KN.STOCK_ID.value, group_keys=False)['4hPrice'].pct_change(periods=n, fill_method=None)

        # filter momentum
        data[factor_name] = data.groupby(KN.TRADE_DATE.value, group_keys=False).apply(lambda x: cls._reg(x, 'ret20', 'stat'))

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def Distribution030(cls,
                        data: pd.DataFrame,
                        n: int = 20,
                        **kwargs):
        """
        N日隔夜收益与下午收益差和(OVP)
        """
        factor_name = sys._getframe().f_code.co_name + f"_{n}days"
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        # Calculate AM and PM returns on stocks
        data['Overnight_stock'] = data.groupby(KN.STOCK_ID.value, group_keys=False).apply(
            lambda x: x[PVN.OPEN.value] / x['4hPrice'].shift(1) - 1)
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        data['diff'] = data['Overnight_stock'] - data['pm_ret_stock']

        # filter momentum
        data = cls().reindex(data)
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def Distribution031(cls,
                        data: pd.DataFrame,
                        n: int = 20,
                        **kwargs):
        """
        N日上午收益与下午收益差和(AVP)
        """
        factor_name = sys._getframe().f_code.co_name + f"_{n}days"
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        # Calculate AM and PM returns on stocks
        data['am_ret_stock'] = data['2hPrice'] / data[PVN.OPEN.value] - 1
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        data['diff'] = data['am_ret_stock'] - data['pm_ret_stock']

        # filter momentum
        data = cls().reindex(data)
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    ####################################################################################################################
    @classmethod
    def Distribution001_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 method: str = 'raw',
                                 **kwargs) -> pd.Series(float):
        """高频波动(HFD_std_ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: pow(pow(x[KN.RETURN.value], 2).sum(), 0.5))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = pow(data['ret2Up_0'] + data['ret2Down_0'], 0.5)

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution004_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 method: str = 'raw',
                                 **kwargs) -> pd.Series(float):
        """高频上行波动(HFD_std_up)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: pow(pow(x[KN.RETURN.value][x[KN.RETURN.value] > 0], 2).sum(), 0.5))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Up_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = pow(data['ret2Up_0'], 0.5)

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution005_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """高频下行波动(HFD_std_down)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: pow(pow(x[KN.RETURN.value][x[KN.RETURN.value] < 0], 2).sum(), 0.5))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Down_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = pow(data['ret2Down_0'], 0.5)

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution006_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """高频上行波动占比(HFD_std_up_occupy)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: np.nan if pow(x[KN.RETURN.value], 2).sum() == 0
                    else pow(x[KN.RETURN.value][x[KN.RETURN.value] > 0], 2).sum() / pow(x[KN.RETURN.value], 2).sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['ret2Up_0'] / (data['ret2Up_0'] + data['ret2Down_0'])

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution007_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 method: str = 'raw',
                                 **kwargs) -> pd.Series(float):
        """高频下行波动占比(HFD_std_down_occupy)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: np.nan if pow(x[KN.RETURN.value], 2).sum() == 0
                    else pow(x[KN.RETURN.value][x[KN.RETURN.value] < 0], 2).sum() / pow(x[KN.RETURN.value], 2).sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['ret2Down_0'] / (data['ret2Up_0'] + data['ret2Down_0'])

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution008_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """高频量价相关性(HFD_Corr_Vol_P)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value,
                                  group_keys=False).apply(lambda x: x[PVN.CLOSE.value].corr(x[PVN.VOLUME.value]))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['corCloseVol'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['corCloseVol']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution009_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """高频收益偏度(HFD_ret_skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['ret_2'], d_sub['ret_3'] = pow(d_sub[KN.RETURN.value], 2), pow(d_sub[KN.RETURN.value], 3)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: np.nan if x['ret_2'].sum() == 0
                    else x['ret_3'].sum() * pow(len(x), 0.5) / pow(x['ret_2'].sum(), 1.5))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0', 'ret3Up_0', 'ret3Down_0'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            data['ret2Sum'] = data['ret2Up_0'] + data['ret2Down_0']
            data['ret3Sum'] = data['ret3Up_0'] + data['ret3Down_0']

            res = data['ret3Sum'] * pow(239, 0.5) / pow(data['ret2Sum'], 1.5)

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    # @classmethod
    # def Distribution010_data_raw(cls,
    #                              minute: int = 5,
    #                              n: int = 20,
    #                              method: str = 'mid',
    #                              **kwargs):
    #     """量价相关性(Cor_Vol_Price)"""
    #     factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
    #
    #     if method == 'raw':
    #         def func(d: pd.DataFrame):
    #             d_sub = d[cls().range]
    #             r = d_sub.groupby(KN.STOCK_ID.value,
    #                               group_keys=False).apply(lambda x: x[PVN.CLOSE.value].corr(x[PVN.VOLUME.value]))
    #             return r
    #
    #         Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
    #                                func=func,
    #                                file_path=FPN.HFD.value,
    #                                sub_file=f"{minute}minute")
    #         res = pd.concat(Q)
    #
    #     elif method == 'mid':
    #         data = cls()._csv_data(data_name=['corCloseVol'],
    #                                file_path=FPN.HFD_MidData.value,
    #                                file_name='TradeSpecial1',
    #                                stock_id=KN.STOCK_ID.value)
    #
    #         data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
    #         res = data['corCloseVol']
    #
    #     else:
    #         print('Parameter is wrong!')
    #         res = pd.Series()
    #
    #     res = cls().reindex(res)
    #     res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
    #     res.name = factor_name
    #
    #     return res

    @classmethod
    def Distribution011_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """量收益率相关性(Cor_Vol_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value,
                                  group_keys=False).apply(lambda x: x[KN.RETURN.value].corr(x[PVN.VOLUME.value]))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['corRetVol'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['corRetVol']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution012_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """收益率方差(Var_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False)[KN.RETURN.value].var()
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['retVar'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['retVar']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution013_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """收益率偏度(Var_Skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False)[KN.RETURN.value].skew()
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['retSkew'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['retSkew']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution014_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """收益率峰度(Var_kurt)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False)[KN.RETURN.value].apply(lambda x: x.kurt())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['retKurt'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['retKurt']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution015_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        加权收盘价偏度(Close_Weight_Skew)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: (pow((x[PVN.CLOSE.value] - x[PVN.CLOSE.value].mean()) / x[PVN.CLOSE.value].std(), 3) * (
                            x[PVN.VOLUME.value] / x[PVN.VOLUME.value].sum())).sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['closeVolWeightSkew'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['closeVolWeightSkew']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution016_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        单位一成交量占比熵(Vol_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: cls.entropy(x[PVN.CLOSE.value] * x[PVN.VOLUME.value]))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['volEntropy'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['volEntropy']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution017_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        成交额占比熵(Amt_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: cls.entropy(x[PVN.AMOUNT.value]))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['amtEntropy'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['amtEntropy']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution018_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """每笔成交量差分均值(Vol_diff_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub['volb'] = d_sub[PVN.VOLUME.value] / d_sub['tradenum']
                d_sub['diff'] = d_sub.groupby(KN.STOCK_ID.value)['volb'].diff(1)
                d_sub = d_sub.groupby(KN.STOCK_ID.value).agg({"diff": 'std', PVN.VOLUME.value: 'mean'})
                r = d_sub['diff'] / d_sub[PVN.VOLUME.value]
                return r

            Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['volPerDiffStd'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeVol',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['volume'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)
            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['volPerDiffStd'] / data['volume'] * 240

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution019_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """每笔成交量差分绝对值均值(Vol_diff_abs_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub['volb'] = d_sub[PVN.VOLUME.value] / d_sub['tradenum']
                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: abs(x['volb'].diff(1)).mean() / x['volb'].mean())
                return r

            Q = cls().csv_HFD_data(data_name=['tradenum', PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['volPerDiffAbsMean', 'volPerMean'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeVol',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['volPerDiffAbsMean'] / data['volPerMean']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution020_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        朴素主动占比因子(Naïve_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub["closeStand"] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: x[PVN.CLOSE.value].diff(1) / x[PVN.CLOSE.value].diff(1).std())

                d_sub['buy_T'] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: st.t.cdf(x["closeStand"], len(x) - 1) * x[PVN.AMOUNT.value])

                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['naiveAmtR'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['naiveAmtR']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution021_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        T分布主动占比因子(T_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['retStand'] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: x[KN.RETURN.value] / x[KN.RETURN.value].std())

                d_sub['buy_T'] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: st.t.cdf(x["retStand"], len(x) - 1) * x[PVN.AMOUNT.value])

                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['TAmtR'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['TAmtR']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution022_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        正态分布主动占比因子(N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['retStand'] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: x[KN.RETURN.value] / x[KN.RETURN.value].std())
                d_sub['buy_T'] = st.norm.cdf(d_sub["retStand"]) * d_sub[PVN.AMOUNT.value]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)
        elif method == 'mid':
            data = cls()._csv_data(data_name=['NAmtR'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['NAmtR']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution023_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        置信正态分布主动占比因子(C_N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['buy_T'] = st.norm.cdf(d_sub[KN.RETURN.value] / 0.1 * 1.96) * d_sub[PVN.AMOUNT.value]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['CNAmtR'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['CNAmtR']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution024_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        均匀分布主动占比因子(Event_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['buy_T'] = (d_sub[KN.RETURN.value] - 0.1) / 0.2 * d_sub[PVN.AMOUNT.value]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(lambda x: x['buy_T'].sum() / x[PVN.AMOUNT.value].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['EventAmtR'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['EventAmtR']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution025_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟成交量差分绝对值波动稳定性(HFD_vol_diff_abs_std)
        考虑集合竞价成交量波动较大，计算日内成交量波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: abs(x[PVN.VOLUME.value].diff(1)).mean() / x[PVN.VOLUME.value].mean())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['volDiffAbsMean'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeVol',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['volume'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['volDiffAbsMean'] / data['volume'] * 240

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution026_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟成交笔数差分绝对值波动稳定性(HFD_num_diff_abs_std)
        考虑集合竞价笔数实际意义可能不大，计算日内笔数波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: abs(x['tradenum'].diff(1)).mean() / x['tradenum'].mean())
                return r

            Q = cls().csv_HFD_data(data_name=['tradenum'],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0',
                                              'tradeNumDiffAbsMean'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeTradeNum',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            data['tradeNumMean'] = (data['tradeNumRetUpSum_0'] +
                                    data['tradeNumRetDownSum_0'] +
                                    data['tradeNumRetEqualSum_0']) / 240
            res = data['tradeNumDiffAbsMean'] / data['tradeNumMean']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution027_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 method: str = 'mid',
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟振幅差分绝对值波动稳定性(HFD_ret_diff_abs_std)
        集合竞价只存在唯一价格，振幅为零, 剔除
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: abs(x[KN.RETURN.value].diff(1)).mean() / x[KN.RETURN.value].mean())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['retDiffAbsMean', 'retMean'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeRet',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['retDiffAbsMean'] / data['retMean']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution028_data_raw(cls,
                                 **kwargs):
        """上下午残差收益差的稳定性(APM)"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id=KN.STOCK_ID.value)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        data_index = cls().csv_index(data_name=[PVN.OPEN.value, PVN.CLOSE.value, KN.TRADE_TIME.value],
                                     file_path=FPN.HFD.value,
                                     index_name='000905.SH',
                                     file_name='HFDIndex')

        data_index_close = data_index.pivot_table(values=PVN.CLOSE.value,
                                                  columns=KN.TRADE_TIME.value,
                                                  index=KN.TRADE_DATE.value)
        data_index_open = data_index.groupby(KN.TRADE_DATE.value)[PVN.OPEN.value].first()

        data_index_new = pd.concat([data_index_close, data_index_open], axis=1)
        data_index_new = data_index_new.rename(columns={"10:30": '1hPrice_index',
                                                        "11:30": '2hPrice_index',
                                                        "14:00": '3hPrice_index',
                                                        "15:00": '4hPrice_index',
                                                        "open": 'open_index'})

        data_raw = pd.merge(data_s, data_index_new, on=KN.TRADE_DATE.value, how='left')

        return data_raw

    @classmethod
    def Distribution029_data_raw(cls,
                                 **kwargs):
        """隔夜和下午残差收益差的稳定性(APM_new)"""
        return cls.Distribution028_data_raw()

    @classmethod
    def Distribution030_data_raw(cls,
                                 **kwargs):
        """N日隔夜收益与下午收益差和(OVP)"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id=KN.STOCK_ID.value)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        return data_s

    @classmethod
    def Distribution031_data_raw(cls,
                                 **kwargs):
        """N日上午收益与下午收益差和(AVP)"""
        return cls.Distribution030_data_raw()

    @staticmethod
    def _reg_rolling_APM(reg_: pd.DataFrame,
                         x1: str,
                         y1: str,
                         x2: str,
                         y2: str,
                         has_const: bool = False,
                         use_const: bool = True,
                         window: int = 20) -> pd.Series:
        # print(reg_.index[0])
        if len(reg_) <= window:
            res = pd.Series(index=reg_.index)
        else:
            reg_object_am = PandasRollingOLS(x=reg_[x1],
                                             y=reg_[y1],
                                             has_const=has_const,
                                             use_const=use_const,
                                             window=window)

            reg_object_pm = PandasRollingOLS(x=reg_[x2],
                                             y=reg_[y2],
                                             has_const=has_const,
                                             use_const=use_const,
                                             window=window)

            diff_resids = reg_object_am._resids - reg_object_pm._resids
            stat = np.nanmean(diff_resids, axis=1) / np.nanstd(diff_resids, axis=1, ddof=1) * np.sqrt(window)
            res = pd.Series(stat, index=reg_object_am.index[window - 1:])
        return res

    # 信息熵
    @staticmethod
    def entropy(x: pd.Series, bottom: int = 2):
        """
        离散熵
        空值不剔除
        :param x:
        :param bottom:
        :return:
        """
        Probability = (x.groupby(x).count()).div(len(x))
        log2 = np.log(Probability) / np.log(bottom)
        result = - (Probability * log2).sum()
        return result

    @staticmethod
    def _reg(d: pd.DataFrame,
             x_name: str,
             y_name: str) -> pd.Series:
        """！！！不排序回归结果会不一样！！！"""
        d_sub_ = d.dropna(how='any').sort_index()

        if d_sub_.shape[0] < d_sub_.shape[1]:
            Residual = pd.Series(data=np.nan, index=d.index)
        else:
            X, Y = d_sub_[x_name].to_frame(), d_sub_[y_name]
            reg = np.linalg.lstsq(X, Y)
            Residual = Y - (reg[0] * X).sum(axis=1)
        return Residual


if __name__ == '__main__':
    A = HighFrequencyDistributionFactor()
    r_df = A.Distribution007_data_raw(minute=10, n=1)
    A.Distribution007(r_df)
    pass