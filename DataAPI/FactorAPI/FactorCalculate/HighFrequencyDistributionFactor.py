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
                                 **kwargs) -> pd.Series(float):
        """高频波动(HFD_std_ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = pow(data['ret2Up_0'] + data['ret2Down_0'], 0.5)

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution004_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 21,
                                 **kwargs) -> pd.Series(float):
        """高频上行波动(HFD_std_up)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['ret2Up_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = pow(data['ret2Up_0'], 0.5)

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
        data = cls()._csv_data(data_name=['ret2Down_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = pow(data['ret2Down_0'], 0.5)

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
        data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['ret2Up_0'] / (data['ret2Up_0'] + data['ret2Down_0'])

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
        data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data['ret2Down_0'] / (data['ret2Up_0'] + data['ret2Down_0'])

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution008_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """高频量价相关性(HFD_Corr_Vol_P)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['corCloseVol'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial1',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['corCloseVol']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution009_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """高频收益偏度(HFD_ret_skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['ret2Up_0', 'ret2Down_0', 'ret3Up_0', 'ret3Down_0'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data['ret2Sum'] = data['ret2Up_0'] + data['ret2Down_0']
        data['ret3Sum'] = data['ret3Up_0'] + data['ret3Down_0']

        res = data['ret3Sum'] * pow(239, 0.5) / pow(data['ret2Sum'], 1.5)

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution011_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """量收益率相关性(Cor_Vol_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['corRetVol'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial1',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['corRetVol']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution012_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """收益率方差(Var_Ret)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['retVar'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['retVar']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution013_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """收益率偏度(Var_Skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['retSkew'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['retSkew']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution014_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """收益率峰度(Var_kurt)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['retKurt'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['retKurt']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution015_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        加权收盘价偏度(Close_Weight_Skew)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['closeVolWeightSkew'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial1',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['closeVolWeightSkew']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution016_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        单位一成交量占比熵(Vol_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['volEntropy'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['volEntropy']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution017_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        成交额占比熵(Amt_Entropy)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['amtEntropy'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['amtEntropy']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution018_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """每笔成交量差分均值(Vol_diff_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data1 = cls()._csv_data(data_name=['volPerDiffStd'],
                                file_path=FPN.HFD_midData.value,
                                file_name='TradeVol',
                                stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(data_name=['volume'],
                                file_path=FPN.HFD_depthVwap.value,
                                file_name='MarketData',
                                stock_id=KN.STOCK_ID.value)
        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['volPerDiffStd'] / data['volume'] * 240

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution019_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """每笔成交量差分绝对值均值(Vol_diff_abs_mean)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['volPerDiffAbsMean', 'volPerMean'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeVol',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['volPerDiffAbsMean'] / data['volPerMean']

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution020_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        朴素主动占比因子(Naïve_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['naiveAmtR'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['naiveAmtR']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution021_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        T分布主动占比因子(T_Amt_R)
        自由度采用样本长度减一
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['TAmtR'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['TAmtR']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution022_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        正态分布主动占比因子(N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['NAmtR'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['NAmtR']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution023_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        置信正态分布主动占比因子(C_N_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['CNAmtR'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['CNAmtR']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution024_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        均匀分布主动占比因子(Event_Amt_R)
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['EventAmtR'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['EventAmtR']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def Distribution025_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟成交量差分绝对值波动稳定性(HFD_vol_diff_abs_std)
        考虑集合竞价成交量波动较大，计算日内成交量波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data1 = cls()._csv_data(data_name=['volDiffAbsMean'],
                                file_path=FPN.HFD_midData.value,
                                file_name='TradeVol',
                                stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(data_name=['volume'],
                                file_path=FPN.HFD_depthVwap.value,
                                file_name='MarketData',
                                stock_id=KN.STOCK_ID.value)

        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['volDiffAbsMean'] / data['volume'] * 240

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution026_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟成交笔数差分绝对值波动稳定性(HFD_num_diff_abs_std)
        考虑集合竞价笔数实际意义可能不大，计算日内笔数波动时剔除集合竞价数据
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0',
                                          'tradeNumDiffAbsMean'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeTradeNum',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        data['tradeNumMean'] = (data['tradeNumRetUpSum_0'] +
                                data['tradeNumRetDownSum_0'] +
                                data['tradeNumRetEqualSum_0']) / 240
        res = data['tradeNumDiffAbsMean'] / data['tradeNumMean']

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def Distribution027_data_raw(cls,
                                 minute: int = 5,
                                 n: int = 20,
                                 **kwargs) -> pd.Series(float):
        """
        N日分钟振幅差分绝对值波动稳定性(HFD_ret_diff_abs_std)
        集合竞价只存在唯一价格，振幅为零, 剔除
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"
        data = cls()._csv_data(data_name=['retDiffAbsMean', 'retMean'],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeRet',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data['retDiffAbsMean'] / data['retMean']

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
                               file_path=FPN.HFD_depthVwap.value,
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
                               file_path=FPN.HFD_depthVwap.value,
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
