import pandas as pd
import numpy as np
import sys

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
若不做特殊说明， 分钟级别数据运算会包含集合竞价信息
"""


class HighFrequencyVolPriceFactor(FactorBase):
    """
    高频因子
    """

    callAM = '09:30:00'
    callPM = '15:00:00'

    def __init__(self):
        super(HighFrequencyVolPriceFactor, self).__init__()
        self.range = lambda x: (x[KN.TRADE_TIME.value] >= self.callAM) & (x[KN.TRADE_TIME.value] < self.callPM)

    @classmethod
    def VolPrice008(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单驱动涨幅(MOM_bigOrder)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice009(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """改进反转(Rev_improve)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice011(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子(SmartQ)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice012(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """高频反转因子(HFD_Rev)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice013(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """轨迹非流动因子(Illiq_Track)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice014(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        加权收盘价比(Close_Weight)
        默认一分钟频率
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice015(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """结构化反转因子(Rev_struct)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice016(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子改进(SmartQ_ln)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice017(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """PMA 特殊"""
        factor_name = sys._getframe().f_code.co_name + f"_{n}days"
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        # Calculate AM and PM returns on stocks
        data['am_ret_stock'] = data['2hPrice'] / data[PVN.OPEN.value] - 1
        data['pm_ret_stock'] = data['4hPrice'] / data['2hPrice'] - 1

        # filter momentum
        # data[factor_name] = data.groupby(KN.TRADE_DATE.value,
        #                                  group_keys=False).apply(lambda x: cls._reg(x, 'am_ret_stock', 'pm_ret_stock'))
        # data['mean'] = data['res'].groupby(KN.STOCK_ID.value,
        #                                    group_keys=False).rolling(n, min_periods=1).apply(np.nanmean)
        # data['std'] = data['res'].groupby(KN.STOCK_ID.value,
        #                                   group_keys=False).rolling(n, min_periods=1).apply(np.nanstd)
        # data[factor_name] = data['mean'] / data['std']
        # data[factor_name][np.isinf(data[factor_name])] = 0
        data = cls().reindex(data)
        data[factor_name] = data['pm_ret_stock'].groupby(KN.TRADE_DATE.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = factor_name

        return F

    @classmethod
    def VolPrice018(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        订单失衡(HFD_VOI)
        日频转月频需要采用衰减加权的方式
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice019(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        订单失衡率(HFD_OLR)
        日频转月频需要采用衰减加权的方式
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    @classmethod
    def VolPrice020(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """
        市价偏离率(HFD_MPB)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        集合竞价会存在0盘口，用前值填充
        """

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.factor_category = cls().__class__.__name__
        F.factor_name = data.name

        return F

    ####################################################################################################################
    @classmethod
    def VolPrice008_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             method: str = 'mid',
                             **kwargs):
        """大单驱动涨幅(MOM_bigOrder)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{str(q).replace('.', '')}q_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['amtPer'] = d_sub[PVN.AMOUNT.value] / d_sub['tradenum']
                r = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: (x[x['amtPer'] >= x['amtPer'].quantile(1 - q)][KN.RETURN.value] + 1).prod(min_count=1))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.OPEN.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["MOMBigOrder"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data["MOMBigOrder"]
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice009_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """改进反转(Rev_improve)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d = d.dropna()
                d_sub = d[d[KN.TRADE_TIME.value] >= '10:00:00']
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].apply(
                    lambda x: np.nan if len(x) == 1 else x.iloc[-1] / x.iloc[0] - 1)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["close240m", "close30m"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeClose',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data["close240m"] / data['close30m'] - 1
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice011_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """
        聪明钱因子(SmartQ)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_1min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(cls.func_M_sqrt)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)

            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["SmartQ"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data["SmartQ"]
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice012_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             method: str = 'mid',
                             **kwargs):
        """高频反转因子(HFD_Rev)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r1 = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: (x[KN.RETURN.value] * x[PVN.VOLUME.value]).sum() / x[PVN.VOLUME.value].sum())
                r2 = d_sub.groupby(KN.STOCK_ID.value)[PVN.VOLUME.value].sum()
                r = pd.concat([r1, r2], axis=1)
                r.columns = ['retVolWeight', PVN.VOLUME.value]
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=["retVolWeight"],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeRet',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=[PVN.VOLUME.value],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            res = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res['rev_vol'] = res['retVolWeight'] * res[PVN.VOLUME.value]
        res = cls().reindex(res)
        res_sub = res[['rev_vol', PVN.VOLUME.value]].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())
        res[factor_name] = res_sub['rev_vol'] / res_sub[PVN.VOLUME.value]
        return res[factor_name]

    @classmethod
    def VolPrice013_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             method: str = 'mid',
                             **kwargs):
        """轨迹非流动因子(Illiq_Track)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                r1 = d_sub.groupby(KN.STOCK_ID.value).apply(
                    lambda x: np.log(1 + abs(np.log(x[PVN.CLOSE.value] / x[PVN.CLOSE.value].shift(1)))).sum())
                r2 = d_sub.groupby(KN.STOCK_ID.value)[PVN.AMOUNT.value].sum()
                r = pd.concat([r1, r2], axis=1)
                r.columns = ['retD', PVN.AMOUNT.value]
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=["retD"],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeSpecial1',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=[PVN.AMOUNT.value],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            res = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())
        res[factor_name] = res['retD'] / res[PVN.AMOUNT.value]
        return res[factor_name]

    @classmethod
    def VolPrice014_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             method: str = 'mid',
                             **kwargs):
        """加权收盘价比(Close_Weight)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: (x[PVN.CLOSE.value] * x[PVN.AMOUNT.value]).sum() /
                              (x[PVN.CLOSE.value].mean() * x[PVN.AMOUNT.value].sum()))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)

            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["closeMean", "closeAmtWeight"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeClose',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['closeAmtWeight'] / data['closeMean']
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())
        res.name = factor_name

        return res

    @classmethod  # TODO 可能需要滚动
    def VolPrice015_data_raw(cls,
                             n: int = 20,
                             minute: int = 5,
                             ratio: float = 0.1,
                             method: str = 'mid',
                             **kwargs):
        """
        结构化反转因子(Rev_struct)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{str(ratio).replace('.', '')}R_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)

                rev_struct = d_sub.groupby(KN.STOCK_ID.value).apply(cls.func_Structured_reversal, ratio)
                return rev_struct

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value],
                                   func=func,
                                   file_path=FPN.HFD.value,
                                   sub_file=f"{minute}minute")

            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["RevStruct"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['RevStruct']
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice016_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """
        聪明钱因子改进(SmartQ_ln)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_1min_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value).apply(cls.func_M_ln)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.VOLUME.value], func=func)

            res = pd.concat(Q)
        elif method == 'mid':
            data = cls()._csv_data(data_name=["SmartQln"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial2',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            res = data['SmartQln']
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice017_data_raw(cls,
                             **kwargs):
        """PMA 特殊"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id=KN.STOCK_ID.value)

        # 2h 数据存在异常
        data_s = data[~((data['2hPrice'] == 0) | (np.isnan(data['2hPrice'])))]

        return data_s

    @classmethod
    def VolPrice018_data_raw(cls,
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """
        订单失衡(HFD_VOI)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{n}days"

        def func(data: pd.DataFrame):
            print(data[KN.TRADE_DATE.value].iloc[0])
            data_sub = data[data[KN.TRADE_TIME.value] >= '09:30:00']
            # data_sub['bid_Vol_weight'] = data_sub[bidvolume] @ cls.weight_attenuation(depth)
            # data_sub['ask_Vol_weight'] = data_sub[askvolume] @ cls.weight_attenuation(depth)
            data_sub[['diffBidPrice1', 'diffAskPrice1',
                      'diffBidVol', 'diffAskVol']] = data_sub.groupby(KN.STOCK_ID.value,
                                                                      group_keys=False).apply(
                lambda x: x[['bidprice1', 'askprice1', f'bidvolume{depth}sum', f'askvolume{depth}sum']].diff(1))
            data_sub = data_sub.dropna()

            data_sub[['bid_judge', 'ask_judge']] = np.sign(data_sub[['diffBidPrice1', 'diffAskPrice1']])

            bid_equal = data_sub[data_sub['bid_judge'] == 0]['diffBidVol']
            bid_small = pd.Series(data=0, index=data_sub[data_sub['bid_judge'] < 0]['diffBidVol'].index, name='diffBidVol')
            bid_more = data_sub[data_sub['bid_judge'] > 0][f'bidvolume{depth}sum']

            ask_equal = data_sub[data_sub['ask_judge'] == 0]['diffAskVol']
            ask_small = pd.Series(data=0, index=data_sub[data_sub['ask_judge'] > 0]['diffAskVol'].index, name='diffAskVol')
            ask_more = data_sub[data_sub['ask_judge'] < 0][f'askvolume{depth}sum']
            data_sub['delta_V_bid'] = pd.concat([bid_equal, bid_small, bid_more])
            data_sub['delta_V_ask'] = pd.concat([ask_equal, ask_small, ask_more])
            data_sub['VOI'] = data_sub['delta_V_bid'] - data_sub['delta_V_ask']

            # 截面标准化
            data_sub['VOI_stand'] = data_sub.groupby(KN.TRADE_TIME.value,
                                                     group_keys=False).apply(
                lambda x: (x['VOI'] - x['VOI'].mean()) / x['VOI'].std())

            data_sub['VOI_stand'][np.isinf(data_sub['VOI_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['VOI_stand'].mean()

            return r

        Q = cls().csv_HFD_data(data_name=['bidprice1', 'askprice1'] + [f'bidvolume{depth}sum', f'askvolume{depth}sum'],
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value)
        res = pd.concat(Q)
        res = cls().reindex(res)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=round(n * 0.8)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))

        res.name = factor_name
        return res

    @classmethod
    def VolPrice019_data_raw(cls,
                             depth: int = 5,
                             n: int = 20,
                             **kwargs):
        """
        订单失衡率(HFD_OLR)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{n}days"

        def func(data: pd.DataFrame):
            print(data[KN.TRADE_DATE.value].iloc[0])
            data_sub = data[data[KN.TRADE_TIME.value] >= '09:30:00']
            # data_sub['bid_Vol_weight'] = data_sub[bidvolume] @ cls.weight_attenuation(depth)
            # data_sub['ask_Vol_weight'] = data_sub[askvolume] @ cls.weight_attenuation(depth)

            data_sub['OIR'] = (data_sub[f'bidvolume{depth}sum'] - data_sub[f'askvolume{depth}sum']) / (
                    data_sub[f'bidvolume{depth}sum'] + data_sub[f'askvolume{depth}sum'])

            # 截面标准化
            data_sub['OIR_stand'] = data_sub.groupby(KN.TRADE_TIME.value,
                                                     group_keys=False).apply(
                lambda x: (x['OIR'] - x['OIR'].mean()) / x['OIR'].std())

            data_sub['OIR_stand'][np.isinf(data_sub['OIR_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['OIR_stand'].mean()
            return r

        Q = cls().csv_HFD_data(data_name=[f'askvolume{depth}sum', f'bidvolume{depth}sum'],
                               func=func,
                               file_path=FPN.HFD_Stock_Depth_1min.value)
        res = pd.concat(Q)
        res = cls().reindex(res)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=round(n * 0.8)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))

        res.name = factor_name
        return res

    @classmethod
    def VolPrice020_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        市价偏离率(HFD_MPB)
        日频转月频需要采用衰减加权的方式
        剔除开盘集合竞价
        集合竞价会存在0盘口，用前值填充
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        def func(data: pd.DataFrame):
            print(data[KN.TRADE_DATE.value].iloc[0])
            data_sub = data[data[KN.TRADE_TIME.value] >= '09:30:00']

            data_sub['TP'] = data_sub[PVN.AMOUNT.value] / data_sub[PVN.VOLUME.value]
            data_sub['TP'] = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)['TP'].ffill()

            data_sub['MP'] = (data_sub['bidprice1'] + data_sub['askprice1']) / 2
            data_sub['MP'][data_sub['MP'] == 0] = np.nan
            data_sub['MP'] = data_sub.groupby(KN.STOCK_ID.value, group_keys=False)['MP'].ffill()

            data_sub['delta_MP'] = data_sub[[KN.STOCK_ID.value, 'MP']].groupby(KN.STOCK_ID.value,
                                                                               group_keys=False).rolling(2).mean()

            data_sub['MPB'] = data_sub['TP'] - data_sub['delta_MP']

            # 截面标准化
            data_sub['MPB_stand'] = data_sub.groupby(KN.TRADE_TIME.value,
                                                     group_keys=False).apply(
                lambda x: (x['MPB'] - x['MPB'].mean()) / x['MPB'].std())

            data_sub['MPB_stand'][np.isinf(data_sub['MPB_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['MPB_stand'].mean()
            return r

        Q = cls().csv_HFD_data(
            data_name=[PVN.CLOSE.value, PVN.VOLUME.value, PVN.AMOUNT.value, 'bidprice1', 'askprice1'],
            func=func,
            file_path=FPN.HFD_Stock_Depth_1min.value)

        res = pd.concat(Q)
        res = cls().reindex(res)
        # 滚动
        res = res.groupby(KN.STOCK_ID.value,
                          group_keys=False).rolling(n, min_periods=round(n * 0.8)).apply(
            lambda x: x @ cls.weight_attenuation(len(x)))

        res.name = factor_name
        return res

    @staticmethod
    def func_Structured_reversal(data: pd.DataFrame,
                                 ratio: float):

        data = data.sort_values(PVN.VOLUME.value, ascending=True)
        data['cum_volume'] = data[PVN.VOLUME.value].cumsum() / data[PVN.VOLUME.value].sum()

        # momentum
        data_mom = data[data['cum_volume'] <= ratio]
        rev_mom = (data_mom['ret'] * (1 / data_mom[PVN.VOLUME.value])).sum() / (1 / data_mom[PVN.VOLUME.value]).sum()

        # Reverse
        data_rev = data[data['cum_volume'] > ratio]
        rev_rev = (data_rev['ret'] * (data_rev[PVN.VOLUME.value])).sum() / (data_rev[PVN.VOLUME.value]).sum()

        rev_struct = rev_rev - rev_mom
        if np.isnan(rev_struct):
            print("Nan error!")
        return rev_struct

    @staticmethod
    def func_M_ln(data: pd.DataFrame):

        data['S'] = abs(data[PVN.CLOSE.value].pct_change(fill_method=None)) / np.log(data[PVN.VOLUME.value])
        VWAP = (data[PVN.CLOSE.value] * data[PVN.VOLUME.value] / (data[PVN.VOLUME.value]).sum()).sum()
        data = data.sort_values('S', ascending=False)
        data['cum_volume_R'] = data[PVN.VOLUME.value].cumsum() / (data[PVN.VOLUME.value]).sum()
        data_ = data[data['cum_volume_R'] <= 0.2]
        res = (data_[PVN.CLOSE.value] * data_[PVN.VOLUME.value] / (data_[PVN.VOLUME.value]).sum()).sum() / VWAP

        return res

    @staticmethod
    def func_M_sqrt(data: pd.DataFrame):

        # 可能存在分钟线丢失
        data['S'] = abs(data[PVN.CLOSE.value].pct_change(fill_method=None)) / np.sqrt(data[PVN.VOLUME.value])
        VWAP = (data[PVN.CLOSE.value] * data[PVN.VOLUME.value] / (data[PVN.VOLUME.value]).sum()).sum()
        data = data.sort_values('S', ascending=False)
        data['cum_volume_R'] = data[PVN.VOLUME.value].cumsum() / (data[PVN.VOLUME.value]).sum()
        data_ = data[data['cum_volume_R'] <= 0.2]
        res = (data_[PVN.CLOSE.value] * data_[PVN.VOLUME.value] / (data_[PVN.VOLUME.value]).sum()).sum() / VWAP

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

    @staticmethod
    def weight_attenuation(n: int = 5):
        W_sum = sum(i for i in range(1, n + 1))
        W = [i / W_sum for i in range(1, n + 1)]
        return W


if __name__ == '__main__':
    pass
