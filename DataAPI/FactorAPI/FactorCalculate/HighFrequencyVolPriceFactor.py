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
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice009(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """改进反转(Rev_improve)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice011(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子(SmartQ)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice012(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """高频反转因子(HFD_Rev)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice013(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """轨迹非流动因子(Illiq_Track)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

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
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice015(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """结构化反转因子(Rev_struct)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def VolPrice016(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """聪明钱因子改进(SmartQ_ln)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

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
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

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
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

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
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

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
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    ####################################################################################################################
    @classmethod
    def VolPrice008_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             **kwargs):
        """大单驱动涨幅(MOM_bigOrder)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{str(q).replace('.', '')}q_{n}days"

        data = cls()._csv_data(data_name=["MOMBigOrder"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial1',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data["MOMBigOrder"]

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice009_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """改进反转(Rev_improve)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        data = cls()._csv_data(data_name=["close240m", "close30m"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeClose',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data["close240m"] / data['close30m'] - 1

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice011_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        聪明钱因子(SmartQ)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_1min_{n}days"

        data = cls()._csv_data(data_name=["SmartQ"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = data["SmartQ"]

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice012_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             **kwargs):
        """高频反转因子(HFD_Rev)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data1 = cls()._csv_data(data_name=["retVolWeight"],
                                file_path=FPN.HFD_midData.value,
                                file_name='TradeRet',
                                stock_id=KN.STOCK_ID.value)
        data2 = cls()._csv_data(data_name=[PVN.VOLUME.value],
                                file_path=FPN.HFD_depthVwap.value,
                                file_name='MarketData',
                                stock_id=KN.STOCK_ID.value)

        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        res = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

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
                             **kwargs):
        """轨迹非流动因子(Illiq_Track)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data1 = cls()._csv_data(data_name=["retD"],
                                file_path=FPN.HFD_midData.value,
                                file_name='TradeSpecial1',
                                stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(data_name=[PVN.AMOUNT.value],
                                file_path=FPN.HFD_depthVwap.value,
                                file_name='MarketData',
                                stock_id=KN.STOCK_ID.value)

        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        res = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).sum())
        res[factor_name] = res['retD'] / res[PVN.AMOUNT.value]
        return res[factor_name]

    @classmethod
    def VolPrice014_data_raw(cls,
                             minute: int = 5,
                             n: int = 21,
                             **kwargs):
        """加权收盘价比(Close_Weight)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{n}days"

        data = cls()._csv_data(data_name=["closeMean", "closeAmtWeight"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeClose',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['closeAmtWeight'] / data['closeMean']

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
                             **kwargs):
        """
        结构化反转因子(Rev_struct)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{minute}min_{str(ratio).replace('.', '')}R_{n}days"

        data = cls()._csv_data(data_name=["RevStruct"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial1',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['RevStruct']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod  # TODO 滚动10个交易日
    def VolPrice016_data_raw(cls,
                             n: int = 20,
                             **kwargs):
        """
        聪明钱因子改进(SmartQ_ln)
        原因子为过去N日分钟行情进行构建，而不是单日分钟行情构建后取过去N日平均值
        """
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_1min_{n}days"

        data = cls()._csv_data(data_name=["SmartQln"],
                               file_path=FPN.HFD_midData.value,
                               file_name='TradeSpecial2',
                               stock_id=KN.STOCK_ID.value)

        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        res = data['SmartQln']

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def VolPrice017_data_raw(cls,
                             **kwargs):
        """PMA 特殊"""
        data = cls()._csv_data(data_name=[PVN.OPEN.value, '2hPrice', '4hPrice'],
                               file_path=FPN.HFD_depthVwap.value,
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
            data_sub[['diffBidPrice1', 'diffAskPrice1', 'diffBidVol', 'diffAskVol']] = \
                data_sub[['code', 'bidprice1', 'askprice1', f'bidvolume{depth}sum', f'askvolume{depth}sum']].groupby(KN.STOCK_ID.value).diff(1)

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
            data_df = data_sub.pivot(columns='code', index='time', values='VOI')
            data_df_stand = data_df.sub(data_df.mean(axis=1), axis=0).div(data_df.std(axis=1), axis=0).stack()
            data_df_stand.name = 'VOI_stand'
            data_df_stand = data_df_stand.reset_index()
            data_sub = pd.merge(data_sub, data_df_stand, on=['time', 'code'], how='left')
            data_sub['VOI_stand'][np.isinf(data_sub['VOI_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['VOI_stand'].mean()

            return r

        Q = cls().csv_HFD_data(data_name=['bidprice1', 'askprice1'] + [f'bidvolume{depth}sum', f'askvolume{depth}sum'],
                               func=func,
                               file_path=FPN.HFD_depth1min.value)
        res = pd.concat(Q)
        res = cls().reindex(res).reset_index()
        # 滚动
        res = res.pivot(columns='code', index='date', values='VOI_stand')
        res = res.rolling(n, min_periods=round(n * 0.8)).apply(lambda x: x @ cls.weight_attenuation(len(x))).stack()

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

            data_sub['OIR'] = (data_sub[f'bidvolume{depth}sum'] - data_sub[f'askvolume{depth}sum']) / (
                    data_sub[f'bidvolume{depth}sum'] + data_sub[f'askvolume{depth}sum'])

            # 截面标准化
            data_df = data_sub.pivot(columns='code', index='time', values='OIR')
            data_df_stand = data_df.sub(data_df.mean(axis=1), axis=0).div(data_df.std(axis=1), axis=0).stack()
            data_df_stand.name = 'OIR_stand'
            data_df_stand = data_df_stand.reset_index()
            data_sub = pd.merge(data_sub, data_df_stand, on=['time', 'code'], how='left')

            data_sub['OIR_stand'][np.isinf(data_sub['OIR_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['OIR_stand'].mean()
            return r

        Q = cls().csv_HFD_data(data_name=[f'askvolume{depth}sum', f'bidvolume{depth}sum'],
                               func=func,
                               file_path=FPN.HFD_depth1min.value)
        res = pd.concat(Q)
        res = cls().reindex(res).reset_index()
        # 滚动
        res = res.pivot(columns='code', index='date', values='OIR_stand')
        res = res.rolling(n, min_periods=round(n * 0.8)).apply(lambda x: x @ cls.weight_attenuation(len(x))).stack()

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

            data_sub.set_index(['time', 'code'], inplace=True)
            data_sub['delta_MP'] = data_sub.groupby(KN.STOCK_ID.value)['MP'].apply(lambda x: x.rolling(2).mean())

            data_sub['MPB'] = data_sub['TP'] - data_sub['delta_MP']

            # 截面标准化
            data_df = data_sub['MPB'].unstack()
            data_df_stand = data_df.sub(data_df.mean(axis=1), axis=0).div(data_df.std(axis=1), axis=0).stack()
            data_df_stand.name = 'MPB_stand'
            data_df_stand = data_df_stand.reset_index()
            data_sub = pd.merge(data_sub, data_df_stand, on=['time', 'code'], how='left')

            data_sub['MPB_stand'][np.isinf(data_sub['MPB_stand'])] = 0

            # 转日频
            r = data_sub.groupby(KN.STOCK_ID.value)['MPB_stand'].mean()
            return r

        Q = cls().csv_HFD_data(
            data_name=[PVN.CLOSE.value, PVN.VOLUME.value, PVN.AMOUNT.value, 'bidprice1', 'askprice1'],
            func=func,
            file_path=FPN.HFD_depth1min.value)

        res = pd.concat(Q)
        res = cls().reindex(res).reset_index()
        # 滚动
        res = res.pivot(columns='code', index='date', values='MPB_stand')
        res = res.rolling(n, min_periods=round(n * 0.8)).apply(lambda x: x @ cls.weight_attenuation(len(x))).stack()

        res.name = factor_name
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
