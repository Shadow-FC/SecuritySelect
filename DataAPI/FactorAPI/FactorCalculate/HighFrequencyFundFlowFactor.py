import pandas as pd
import datetime as dt
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


class HighFrequencyFundFlowFactor(FactorBase):
    """
    高频因子
    """

    callAM = '09:30:00'
    callPM = '15:00:00'

    def __init__(self):
        super(HighFrequencyFundFlowFactor, self).__init__()
        self.range = lambda x: (x[KN.TRADE_TIME.value] >= self.callAM) & (x[KN.TRADE_TIME.value] < self.callPM)

    @classmethod
    def FundFlow001(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔成交金额(AMTperTRD)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow002(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流入金额占比(AMTperTRD_IN_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow003(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流出金额占比(AMTperTRD_OUT_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow004(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均单笔流入流出金额之比(AMTperTRD_IN_OUT)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow005(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单资金净流入金额(AMT_NetIN_bigOrder)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow006(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """大单资金净流入率(AMT_NetIN_bigOrder_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    # 先用主买代替
    @classmethod
    def FundFlow009(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        大买成交金额占比(MFD_buy_Nstd_R)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['buyBig'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['bigBuyOccupy'] = data['buyBig'] / data[PVN.AMOUNT.value]

        data = cls().reindex(data)
        data[factor_name] = data['bigBuyOccupy'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    # 先用主卖代替
    @classmethod
    def FundFlow010(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """
        大卖成交金额占比(MFD_sell_Nstd_R)
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['saleBig'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']
        data['bigSaleOccupy'] = data['saleBig'] / data[PVN.AMOUNT.value]

        data = cls().reindex(data)
        data[factor_name] = data['bigSaleOccupy'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    # 先用主买主卖代替
    @classmethod
    def FundFlow011(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """大买大卖成交金额占比差值(MFD_buy_sell_R_sub)"""

        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['buyBig'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['saleBig'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']

        data['bigBuyOccupy'] = data['buyBig'] / data[PVN.AMOUNT.value]
        data['bigSaleOccupy'] = data['saleBig'] / data[PVN.AMOUNT.value]
        data['diff'] = data['bigBuyOccupy'] - data['bigSaleOccupy']

        data = cls().reindex(data)
        data[factor_name] = data['diff'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    # 先用主买主卖代替
    @classmethod
    def FundFlow012(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """大单成交金额占比(MFD_buy_sell_R_add)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['buyBig'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['saleBig'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']

        data['bigBuyOccupy'] = data['buyBig'] / data[PVN.AMOUNT.value]
        data['bigSaleOccupy'] = data['saleBig'] / data[PVN.AMOUNT.value]
        data['sub'] = data['bigBuyOccupy'] + data['bigSaleOccupy']

        data = cls().reindex(data)
        data[factor_name] = data['sub'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow013(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """开盘连续竞价成交占比(HFD_callVol_O_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow018(cls,
                    data: pd.DataFrame,
                    period: str = 'all',
                    n: int = 20,
                    **kwargs):
        """主买占比(buy_drive_prop)"""
        factor_name = sys._getframe().f_code.co_name + f'_{period}_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        if period == 'all':
            data['ratio'] = (data['BuyAll_AM_120min'] + data['BuyAll_PM_120min']) / data[PVN.AMOUNT.value]
        elif period == 'open':
            data['ratio'] = data['BuyAll_AM_30min'] / (data['BuyAll_AM_30min'] + data['SaleAll_AM_30min'])
        elif period == 'between':
            data['ratio'] = (data['BuyAll_AM_120min'] - data['BuyAll_AM_30min'] +
                             data['BuyAll_PM_120min'] - data['BuyAll_PM_30min']) / \
                            (data[PVN.AMOUNT.value] - data['BuyAll_AM_30min'] - data['SaleAll_AM_30min'] -
                             data['BuyAll_PM_30min'] - data['SaleAll_PM_30min'])
        elif period == 'close':
            data['ratio'] = data['buyamount'] / data[PVN.AMOUNT.value]
        else:
            return

        data = cls().reindex(data)
        data[factor_name] = data['ratio'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow019(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """主买强度(buy_strength)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow020(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """净主买强度(net_strength_stand)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow025(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """剔除大卖的大买成交金额占比(HFD_buy_big_R)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
        data['sale_small'] = data['SaleAll_AM_120min'] + data['SaleAll_PM_120min'] - \
                             data['SaleBigOrderMeanStd_AM_120min'] - data['SaleBigOrderMeanStd_PM_120min']

        data['big_buy_occupy'] = (data['buy_Big'] + data['sale_small']) / data[PVN.AMOUNT.value]

        data = cls().reindex(data)
        data[factor_name] = data['big_buy_occupy'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow026(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """剔除大买的大卖成交金额占比(HFD_sell_big_R)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['sale_Big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']
        data['buy_small'] = data['BuyAll_AM_120min'] + data['BuyAll_PM_120min'] - \
                            data['BuyBigOrderMeanStd_AM_120min'] - data['BuyBigOrderMeanStd_PM_120min']

        data['big_sale_occupy'] = (data['sale_Big'] + data['buy_small']) / data[PVN.AMOUNT.value]

        data = cls().reindex(data)
        data[factor_name] = data['big_sale_occupy'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    # @classmethod
    # def FundFlow027(cls,
    #                 data: pd.DataFrame,
    #                 n: int = 20,
    #                 **kwargs):
    #
    #     """大买大卖成交金额占比(HFD_buy_sell_big_R)"""
    #     factor_name = sys._getframe().f_code.co_name + f'_{n}days'
    #     data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
    #     data = data.sort_index()
    #
    #     data['buy_Big'] = data['BuyBigOrderMeanStd_AM_120min'] + data['BuyBigOrderMeanStd_PM_120min']
    #     data['sale_big'] = data['SaleBigOrderMeanStd_AM_120min'] + data['SaleBigOrderMeanStd_PM_120min']
    #
    #     data['big_occupy'] = (data['buy_Big'] + data['sale_big']) / data[PVN.AMOUNT.value]
    #
    #     data = cls().reindex(data)
    #     data[factor_name] = data['big_occupy'].groupby(KN.STOCK_ID.value,
    #                                                    group_keys=False).rolling(n, min_periods=round(n * 0.8)).mean()
    #
    #     F = DataInfo()
    #     F.data = data[factor_name]
    #     F.data_type = 'FundFlow'
    #     F.data_category = cls().__class__.__name__
    #     F.data_name = factor_name
    #
    #     return F

    # @classmethod
    # def FundFlow028(cls,
    #                 data: pd.DataFrame,
    #                 **kwargs):
    #
    #     """尾盘成交占比(Vol_prop_tail)"""
    #
    #     F = DataInfo()
    #     F.data = data
    #     F.data_type = 'FundFlow'
    #     F.data_category = cls().__class__.__name__
    #     F.data_name = data.name
    #
    #     return F

    @classmethod
    def FundFlow029(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):

        """开盘后净主买上午占比(buy_amt_open_am)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['ratio'] = (data['BuyAll_AM_30min'] - data['SaleAll_AM_30min']) / (
                data['BuyAll_AM_30min'] + data['SaleAll_AM_30min'])

        data = cls().reindex(data)
        data[factor_name] = data['ratio'].groupby(KN.STOCK_ID.value).apply(
            lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'FundFlow'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    # TODO 原因子为成交量
    @classmethod
    def FundFlow032(cls,
                    data: pd.DataFrame,
                    n: int = 21,
                    **kwargs):
        """
        博弈因子(Stren):存在涨停主卖为零情况，会导致分母为0，根据数值特征范围将分母为零的计算设置为2
        """
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['buy_amount'] = data['BuyAll_AM_120min'] + data['BuyAll_PM_120min']
        data['sale_amount'] = data['SaleAll_AM_120min'] + data['SaleAll_PM_120min']

        # 升序
        data = cls().reindex(data)
        data[factor_name] = data.groupby(KN.STOCK_ID.value,
                                         group_keys=False,
                                         as_index=False).apply(lambda x: cls().Stren(x, n))

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow033(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """开盘X分钟成交占比(Open_X_vol)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow034(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """收盘X分钟成交占比(Close_X_vol)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow035(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """资金流向(CashFlow)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow039(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """W切割反转因子(Rev_W)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['ret'] = data['4hPrice'].groupby(KN.STOCK_ID.value).pct_change(fill_method=None)
        data = data.dropna()
        data = data.groupby(KN.STOCK_ID.value,
                            group_keys=False).apply(lambda x: cls.W_cut(x, 'AmountMean', 'ret', n))
        data[factor_name] = data['M_high'] - data['M_low']

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow040(cls,
                    data: pd.DataFrame,
                    n: int = 20,
                    **kwargs):
        """高分位W反转因子(Rev_W_HQ)"""
        factor_name = sys._getframe().f_code.co_name + f'_{n}days'
        data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
        data = data.sort_index()

        data['ret'] = data['close'].groupby(KN.STOCK_ID.value).pct_change(fill_method=None)
        data = data.dropna()
        data = data.groupby(KN.STOCK_ID.value,
                            group_keys=False).apply(lambda x: cls.W_cut(x, 'AmountQuantile_9', 'ret', n))
        data[factor_name] = data['M_high'] - data['M_low']

        F = DataInfo()
        F.data = data[factor_name]
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = factor_name

        return F

    @classmethod
    def FundFlow046(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均净委买变化率(bid_mean_R)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow047(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """净委买变化率波动率(bid_R_std)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    @classmethod
    def FundFlow048(cls,
                    data: pd.DataFrame,
                    **kwargs):
        """平均净委买变化率偏度(bid_R_skew)"""

        F = DataInfo()
        F.data = data
        F.data_type = 'HFD'
        F.data_category = cls().__class__.__name__
        F.data_name = data.name

        return F

    ####################################################################################################################
    @classmethod
    def FundFlow001_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """平均单笔成交金额(AMTperTRD)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                    lambda x: x[PVN.AMOUNT.value].sum() / x['tradenum'].sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeTradeNum',
                                    stock_id=KN.STOCK_ID.value)
            data2 = cls()._csv_data(data_name=['amount'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            data['tradeNumSum'] = data['tradeNumRetUpSum_0'] + data['tradeNumRetDownSum_0'] + data[
                'tradeNumRetEqualSum_0']
            res = data['amount'] / data['tradeNumSum']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow002_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """平均单笔流入金额占比(AMTperTRD_IN_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def inflow(d: pd.DataFrame) -> float:
                d_inflow = d[d[KN.RETURN.value] > 0]

                d_inflow_amt, d_inflow_num = (d_inflow[PVN.AMOUNT.value]).sum(), (d_inflow['tradenum']).sum()
                d_all_amt, d_all_num = (d[PVN.AMOUNT.value]).sum(), (d['tradenum']).sum()

                if d_inflow_num != 0 and d_all_amt != 0:
                    inflow_r = d_inflow_amt * d_all_num / d_inflow_num / d_all_amt
                    return inflow_r

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(inflow)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['amtRetUpSum_0', 'amtRetDownSum_0', 'amtRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)
            data2 = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeTradeNum',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            data['amountSum'] = data['amtRetUpSum_0'] + data['amtRetDownSum_0'] + data['amtRetEqualSum_0']
            data['tradeNumSum'] = data['tradeNumRetUpSum_0'] + data['tradeNumRetDownSum_0'] + data[
                'tradeNumRetEqualSum_0']

            res = data['amtRetUpSum_0'] * data['tradeNumSum'] / data['tradeNumRetUpSum_0'] / data['amountSum']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow003_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """平均单笔流出金额占比(AMTperTRD_OUT_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def outflow(d: pd.DataFrame) -> float:
                d_outflow = d[d[KN.RETURN.value] < 0]
                d_outflow_amt, d_outflow_num = (d_outflow[PVN.AMOUNT.value]).sum(), (d_outflow['tradenum']).sum()
                d_all_amt, d_all_num = (d[PVN.AMOUNT.value]).sum(), (d['tradenum']).sum()

                if d_outflow_num != 0 and d_all_amt != 0:
                    outflow_r = d_outflow_amt * d_all_num / d_outflow_num / d_all_amt
                    return outflow_r

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(outflow)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['amtRetUpSum_0', 'amtRetDownSum_0', 'amtRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)
            data2 = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeTradeNum',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            data['amountSum'] = data['amtRetUpSum_0'] + data['amtRetDownSum_0'] + data['amtRetEqualSum_0']
            data['tradeNumSum'] = data['tradeNumRetUpSum_0'] + data['tradeNumRetDownSum_0'] + data[
                'tradeNumRetEqualSum_0']

            res = data['amtRetDownSum_0'] * data['tradeNumSum'] / data['tradeNumRetDownSum_0'] / data['amountSum']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow004_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """平均单笔流入流出金额之比(AMTperTRD_IN_OUT)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def ratio(d: pd.DataFrame) -> float:
                d_inflow, d_outflow = d[d[KN.RETURN.value] > 0], d[d[KN.RETURN.value] < 0]

                d_inflow_amt, d_inflow_num = (d_inflow[PVN.AMOUNT.value]).sum(), (d_inflow['tradenum']).sum()
                d_outflow_amt, d_outflow_num = (d_outflow[PVN.AMOUNT.value]).sum(), (d_outflow['tradenum']).sum()

                if d_inflow_num != 0 and d_outflow_amt != 0:
                    ratio_ = d_inflow_amt * d_outflow_num / d_inflow_num / d_outflow_amt
                    return ratio_

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(ratio)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['amtRetUpSum_0', 'amtRetDownSum_0', 'amtRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)
            data2 = cls()._csv_data(data_name=['tradeNumRetUpSum_0', 'tradeNumRetDownSum_0', 'tradeNumRetEqualSum_0'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeTradeNum',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['amtRetUpSum_0'] * data['tradeNumRetDownSum_0'] / data['tradeNumRetUpSum_0'] / data[
                'amtRetDownSum_0']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow005_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """大单资金净流入金额(AMT_NetIN_bigOrder)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{str(q).replace('.', '')}q_{n}days"

        if method == 'raw':
            def big_order(d: pd.DataFrame) -> float:
                d_sub = d[d['amt_per_min'] >= d['amt_per_min'].quantile(1 - q)]
                d_sub_inflow, d_sub_outflow = d_sub[d_sub[KN.RETURN.value] > 0], d_sub[d_sub[KN.RETURN.value] < 0]
                netInflow = d_sub_inflow[PVN.AMOUNT.value].sum() - d_sub_outflow[PVN.AMOUNT.value].sum()
                return netInflow

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['amt_per_min'] = d_sub[PVN.AMOUNT.value] / d_sub['tradenum']
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(big_order)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)

            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=['AMTInFlowBigOrder', 'AMTOutFlowBigOrder'],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['AMTInFlowBigOrder'] - data['AMTOutFlowBigOrder']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow006_data_raw(cls,
                             n: int = 20,
                             q: float = 0.2,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """大单资金净流入率(AMT_NetIN_bigOrder_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{str(q).replace('.', '')}q_{n}days"

        if method == 'raw':
            def big_order(d: pd.DataFrame) -> float:
                d_sub = d[d['amt_per_min'] >= d['amt_per_min'].quantile(1 - q)]
                d_sub_inflow, d_sub_outflow = d_sub[d_sub[KN.RETURN.value] > 0], d_sub[d_sub[KN.RETURN.value] < 0]
                netInflow_R = (d_sub_inflow[PVN.AMOUNT.value].sum() - d_sub_outflow[PVN.AMOUNT.value].sum()) / \
                              d[PVN.AMOUNT.value].sum()
                return netInflow_R

            def func(d: pd.DataFrame) -> pd.Series(float):
                d_sub = d[cls().range]
                d_sub[KN.RETURN.value] = d_sub.groupby(KN.STOCK_ID.value,
                                                       group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
                d_sub['amt_per_min'] = d_sub[PVN.AMOUNT.value] / d_sub['tradenum']
                r = d_sub.groupby(KN.STOCK_ID.value, group_keys=False).apply(big_order)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value, 'tradenum'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['AMTInFlowBigOrder', 'AMTOutFlowBigOrder'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeSpecial1',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['amount'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = (data['AMTInFlowBigOrder'] - data['AMTOutFlowBigOrder']) / data['amount']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow009_data_raw(cls,
                             **kwargs):
        """大买成交金额占比(MFD_buy_Nstd_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id=KN.STOCK_ID.value)

        res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

        return res

    @classmethod
    def FundFlow010_data_raw(cls,
                             **kwargs):
        """大卖成交金额占比(MFD_sell_Nstd_R)"""
        data1 = cls()._csv_data(
            data_name=['SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id=KN.STOCK_ID.value)

        res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

        return res

    @classmethod
    def FundFlow011_data_raw(cls,
                             **kwargs):
        """大买大卖成交金额占比差值(MFD_buy_sell_R_sub)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id=KN.STOCK_ID.value)

        res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

        return res

    @classmethod
    def FundFlow012_data_raw(cls,
                             **kwargs):
        """大单成交金额占比(MFD_buy_sell_R_add)"""
        return cls.FundFlow011_data_raw()

    @classmethod
    def FundFlow013_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs) -> pd.Series(float):
        """开盘连续竞价成交占比(HFD_callVol_O_R)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def call_volume(d: pd.DataFrame) -> float:
                call = d[d[KN.TRADE_TIME.value] < cls().callAM]
                return call[PVN.AMOUNT.value].sum() / d[PVN.AMOUNT.value].sum()

            def func(d: pd.DataFrame) -> pd.Series(float):
                r = d.groupby(KN.STOCK_ID.value, group_keys=False).apply(call_volume)
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=['amtAM_call'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['amount'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data['amtAM_call'] / data['amount']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow018_data_raw(cls,
                             period: str = 'all',
                             method: str = 'mid',
                             **kwargs) -> pd.DataFrame:
        """
        主买占比(buy_drive_prop)
        """
        if period != 'close':
            if period == 'all':
                name_list = ['BuyAll_AM_120min', 'BuyAll_PM_120min']
            elif period == 'open':
                name_list = ['BuyAll_AM_30min', 'SaleAll_AM_30min']
            elif period == 'between':
                name_list = ['BuyAll_AM_30min', 'BuyAll_AM_120min', 'BuyAll_PM_30min', 'BuyAll_PM_120min',
                             'SaleAll_AM_30min', 'SaleAll_PM_30min']
            else:
                name_list = ['BuyAll_AM_30min']
                print(f'Input error:{period}')

            data1 = cls()._csv_data(
                data_name=name_list,
                file_path=FPN.HFD_Stock_CF.value,
                file_name='CashFlowIntraday',
                stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(
                data_name=[PVN.AMOUNT.value],
                file_path=FPN.HFD_Stock_CF.value,
                file_name='MarketData',
                stock_id=KN.STOCK_ID.value)

            res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        else:
            if method == 'raw':
                def tail_volume(d: pd.DataFrame) -> pd.DataFrame:
                    d_sub = d[d[KN.TRADE_TIME.value] >= '14:30:00']
                    return d_sub[['buyamount', PVN.AMOUNT.value]].sum()

                def func(d: pd.DataFrame) -> pd.DataFrame:
                    d = d.dropna()
                    r = d.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
                    return r

                Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value, 'buyamount'],
                                       func=func,
                                       file_path=FPN.HFD_Stock_M.value)
                res = pd.concat(Q)
            elif method == 'mid':
                data1 = cls()._csv_data(data_name=['buyAmtSumPM_30min'],
                                        file_path=FPN.HFD_MidData.value,
                                        file_name='TradeBuyAmtSum',
                                        stock_id=KN.STOCK_ID.value)

                data2 = cls()._csv_data(data_name=['amtPM_30min'],
                                        file_path=FPN.HFD_MidData.value,
                                        file_name='TradeAmtSum',
                                        stock_id=KN.STOCK_ID.value)

                data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
                res = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
                res = res.rename(columns={"buyAmtSumPM_30min": "buyamount", "amtPM_30min": PVN.AMOUNT.value})

            else:
                print('Parameter is wrong!')
                res = pd.Series()

            res = cls().reindex(res)
            res = res.reset_index()
        return res

    @classmethod
    def FundFlow019_data_raw(cls,
                             period: str = 'all',
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """主买强度(buy_strength)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f'_{period}_{n}days'

        if period == 'all':
            beg, end = '09:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumPM_all"]
            buyAmtStd = ["buyAmtStd_all"]
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
            buyAmtSum = ["buyAmtSumAM_30min"]
            buyAmtStd = ["buyAmtStd_open"]
        elif period == 'between':
            beg, end = '10:00:00', '14:29:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumAM_30min", "buyAmtSumPM_all", "buyAmtSumPM_30min"]
            buyAmtStd = ["buyAmtStd_between"]
        elif period == 'close':
            beg, end = '14:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumPM_30min"]
            buyAmtStd = ["buyAmtStd_close"]
        else:
            beg, end = '09:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumPM_all"]
            buyAmtStd = ["buyAmtStd_all"]
            print(f'Input error:{period}')

        if method == 'raw':
            def tail_volume(d: pd.DataFrame):
                d_sub = d[(d[KN.TRADE_TIME.value] >= beg) & (d[KN.TRADE_TIME.value] <= end)]
                if d_sub['buyamount'].std() != 0:
                    return d_sub['buyamount'].mean() / d_sub['buyamount'].std()

            def func(d: pd.DataFrame):
                d = d.dropna()
                r = d.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
                return r

            Q = cls().csv_HFD_data(data_name=['buyamount'],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)
        elif method == 'mid':
            data1 = cls()._csv_data(data_name=buyAmtSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeBuyAmtSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=buyAmtStd,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtStd',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            if period == 'all':
                res = data[buyAmtSum].sum(axis=1) / 240 / data["buyAmtStd_all"]
            elif period == 'open':
                res = data[buyAmtSum].sum(axis=1) / 30 / data["buyAmtStd_open"]
            elif period == 'between':
                data["buyAM"] = data["buyAmtSumAM_all"] - data["buyAmtSumAM_30min"]
                data["buyPM"] = data["buyAmtSumPM_all"] - data["buyAmtSumPM_30min"]
                res = (data["buyAM"] + data["buyPM"]) / 180 / data["buyAmtStd_between"]
            elif period == 'close':
                res = data[buyAmtSum].sum(axis=1) / 30 / data["buyAmtStd_close"]
            else:
                res = data[buyAmtSum].sum(axis=1) / 240 / data["buyAmtStd_all"]
                print(f'Input error:{period}')
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def FundFlow020_data_raw(cls,
                             period: str = 'all',
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """净主买强度(net_strength_stand)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f'_{period}_{n}days'

        if period == 'all':
            beg, end = '09:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumPM_all"]
            sellAmtSum = ["sellAmtSumAM_all", "sellAmtSumPM_all"]
            netAmtStd = ["netAmtStd_all"]
        elif period == 'open':
            beg, end = '09:30:00', '09:59:00'
            buyAmtSum = ["buyAmtSumAM_30min"]
            sellAmtSum = ["sellAmtSumAM_30min"]
            netAmtStd = ["netAmtStd_open"]
        elif period == 'between':
            beg, end = '10:00:00', '14:29:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumAM_30min", "buyAmtSumPM_all", "buyAmtSumPM_30min"]
            sellAmtSum = ["sellAmtSumAM_all", "sellAmtSumAM_30min", "sellAmtSumPM_all", "sellAmtSumPM_30min"]
            netAmtStd = ["netAmtStd_between"]
        elif period == 'close':
            beg, end = '14:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumPM_30min"]
            sellAmtSum = ["sellAmtSumPM_30min"]
            netAmtStd = ["netAmtStd_close"]
        else:
            beg, end = '09:30:00', '15:00:00'
            buyAmtSum = ["buyAmtSumAM_all", "buyAmtSumPM_all"]
            sellAmtSum = ["sellAmtSumAM_all", "sellAmtSumPM_all"]
            netAmtStd = ["netAmtStd_all"]
            print(f'Input error:{period}')

        if method == 'raw':
            def tail_volume(d: pd.DataFrame):
                d_sub = d[(d[KN.TRADE_TIME.value] >= beg) &
                          (d[KN.TRADE_TIME.value] <= end)]
                d_sub['net'] = 2 * d_sub['buyamount'] - d_sub[PVN.AMOUNT.value]
                if d_sub['net'].std() != 0:
                    return d_sub['net'].mean() / d_sub['net'].std()

            def func(d: pd.DataFrame):
                d = d.dropna()
                r = d.groupby(KN.STOCK_ID.value, group_keys=False).apply(tail_volume)
                return r

            Q = cls().csv_HFD_data(data_name=['buyamount', PVN.AMOUNT.value],
                                   func=func,
                                   file_path=FPN.HFD_Stock_M.value)
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=buyAmtSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeBuyAmtSum',
                                    stock_id=KN.STOCK_ID.value)
            data2 = cls()._csv_data(data_name=sellAmtSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeSellAmtSum',
                                    stock_id=KN.STOCK_ID.value)

            data3 = cls()._csv_data(data_name=netAmtStd,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtStd',
                                    stock_id=KN.STOCK_ID.value)

            data_ = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = pd.merge(data_, data3, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            if period == 'all':
                res = (data[buyAmtSum].sum(axis=1) - data[sellAmtSum].sum(axis=1)) / 240 / data["netAmtStd_all"]
            elif period == 'open':
                res = (data[buyAmtSum].sum(axis=1) - data[sellAmtSum].sum(axis=1)) / 30 / data["netAmtStd_open"]
            elif period == 'between':
                data["buyAM"] = data["buyAmtSumAM_all"] - data["buyAmtSumAM_30min"]
                data["buyPM"] = data["buyAmtSumPM_all"] - data["buyAmtSumPM_30min"]
                data["sellAM"] = data["sellAmtSumAM_all"] - data["sellAmtSumAM_30min"]
                data["sellPM"] = data["sellAmtSumPM_all"] - data["sellAmtSumPM_30min"]
                res = (data["buyAM"] + data["buyPM"] - data['sellAM'] - data['sellPM']) / 180 / data[
                    "netAmtStd_between"]
            elif period == 'close':
                res = (data[buyAmtSum].sum(axis=1) - data[sellAmtSum].sum(axis=1)) / 30 / data["netAmtStd_close"]
            else:
                res = (data[buyAmtSum].sum(axis=1) - data[sellAmtSum].sum(axis=1)) / 240 / data["netAmtStd_all"]
                print(f'Input error:{period}')
        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def FundFlow025_data_raw(cls,
                             **kwargs):
        """剔除大卖的大买成交金额占比(HFD_buy_big_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min',
                       'SaleAll_AM_120min', 'SaleAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id=KN.STOCK_ID.value)

        res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

        return res

    @classmethod
    def FundFlow026_data_raw(cls,
                             **kwargs):
        """剔除大买的大卖成交金额占比(HFD_sell_big_R)"""
        data1 = cls()._csv_data(
            data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
                       'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min',
                       'BuyAll_AM_120min', 'BuyAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(
            data_name=[PVN.AMOUNT.value],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='MarketData',
            stock_id=KN.STOCK_ID.value)

        res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

        return res

    # @classmethod
    # def FundFlow027_data_raw(cls,
    #                          **kwargs):
    #     """大买大卖成交金额占比(HFD_buy_sell_big_R)"""
    #     data1 = cls()._csv_data(
    #         data_name=['BuyBigOrderMeanStd_AM_120min', 'BuyBigOrderMeanStd_PM_120min',
    #                    'SaleBigOrderMeanStd_AM_120min', 'SaleBigOrderMeanStd_PM_120min'],
    #         file_path=FPN.HFD_Stock_CF.value,
    #         file_name='CashFlowIntraday',
    #         stock_id=KN.STOCK_ID.value)
    #
    #     data2 = cls()._csv_data(
    #         data_name=[PVN.AMOUNT.value],
    #         file_path=FPN.HFD_Stock_CF.value,
    #         file_name='MarketData',
    #         stock_id=KN.STOCK_ID.value)
    #
    #     res = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
    #
    #     return res

    # @classmethod
    # def FundFlow028_data_raw(cls,
    #                          n: int = 20,
    #                          method: str = 'mid',
    #                          **kwargs) -> pd.Series(float):
    #     """尾盘成交占比(Vol_prop_tail)"""
    #     factor_name = sys._getframe().f_code.co_name[: -9] + f'_{n}days'
    #
    #     if method == 'raw':
    #         def call_volume(d: pd.DataFrame) -> float:
    #             vol_tail = d[d[KN.TRADE_TIME.value] >= '14:30:00']
    #             return vol_tail[PVN.AMOUNT.value].sum() / d[PVN.AMOUNT.value].sum()
    #
    #         def func(d: pd.DataFrame) -> pd.Series(float):
    #             r = d.groupby(KN.STOCK_ID.value, group_keys=False).apply(call_volume)
    #             return r
    #
    #         Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value],
    #                                func=func,
    #                                file_path=FPN.HFD_Stock_M.value)
    #         res = pd.concat(Q)
    #     elif method == 'mid':
    #         data1 = cls()._csv_data(data_name=['amtPM_30min'],
    #                                 file_path=FPN.HFD_MidData.value,
    #                                 file_name='TradeAmtSum',
    #                                 stock_id=KN.STOCK_ID.value)
    #
    #         data2 = cls()._csv_data(data_name=['amount'],
    #                                 file_path=FPN.HFD_Stock_Depth.value,
    #                                 file_name='MarketData',
    #                                 stock_id=KN.STOCK_ID.value)
    #
    #         data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
    #         data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
    #
    #         res = data['amtPM_30min'] / data['amount']
    #     else:
    #         print('Parameter is wrong!')
    #         res = pd.Series()
    #
    #     res[np.isinf(res)] = np.nan
    #     res = cls().reindex(res)
    #     res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
    #     res.name = factor_name
    #
    #     return res

    @classmethod
    def FundFlow029_data_raw(cls,
                             **kwargs):
        """开盘后净主买上午占比(buy_amt_open_am)"""

        res = cls()._csv_data(
            data_name=['BuyAll_AM_30min', 'SaleAll_AM_30min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        return res

    @classmethod
    def FundFlow032_data_raw(cls,
                             **kwargs):
        """博弈因子(Stren)"""
        data = cls()._csv_data(
            data_name=['BuyAll_AM_120min', 'SaleAll_AM_120min', 'BuyAll_PM_120min', 'SaleAll_PM_120min'],
            file_path=FPN.HFD_Stock_CF.value,
            file_name='CashFlowIntraday',
            stock_id=KN.STOCK_ID.value)

        return data

    @classmethod
    def FundFlow033_data_raw(cls,
                             x_min: int = 5,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """开盘X分钟成交占比(Open_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{x_min}min_O_{n}days"

        if method == 'raw':
            end = (dt.datetime(2020, 12, 31, 9, 30) + dt.timedelta(minutes=x_min)).strftime("%H:%M:%S")

            def func(d: pd.DataFrame):
                r = d.groupby(KN.STOCK_ID.value).apply(lambda x: cls.Volume_Percentage(x, cls().callAM, end))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value], func=func)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=[f'amtAM_{x_min}min', 'amtAM_call'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['amount'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = (data[f'amtAM_{x_min}min'] - data['amtAM_call']) / data['amount']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow034_data_raw(cls,
                             x_min: int = 5,
                             n: int = 20,
                             method: str = 'raw',
                             **kwargs):
        """收盘X分钟成交占比(Close_X_vol)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{x_min}min_C_{n}days"

        if method == 'raw':
            star = (dt.datetime(2020, 12, 31, 15, 00) - dt.timedelta(minutes=x_min)).strftime("%H:%M:%S")

            def func(d: pd.DataFrame):
                r = d.groupby([KN.STOCK_ID.value]).apply(lambda x: cls.Volume_Percentage(x, star, cls().callPM))
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.AMOUNT.value], func=func)

            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=[f'amtPM_{x_min}min'],
                                    file_path=FPN.HFD_MidData.value,
                                    file_name='TradeAmtSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=['amount'],
                                    file_path=FPN.HFD_Stock_Depth.value,
                                    file_name='MarketData',
                                    stock_id=KN.STOCK_ID.value)

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data[f'amtPM_{x_min}min'] / data['amount']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow035_data_raw(cls,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """资金流向(CashFlow)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{n}days"

        if method == 'raw':
            def func(d: pd.DataFrame):
                d_sub = d[cls().range]
                d_sub["closeDiff"] = d_sub.groupby(KN.STOCK_ID.value, group_keys=False)[PVN.CLOSE.value].diff(1)
                r = d_sub.groupby(KN.STOCK_ID.value,
                                  group_keys=False).apply(
                    lambda x: (np.sign(x['closeDiff']) * x[PVN.AMOUNT.value]).sum() / (x[PVN.AMOUNT.value]).sum())
                return r

            Q = cls().csv_HFD_data(data_name=[PVN.CLOSE.value, PVN.AMOUNT.value], func=func)

            res = pd.concat(Q)

        elif method == 'mid':
            data = cls()._csv_data(data_name=["CashFlow"],
                                   file_path=FPN.HFD_MidData.value,
                                   file_name='TradeSpecial1',
                                   stock_id=KN.STOCK_ID.value)

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            res = data["CashFlow"]

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow039_data_raw(cls,
                             **kwargs):
        """W切割反转因子(Rev_W)"""
        data = cls()._csv_data(data_name=['AmountMean', '4hPrice'],
                               file_path=FPN.HFD_Stock_Depth.value,
                               file_name='VwapFactor',
                               stock_id=KN.STOCK_ID.value)
        return data

    @classmethod
    def FundFlow040_data_raw(cls,
                             **kwargs):
        """高分位W反转因子(Rev_W_HQ)"""

        data1 = cls()._csv_data(data_name=['AmountQuantile_9'],
                                file_path=FPN.HFD_Stock_CF.value,
                                file_name='CashFlowStat',
                                stock_id=KN.STOCK_ID.value)

        data2 = cls()._csv_data(data_name=[PVN.CLOSE.value],
                                file_path=FPN.HFD_Stock_CF.value,
                                file_name='MarketData',
                                stock_id=KN.STOCK_ID.value)

        data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')
        return data

    @classmethod  # 还未改
    def FundFlow046_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """平均净委买变化率(bid_mean_R)"""

        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{period}_{n}days"

        # list 字段顺序不能颠倒
        if period == 'all':
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
        elif period == 'open':
            beg, end = '09:30:00', '10:00:00'
            bidVolSum = [f"bid{depth}VolSum_30m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_30m", f"ask{depth}VolSum_0m"]
        elif period == 'between':
            beg, end = '10:00:00', '14:30:00'
            bidVolSum = [f"bid{depth}VolSum_210m", f"bid{depth}VolSum_30m"]
            askVolSum = [f"ask{depth}VolSum_210m", f"ask{depth}VolSum_30m"]
        elif period == 'close':
            beg, end = '14:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_210m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_210m"]
        else:
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
            print(f'Input error:{period}')

        if method == 'raw':
            bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
            askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

            def func(d: pd.DataFrame, **kwargs_sub):
                liq_stock = kwargs_sub['liqStock']
                d_sub = d[(d[KN.TRADE_TIME.value] >= beg) & (d[KN.TRADE_TIME.value] <= end)]
                d_sub['orderDiff'] = d_sub[bidvolume].sum(axis=1) - d_sub[askvolume].sum(axis=1)
                order_diff = d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].last() - \
                             d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == d[KN.TRADE_DATE.value][0]]
                liq_stock_sub = liq_stock_sub.set_index(KN.STOCK_ID.value)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['orderDiff'] / r_merge['liqStock']

                return r_res

            data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

            data_day['liqStock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

            Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                                   func=func,
                                   file_path=FPN.HFD_Stock_Depth_1min.value,
                                   fun_kwargs={
                                       "liqStock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liqStock']]})
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=bidVolSum + askVolSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name=f'Depth{depth}VolSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])
            data2['liqStock'] = data2[PVN.LIQ_MV.value] / data2[PVN.CLOSE.value]

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            data['bidDiff'] = data[bidVolSum[0]] - data[bidVolSum[1]]
            data['askDiff'] = data[askVolSum[0]] - data[askVolSum[1]]

            res = (data['bidDiff'] - data['askDiff']) / data['liqStock']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        res = res.groupby(KN.STOCK_ID.value).apply(lambda x: x.rolling(n, min_periods=round(n * 0.8)).mean())
        res.name = factor_name
        return res

    @classmethod
    def FundFlow047_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """净委买变化率波动率(bid_R_std)"""

        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{period}_{n}days"

        if period == 'all':
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
        elif period == 'open':
            beg, end = '09:30:00', '10:00:00'
            bidVolSum = [f"bid{depth}VolSum_30m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_30m", f"ask{depth}VolSum_0m"]
        elif period == 'between':
            beg, end = '10:00:00', '14:30:00'
            bidVolSum = [f"bid{depth}VolSum_210m", f"bid{depth}VolSum_30m"]
            askVolSum = [f"ask{depth}VolSum_210m", f"ask{depth}VolSum_30m"]
        elif period == 'close':
            beg, end = '14:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_210m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_210m"]
        else:
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
            print(f'Input error:{period}')

        if method == 'raw':

            bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
            askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

            def func(d: pd.DataFrame, **kwargs_sub):
                liq_stock = kwargs_sub['liqStock']
                d_sub = d[(d[KN.TRADE_TIME.value] >= beg) & (d[KN.TRADE_TIME.value] <= end)]
                d_sub['orderDiff'] = d_sub[bidvolume].sum(axis=1) - d_sub[askvolume].sum(axis=1)
                order_diff = d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].last() - \
                             d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == d[KN.TRADE_DATE.value][0]]
                liq_stock_sub = liq_stock_sub.set_index(KN.STOCK_ID.value)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['orderDiff'] / r_merge['liqStock']

                return r_res

            data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

            data_day['liqStock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

            Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                                   func=func,
                                   file_path=FPN.HFD_Stock_Depth_1min.value,
                                   fun_kwargs={
                                       "liqStock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liqStock']]})
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=bidVolSum + askVolSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name=f'Depth{depth}VolSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])
            data2['liqStock'] = data2[PVN.LIQ_MV.value] / data2[PVN.CLOSE.value]

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            data['bidDiff'] = data[bidVolSum[0]] - data[bidVolSum[1]]
            data['askDiff'] = data[askVolSum[0]] - data[askVolSum[1]]

            res = (data['bidDiff'] - data['askDiff']) / data['liqStock']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        if n >= 2:
            res = res.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                lambda x: x.rolling(min(n, 5), min_periods=round(n * 0.8)).std())
        res.name = factor_name

        return res

    @classmethod
    def FundFlow048_data_raw(cls,
                             period: str = 'all',
                             depth: int = 5,
                             n: int = 20,
                             method: str = 'mid',
                             **kwargs):
        """平均净委买变化率偏度(bid_R_skew)"""
        factor_name = sys._getframe().f_code.co_name[: -9] + f"_{depth}depth_{period}_{n}days"

        if period == 'all':
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
        elif period == 'open':
            beg, end = '09:30:00', '10:00:00'
            bidVolSum = [f"bid{depth}VolSum_30m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_30m", f"ask{depth}VolSum_0m"]
        elif period == 'between':
            beg, end = '10:00:00', '14:30:00'
            bidVolSum = [f"bid{depth}VolSum_210m", f"bid{depth}VolSum_30m"]
            askVolSum = [f"ask{depth}VolSum_210m", f"ask{depth}VolSum_30m"]
        elif period == 'close':
            beg, end = '14:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_210m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_210m"]
        else:
            beg, end = '09:30:00', '15:00:00'
            bidVolSum = [f"bid{depth}VolSum_240m", f"bid{depth}VolSum_0m"]
            askVolSum = [f"ask{depth}VolSum_240m", f"ask{depth}VolSum_0m"]
            print(f'Input error:{period}')

        if method == 'raw':

            bidvolume = [f'bidvolume{i}' for i in range(1, depth + 1)]
            askvolume = [f'askvolume{i}' for i in range(1, depth + 1)]

            def func(d: pd.DataFrame, **kwargs_sub):
                liq_stock = kwargs_sub['liqStock']
                d_sub = d[(d[KN.TRADE_TIME.value] >= beg) & (d[KN.TRADE_TIME.value] <= end)]
                d_sub['orderDiff'] = d_sub[bidvolume].sum(axis=1) - d_sub[askvolume].sum(axis=1)
                order_diff = d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].last() - \
                             d_sub.groupby(KN.STOCK_ID.value)['orderDiff'].first()
                liq_stock_sub = liq_stock[liq_stock[KN.TRADE_DATE.value] == d[KN.TRADE_DATE.value][0]]
                liq_stock_sub = liq_stock_sub.set_index(KN.STOCK_ID.value)

                r_merge = pd.merge(order_diff, liq_stock_sub, left_index=True, right_index=True, how='left')

                r_res = r_merge['orderDiff'] / r_merge['liqStock']

                return r_res

            data_day = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])

            data_day['liqStock'] = data_day[PVN.LIQ_MV.value] / data_day[PVN.CLOSE.value]

            Q = cls().csv_HFD_data(data_name=bidvolume + askvolume,
                                   func=func,
                                   file_path=FPN.HFD_Stock_Depth_1min.value,
                                   fun_kwargs={
                                       "liqStock": data_day[[KN.TRADE_DATE.value, KN.STOCK_ID.value, 'liqStock']]})
            res = pd.concat(Q)

        elif method == 'mid':
            data1 = cls()._csv_data(data_name=bidVolSum + askVolSum,
                                    file_path=FPN.HFD_MidData.value,
                                    file_name=f'Depth{depth}VolSum',
                                    stock_id=KN.STOCK_ID.value)

            data2 = cls()._csv_data(data_name=[PVN.LIQ_MV.value, PVN.CLOSE.value])
            data2['liqStock'] = data2[PVN.LIQ_MV.value] / data2[PVN.CLOSE.value]

            data = pd.merge(data1, data2, on=[KN.TRADE_DATE.value, KN.STOCK_ID.value], how='inner')

            data = data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            data['bidDiff'] = data[bidVolSum[0]] - data[bidVolSum[1]]
            data['askDiff'] = data[askVolSum[0]] - data[askVolSum[1]]

            res = (data['bidDiff'] - data['askDiff']) / data['liqStock']

        else:
            print('Parameter is wrong!')
            res = pd.Series()

        res[np.isinf(res)] = np.nan
        res = cls().reindex(res)
        if n >= 5:
            res = res.groupby(KN.STOCK_ID.value, group_keys=False).apply(
                lambda x: x.rolling(min(n, 5), min_periods=round(n * 0.8)).skew())
        res.name = factor_name

        return res

    @staticmethod
    def Volume_Percentage(data: pd.DataFrame,
                          time_star: str,
                          time_end: str):
        data_copy = data.copy(deep=True)
        # 注意时间切片：左闭右开
        data_copy_sub = data_copy[(data_copy[KN.TRADE_TIME.value] >= time_star) &
                                  (data_copy[KN.TRADE_TIME.value] < time_end)]
        res = data_copy_sub[PVN.AMOUNT.value].sum() / data_copy[PVN.AMOUNT.value].sum()
        return res

    # 半衰权重
    @staticmethod
    def Half_time(period: int,
                  decay: int = 2) -> list:

        weight_list = [pow(2, (i - period - 1) / decay) for i in range(1, period + 1)]

        weight_1 = [i / sum(weight_list) for i in weight_list]

        return weight_1

    @staticmethod
    def W_cut(d: pd.DataFrame,
              rank_name: str,
              cut_name: str,
              n: int):
        d['ret_cum'] = d[cut_name].rolling(n).sum()
        for i in range(1, n):
            d[rank_name + f"_{i}"] = d[rank_name].shift(i)

        C = [c_ for c_ in d.columns if rank_name in c_]
        J = d[C].ge(d[C].median(axis=1), axis=0)
        d[J], d[~J] = 1, 0

        for j in range(0, n):
            d[C[j]] = d[cut_name].shift(j) * d[C[j]]

        d['M_high'] = d[C].sum(axis=1)
        d['M_low'] = d['ret_cum'] - d['M_high']

        d = d.dropna()
        return d[['M_high', 'M_low']]

    def Stren(self, data: pd.DataFrame, n: int) -> pd.Series(float):
        weight = sorted(self.Half_time(period=n), reverse=True)

        buy_amount = self.switchForm(data['buy_amount'], n)
        sale_amount = self.switchForm(data['sale_amount'], n)

        buy_amount_w = (buy_amount * weight).sum(axis=1)
        sale_amount_w = (sale_amount * weight).sum(axis=1)

        res = buy_amount_w / (buy_amount_w + sale_amount_w)

        buy_amount_count, sale_amount_count = buy_amount.count(axis=1), sale_amount.count(axis=1)
        res = res[(buy_amount_count >= round(n * 0.8)) & (sale_amount_count >= round(n * 0.8))]
        return res


if __name__ == '__main__':
    pass
