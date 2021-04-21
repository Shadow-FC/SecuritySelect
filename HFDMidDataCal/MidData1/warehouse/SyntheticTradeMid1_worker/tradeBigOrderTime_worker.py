# -*-coding:utf-8-*-
# @Time:   2021/4/14 13:47
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from utility.MidDataUtility import boxEdge


# 大单持续时间相关中间过程
def tradeBigOrderTime_worker(data: pd.DataFrame,
                             code: str,
                             date: str,
                             **kwargs) -> pd.Series(float):
    res = {'code': code,
           'date': date,
           'BigBuyEquTimePas': 0,
           'BigBuyAmtWetAllTime': 0,
           'BigBuyAmtWetPasTime': 0,
           'BigBuyEatSellNumWetAllTime': 0,
           'BigBuyEatSellNumWetPasTime': 0,
           'BigBuyTimeRet': 0,
           'BigBuyTimeRetCum': 0,
           'BigBuyTimeRetAmtWet': 0,
           'BigBuyTimeRetCumAmtWet': 0,
           'BigSellEquTimePas': 0,
           'BigSellAmtWetAllTime': 0,
           'BigSellAmtWetPasTime': 0,
           'BigSellEatBuyNumWetAllTime': 0,
           'BigSellEatBuyNumWetPasTime': 0,
           'BigSellTimeRet': 0,
           'BigSellTimeRetCum': 0,
           'BigSellTimeRetAmtWet': 0,
           'BigSellTimeRetCumAmtWet': 0,
           }

    data['Amt'] = data['Volume'] * data['Price']
    data['Time'] = data['Time'].map(lambda x: np.dot(np.array((x.split(':')), dtype=float), [3600, 60, 1]))

    # 买单和卖单挂单时长及造成的价格波动(累积和非累积)
    BuySta = data.groupby(['Type', 'BuyOrderID']).agg({"Time": ["first", "last", "count"],
                                                       "Price": ["first", "last"],
                                                       "Amt": "sum"})
    SellSta = data.groupby(['Type', 'SaleOrderID']).agg({"Time": ["first", "last", "count"],
                                                         "Price": ["first", "last"],
                                                         "Amt": "sum"})

    BuyRetCum = data.groupby('BuyOrderID')['Price'].last() / data.groupby('BuyOrderID')['Price'].first() - 1
    SellRetCum = data.groupby('SaleOrderID')['Price'].last() / data.groupby('SaleOrderID')['Price'].first() - 1
    BuyTime = (BuySta['Time']['last'] - BuySta['Time']['first']).unstack().T
    SellTime = (SellSta['Time']['last'] - SellSta['Time']['first']).unstack().T
    BuyRet = (BuySta['Price']['last'] / BuySta['Price']['first'] - 1).unstack().sum()
    SellRet = (SellSta['Price']['last'] / SellSta['Price']['first'] - 1).unstack().sum()

    BuyAmt, SellAmt = BuySta['Amt']['sum'].unstack().T, SellSta['Amt']['sum'].unstack().T
    BuyEatSellNum, SellEatBuyNum = BuySta['Time']['count'].unstack().T, SellSta['Time']['count'].unstack().T

    # 大单(双层过滤)
    if 'S' in BuyTime.columns:
        BuyTimeSub = boxEdge(BuyTime['S'].dropna())
        BuyBigID = boxEdge(BuyTimeSub)
    else:
        BuyBigID = pd.Series(dtype=object)
    if 'B' in SellTime.columns:
        SellTimeSub = boxEdge(SellTime['B'].dropna())
        SellBigID = boxEdge(SellTimeSub)
    else:
        SellBigID = pd.Series(dtype=object)

    if not BuyBigID.empty:
        # 等权，成交额加权，被动吃单额加权, 总吃单量加权，被动吃单量加权
        res['BigBuyEquTimePas'] = BuyBigID.sum()
        res['BigBuyAmtWetAllTime'] = np.average(BuyBigID, weights=BuyAmt.loc[BuyBigID.index].sum(axis=1))
        res['BigBuyAmtWetPasTime'] = np.average(BuyBigID, weights=BuyAmt.loc[BuyBigID.index, 'S'])
        res['BigBuyEatSellNumWetAllTime'] = np.average(BuyBigID,
                                                       weights=BuyEatSellNum.loc[BuyBigID.index].sum(axis=1))
        res['BigBuyEatSellNumWetPasTime'] = np.average(BuyBigID, weights=BuyEatSellNum.loc[BuyBigID.index, 'S'])
        # 大单造成的价格波动(累加, 非累加, 等权，成交量加权)
        res['BigBuyTimeRet'] = BuyRet.reindex(BuyBigID.index).sum()
        res['BigBuyTimeRetCum'] = BuyRetCum.reindex(BuyBigID.index).sum()
        res['BigBuyTimeRetAmtWet'] = np.average(BuyRet.reindex(BuyBigID.index),
                                                weights=BuyAmt.loc[BuyBigID.index, 'S'])
        res['BigBuyTimeRetCumAmtWet'] = np.average(BuyRetCum.reindex(BuyBigID.index),
                                                   weights=BuyAmt.loc[BuyBigID.index, 'S'])
    if not SellBigID.empty:
        # 等权，成交额加权，被动吃单额加权, 总吃单量加权，被动吃单量加权
        res['BigSellEquTimePas'] = SellBigID.sum()
        res['BigSellAmtWetAllTime'] = np.average(SellBigID, weights=SellAmt.loc[SellBigID.index].sum(axis=1))
        res['BigSellAmtWetPasTime'] = np.average(SellBigID, weights=SellAmt.loc[SellBigID.index, 'B'])
        res['BigSellEatBuyNumWetAllTime'] = np.average(SellBigID,
                                                       weights=SellEatBuyNum.loc[SellBigID.index].sum(axis=1))
        res['BigSellEatBuyNumWetPasTime'] = np.average(SellBigID, weights=SellEatBuyNum.loc[SellBigID.index, 'B'])
        # 大单造成的价格波动(累加, 非累加, 等权，成交量加权)
        res['BigSellTimeRet'] = SellRet.reindex(SellBigID.index).sum()
        res['BigSellTimeRetCum'] = SellRetCum.reindex(SellBigID.index).sum()
        res['BigSellTimeRetAmtWet'] = np.average(SellRet.reindex(SellBigID.index),
                                                 weights=SellAmt.loc[SellBigID.index, 'B'])
        res['BigSellTimeRetCumAmtWet'] = np.average(SellRetCum.reindex(SellBigID.index),
                                                    weights=SellAmt.loc[SellBigID.index, 'B'])

    res = pd.Series(res)
    return res
