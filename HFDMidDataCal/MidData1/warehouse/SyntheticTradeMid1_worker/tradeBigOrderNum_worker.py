# -*-coding:utf-8-*-
# @Time:   2021/4/14 13:41
# @Author: FC
# @Email:  18817289038@163.com


import numpy as np
import pandas as pd
from utility.MidDataUtility import boxEdge


# 大单吃单量相关中间过程
def tradeBigOrderNum_worker(data: pd.DataFrame,
                            code: str,
                            date: str,
                            **kwargs) -> pd.Series(float):
    """
    当天总订单量/成交额加权订单量(买/卖)
    (成交额加权/等权)大单(主动/被动/总)吃单量(买/卖)
    (等权/成交额加权)大单引起的价格波动(累积/非累计)(买/卖)
    """
    res = {
        "code": code,
        "date": date,

        "AllNumSum": 0,

        "BuyEatNumAct": 0,
        "BuyEatNumPas": 0,
        "BigBuyEatNumAllAmtWet": 0,
        "BigBuyEatNumActAmtWet": 0,
        "BigBuyEatNumPasAmtWet": 0,
        "AllBuyNumAmtWet": 0,

        "BigBuyEatSellRet": 0,
        "BigBuyEatSellRetCum": 0,
        "BigBuyEatSellRetAmtWet": 0,
        "BigBuyEatSellRetCumAmtWet": 0,

        "SellEatNumAct": 0,
        "SellEatNumPas": 0,
        "BigSellEatNumAllAmtWet": 0,
        "BigSellEatNumActAmtWet": 0,
        "BigSellEatNumPasAmtWet": 0,
        "AllSellNumAmtWet": 0,

        "BigSellEatBuyRet": 0,
        "BigSellEatBuyRetCum": 0,
        "BigSellEatBuyRetAmtWet": 0,
        "BigSellEatBuyRetCumAmtWet": 0,
    }
    data['Amt'] = data['Volume'] * data['Price']
    # 买单和卖单吃单量, 买单和卖单造成的价格波动(累积和非累积)
    BuyEatSell = data.groupby('BuyOrderID').agg({"SaleOrderID": 'count', "Amt": 'sum', "Price": ['last', 'first']})
    SellEatBuy = data.groupby('SaleOrderID').agg({"BuyOrderID": 'count', "Amt": 'sum', "Price": ['last', 'first']})

    BuyPrice = data.groupby(['Type', 'BuyOrderID']).agg({"Price": ['last', 'first']})
    SellPrice = data.groupby(['Type', 'SaleOrderID']).agg({"Price": ['last', 'first']})

    BuyRetCum = BuyEatSell['Price']['last'] / BuyEatSell['Price']['first'] - 1
    SellRetCum = SellEatBuy['Price']['last'] / SellEatBuy['Price']['first'] - 1

    BuyRet = (BuyPrice['Price']['last'] / BuyPrice['Price']['first'] - 1).unstack().sum()
    SellRet = (SellPrice['Price']['last'] / SellPrice['Price']['first'] - 1).unstack().sum()

    # 大单(双层过滤)(主动，被动, 总)
    BuyEatSellNum = boxEdge(BuyEatSell['SaleOrderID']['count'])
    SellEatBuyNum = boxEdge(SellEatBuy['BuyOrderID']['count'])
    BuyBigID, SellBigID = boxEdge(BuyEatSellNum), boxEdge(SellEatBuyNum)

    BuyBigData = data[data['BuyOrderID'].isin(BuyBigID.index)]
    SellBigData = data[data['SaleOrderID'].isin(SellBigID.index)]

    BuyBigAct, BuyBigPas = BuyBigData[BuyBigData['Type'] == 'B'], BuyBigData[BuyBigData['Type'] == 'S']
    SellBigAct, SellBigPas = SellBigData[SellBigData['Type'] == 'S'], SellBigData[SellBigData['Type'] == 'B']

    BuyBigActID, BuyBigPasID = BuyBigAct['BuyOrderID'].drop_duplicates(), BuyBigPas['BuyOrderID'].drop_duplicates()
    SellBigActID = SellBigAct['SaleOrderID'].drop_duplicates()
    SellBigPasID = SellBigPas['SaleOrderID'].drop_duplicates()

    # 主动吃单数, 被动吃单数, 总订单数(买卖), 等权
    res["AllNumSum"] = data.shape[0]
    res["BuyEatNumAct"], res["BuyEatNumPas"] = BuyBigAct.shape[0], BuyBigPas.shape[0]
    res["SellEatNumAct"], res["SellEatNumPas"] = SellBigAct.shape[0], SellBigPas.shape[0]

    # 成交额加权吃单量-总
    if not BuyEatSell.empty:
        res["AllBuyNumAmtWet"] = np.average(BuyEatSell['SaleOrderID']['count'], weights=BuyEatSell['Amt']['sum'])
    if not SellEatBuy.empty:
        res["AllSellNumAmtWet"] = np.average(SellEatBuy['BuyOrderID']['count'], weights=SellEatBuy['Amt']['sum'])

    # 吃单量按照成交额加权/订单造成的收益率变动(成交额加权/等权)-大单
    if not BuyBigID.empty:
        res["BigBuyEatNumAllAmtWet"] = np.average(BuyEatSell.loc[BuyBigID.index, 'SaleOrderID']['count'],
                                                  weights=BuyEatSell.loc[BuyBigID.index, 'Amt']['sum'])
        res['BigBuyEatSellRet'] = BuyRet.reindex(BuyBigID.index).sum()
        res['BigBuyEatSellRetCum'] = BuyRetCum.reindex(BuyBigID.index).sum()
        res['BigBuyEatSellRetAmtWet'] = np.average(BuyRet.reindex(BuyBigID.index),
                                                   weights=BuyEatSell.loc[BuyBigID.index, 'Amt']['sum'])
        res['BigBuyEatSellRetCumAmtWet'] = np.average(BuyRetCum.reindex(BuyBigID.index),
                                                      weights=BuyEatSell.loc[BuyBigID.index, 'Amt']['sum'])
    if not SellBigID.empty:
        res["BigSellEatNumAllAmtWet"] = np.average(SellEatBuy.loc[SellBigID.index, 'BuyOrderID']['count'],
                                                   weights=SellEatBuy.loc[SellBigID.index, 'Amt']['sum'])
        res['BigSellEatBuyRet'] = SellRet.reindex(SellBigID.index).sum()
        res['BigSellEatBuyRetCum'] = SellRetCum.reindex(SellBigID.index).sum()
        res['BigSellEatBuyRetAmtWet'] = np.average(SellRet.reindex(SellBigID.index),
                                                   weights=SellEatBuy.loc[SellBigID.index, 'Amt']['sum'])
        res['BigSellEatBuyRetCumAmtWet'] = np.average(SellRetCum.reindex(SellBigID.index),
                                                      weights=SellEatBuy.loc[SellBigID.index, 'Amt']['sum'])
    # 主动被动吃单量成交额加权
    if not BuyBigActID.empty:
        res["BigBuyEatNumActAmtWet"] = np.average(BuyEatSell.loc[BuyBigActID, 'SaleOrderID']['count'],
                                                  weights=BuyEatSell.loc[BuyBigActID, 'Amt']['sum'])
    if not BuyBigPasID.empty:
        res["BigBuyEatNumPasAmtWet"] = np.average(BuyEatSell.loc[BuyBigPasID, 'SaleOrderID']['count'],
                                                  weights=BuyEatSell.loc[BuyBigPasID, 'Amt']['sum'])
    if not SellBigActID.empty:
        res["BigSellEatNumActAmtWet"] = np.average(SellEatBuy.loc[SellBigActID, 'BuyOrderID']['count'],
                                                   weights=SellEatBuy.loc[SellBigActID, 'Amt']['sum'])
    if not SellBigPasID.empty:
        res["BigSellEatNumPasAmtWet"] = np.average(SellEatBuy.loc[SellBigPasID, 'BuyOrderID']['count'],
                                                   weights=SellEatBuy.loc[SellBigPasID, 'Amt']['sum'])

    res = pd.Series(res)
    return res

