# -*-coding:utf-8-*-
# @Time:   2021/4/14 14:29
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd


def tradeInvF_005_ZBQX_worker(data: pd.DataFrame,
                              code: str,
                              date: str,
                              **kwargs) -> pd.Series(float):
    res = {'code': code, 'date': date}
    #  数据准备
    data['Amount'] = data['Volume'] * data['Price']
    transData = data[data['Time'] >= '09:30:00'].copy()
    sellOrderAmt = transData.groupby('SaleOrderID')['Amount'].sum()
    buyOrderAmt = transData.groupby('BuyOrderID')['Amount'].sum()
    sellOrderTotalNum = sellOrderAmt.shape[0]
    buyOrderTotalNum = buyOrderAmt.shape[0]
    res['sellOrderTotalNum'] = sellOrderTotalNum
    res['buyOrderTotalNum'] = buyOrderTotalNum
    # 大单、超大单划分分位点
    superSellOrderQuantile = sellOrderAmt.quantile([0.99]).iloc[0]
    superBuyOrderQuantile = buyOrderAmt.quantile([0.99]).iloc[0]
    largeSellOrderQuantile = sellOrderAmt.quantile([0.95]).iloc[0]
    largeBuyOrderQuantile = buyOrderAmt.quantile([0.95]).iloc[0]
    ##################
    #### 指标计算 ####
    ##################
    # 超大卖单统计指标
    superSellOrderID = sellOrderAmt[sellOrderAmt >= superSellOrderQuantile].reset_index()[['SaleOrderID']]
    if superSellOrderID.empty:
        pass
    else:
        superSellOrder = pd.merge(transData, superSellOrderID, on=['SaleOrderID'], how='inner')
        superSellOrderTotalNum = superSellOrderID.shape[0]
        # 超大卖单交易笔数指标
        superSellOrderEachTransNumDf = superSellOrder.groupby(['Type', 'SaleOrderID'])[
            'Volume'].count().reset_index().rename(columns={'Volume': 'count'})
        superSellOrderTotalTransNum = superSellOrderEachTransNumDf['count'].sum()
        superSellOrderSellTransNum = superSellOrderEachTransNumDf[superSellOrderEachTransNumDf['Type'] == 'S'][
            'count'].sum()
        superSellOrderBuyTransNum = superSellOrderEachTransNumDf[superSellOrderEachTransNumDf['Type'] == 'B'][
            'count'].sum()
        # 超大卖单成交额指标
        superSellOrderEachTotalAmtDf = superSellOrder.groupby(['Type', 'SaleOrderID'])['Amount'].sum().reset_index()
        superSellOrderTotalAmt = superSellOrderEachTotalAmtDf['Amount'].sum()
        superSellOrderSellAmt = superSellOrderEachTotalAmtDf[superSellOrderEachTotalAmtDf['Type'] == 'S'][
            'Amount'].sum()
        superSellOrderBuyAmt = superSellOrderEachTotalAmtDf[superSellOrderEachTotalAmtDf['Type'] == 'B'][
            'Amount'].sum()
        # 汇总结果
        res['superSellOrderTotalNum'] = superSellOrderTotalNum
        res['superSellOrderTotalTransNum'] = superSellOrderTotalTransNum
        res['superSellOrderSellTransNum'] = superSellOrderSellTransNum
        res['superSellOrderBuyTransNum'] = superSellOrderBuyTransNum
        res['superSellOrderTotalAmt'] = superSellOrderTotalAmt
        res['superSellOrderSellAmt'] = superSellOrderSellAmt
        res['superSellOrderBuyAmt'] = superSellOrderBuyAmt
        # 超大卖单成交价差指标
        superSellOrderEachPriceSpreadSe = superSellOrder.groupby('SaleOrderID')['Price'].apply(
            lambda x: x.iloc[0] - x.iloc[-1])
        superSellOrderNonZeroPriceSpreadDf = superSellOrderEachPriceSpreadSe[
            superSellOrderEachPriceSpreadSe != 0].reset_index().rename(columns={'Price': 'PriceSpread'})
        if superSellOrderNonZeroPriceSpreadDf.empty:
            pass
        else:
            superSellOrderPriceSpreadOrderDf = pd.merge(transData, superSellOrderNonZeroPriceSpreadDf,
                                                        on=['SaleOrderID'], how='inner')
            superSellOrderPriceSpreadOrderTotalAmt = superSellOrderPriceSpreadOrderDf['Amount'].sum()
            superSellOrderPriceSpreadOrderDiffAmt = (
                    superSellOrderPriceSpreadOrderDf['Volume'] * superSellOrderPriceSpreadOrderDf[
                'PriceSpread']).sum()
            superSellOrderPriceSpreadOrderVol = superSellOrderPriceSpreadOrderDf['Volume'].sum()
            superSellOrderPriceSpreadOrderPrice = superSellOrderPriceSpreadOrderDiffAmt / superSellOrderPriceSpreadOrderVol
            # 汇总结果
            res['superSellOrderPriceSpreadOrderTotalAmt'] = superSellOrderPriceSpreadOrderTotalAmt
            res['superSellOrderPriceSpreadOrderDiffAmt'] = superSellOrderPriceSpreadOrderDiffAmt
            res['superSellOrderPriceSpreadOrderTotalVol'] = superSellOrderPriceSpreadOrderVol
            res['superSellOrderPriceSpreadOrderMeanPrice'] = superSellOrderPriceSpreadOrderPrice

    # 大卖单统计指标
    largeSellOrderID = sellOrderAmt[sellOrderAmt >= largeSellOrderQuantile].reset_index()[['SaleOrderID']]
    if largeSellOrderID.empty:
        pass
    else:
        largeSellOrder = pd.merge(transData, largeSellOrderID, on=['SaleOrderID'], how='inner')
        largeSellOrderTotalNum = largeSellOrderID.shape[0]
        # 大卖单交易笔数指标
        largeSellOrderEachTotalTransNumDf = largeSellOrder.groupby(['Type', 'SaleOrderID'])[
            'Volume'].count().reset_index().rename(columns={'Volume': 'count'})
        largeSellOrderTotalTransNum = largeSellOrderEachTotalTransNumDf['count'].sum()
        largeSellOrderSellTransNum = \
            largeSellOrderEachTotalTransNumDf[largeSellOrderEachTotalTransNumDf['Type'] == 'S']['count'].sum()
        largeSellOrderBuyTransNum = \
            largeSellOrderEachTotalTransNumDf[largeSellOrderEachTotalTransNumDf['Type'] == 'B']['count'].sum()
        # 大卖单成交额指标
        largeSellOrderEachTotalAmtDf = largeSellOrder.groupby(['Type', 'SaleOrderID'])['Amount'].sum().reset_index()
        largeSellOrderTotalAmt = largeSellOrderEachTotalAmtDf['Amount'].sum()
        largeSellOrderSellAmt = largeSellOrderEachTotalAmtDf[largeSellOrderEachTotalAmtDf['Type'] == 'S'][
            'Amount'].sum()
        largeSellOrderBuyAmt = largeSellOrderEachTotalAmtDf[largeSellOrderEachTotalAmtDf['Type'] == 'B'][
            'Amount'].sum()
        # 汇总结果
        res['largeSellOrderTotalNum'] = largeSellOrderTotalNum
        res['largeSellOrderTotalTransNum'] = largeSellOrderTotalTransNum
        res['largeSellOrderSellTransNum'] = largeSellOrderSellTransNum
        res['largeSellOrderBuyTransNum'] = largeSellOrderBuyTransNum
        res['largeSellOrderTotalAmt'] = largeSellOrderTotalAmt
        res['largeSellOrderSellAmt'] = largeSellOrderSellAmt
        res['largeSellOrderBuyAmt'] = largeSellOrderBuyAmt
        # 大卖单成交价差指标
        largeSellOrderEachPriceSpreadSe = largeSellOrder.groupby('SaleOrderID')['Price'].apply(
            lambda x: x.iloc[0] - x.iloc[-1])
        largeSellOrderNonZeroPriceSpreadDf = largeSellOrderEachPriceSpreadSe[
            largeSellOrderEachPriceSpreadSe != 0].reset_index().rename(columns={'Price': 'PriceSpread'})
        if largeSellOrderNonZeroPriceSpreadDf.empty:
            pass
        else:
            largeSellOrderPriceSpreadOrderDf = pd.merge(transData, largeSellOrderNonZeroPriceSpreadDf,
                                                        on=['SaleOrderID'], how='inner')
            largeSellOrderPriceSpreadOrderTotalAmt = largeSellOrderPriceSpreadOrderDf['Amount'].sum()
            largeSellOrderPriceSpreadOrderDiffAmt = (
                    largeSellOrderPriceSpreadOrderDf['Volume'] * largeSellOrderPriceSpreadOrderDf[
                'PriceSpread']).sum()
            largeSellOrderPriceSpreadOrderVol = largeSellOrderPriceSpreadOrderDf['Volume'].sum()
            largeSellOrderPriceSpreadOrderPrice = largeSellOrderPriceSpreadOrderDiffAmt / largeSellOrderPriceSpreadOrderVol
            # 汇总结果
            res['largeSellOrderPriceSpreadOrderTotalAmt'] = largeSellOrderPriceSpreadOrderTotalAmt
            res['largeSellOrderPriceSpreadOrderDiffAmt'] = largeSellOrderPriceSpreadOrderDiffAmt
            res['largeSellOrderPriceSpreadOrderTotalVol'] = largeSellOrderPriceSpreadOrderVol
            res['largeSellOrderPriceSpreadOrderMeanPrice'] = largeSellOrderPriceSpreadOrderPrice

    # 超大买单统计指标
    superBuyOrderID = buyOrderAmt[buyOrderAmt >= superBuyOrderQuantile].reset_index()[['BuyOrderID']]
    if superBuyOrderID.empty:
        pass
    else:
        superBuyOrder = pd.merge(transData, superBuyOrderID, on=['BuyOrderID'], how='inner')
        superBuyOrderTotalNum = superBuyOrderID.shape[0]
        # 超大买单交易笔数指标
        superBuyOrderEachTransNumDf = superBuyOrder.groupby(['Type', 'BuyOrderID'])[
            'Volume'].count().reset_index().rename(columns={'Volume': 'count'})
        superBuyOrderTotalTransNum = superBuyOrderEachTransNumDf['count'].sum()
        superBuyOrderSellTransNum = superBuyOrderEachTransNumDf[superBuyOrderEachTransNumDf['Type'] == 'S'][
            'count'].sum()
        superBuyOrderBuyTransNum = superBuyOrderEachTransNumDf[superBuyOrderEachTransNumDf['Type'] == 'B'][
            'count'].sum()
        # 超大买单成交额指标
        superBuyOrderEachTotalAmtDf = superBuyOrder.groupby(['Type', 'BuyOrderID'])['Amount'].sum().reset_index()
        superBuyOrderTotalAmt = superBuyOrderEachTotalAmtDf['Amount'].sum()
        superBuyOrderSellAmt = superBuyOrderEachTotalAmtDf[superBuyOrderEachTotalAmtDf['Type'] == 'S'][
            'Amount'].sum()
        superBuyOrderBuyAmt = superBuyOrderEachTotalAmtDf[superBuyOrderEachTotalAmtDf['Type'] == 'B'][
            'Amount'].sum()
        # 汇总结果
        res['superBuyOrderTotalNum'] = superBuyOrderTotalNum
        res['superBuyOrderTotalTransNum'] = superBuyOrderTotalTransNum
        res['superBuyOrderSellTransNum'] = superBuyOrderSellTransNum
        res['superBuyOrderBuyTransNum'] = superBuyOrderBuyTransNum
        res['superBuyOrderTotalAmt'] = superBuyOrderTotalAmt
        res['superBuyOrderSellAmt'] = superBuyOrderSellAmt
        res['superBuyOrderBuyAmt'] = superBuyOrderBuyAmt
        # 超大买单成交价差指标
        superBuyOrderEachPriceSpreadSe = superBuyOrder.groupby('BuyOrderID')['Price'].apply(
            lambda x: x.iloc[0] - x.iloc[-1])
        superBuyOrderNonZeroPriceSpreadDf = superBuyOrderEachPriceSpreadSe[
            superBuyOrderEachPriceSpreadSe != 0].reset_index().rename(columns={'Price': 'PriceSpread'})
        if superBuyOrderNonZeroPriceSpreadDf.empty:
            pass
        else:
            superBuyOrderPriceSpreadOrderDf = pd.merge(transData, superBuyOrderNonZeroPriceSpreadDf,
                                                       on=['BuyOrderID'], how='inner')
            superBuyOrderPriceSpreadOrderTotalAmt = superBuyOrderPriceSpreadOrderDf['Amount'].sum()
            superBuyOrderPriceSpreadOrderDiffAmt = (
                    superBuyOrderPriceSpreadOrderDf['Volume'] * superBuyOrderPriceSpreadOrderDf[
                'PriceSpread']).sum()
            superBuyOrderPriceSpreadOrderVol = superBuyOrderPriceSpreadOrderDf['Volume'].sum()
            superBuyOrderPriceSpreadOrderPrice = superBuyOrderPriceSpreadOrderDiffAmt / superBuyOrderPriceSpreadOrderVol
            # 汇总结果
            res['superBuyOrderPriceSpreadOrderTotalAmt'] = superBuyOrderPriceSpreadOrderTotalAmt
            res['superBuyOrderPriceSpreadOrderDiffAmt'] = superBuyOrderPriceSpreadOrderDiffAmt
            res['superBuyOrderPriceSpreadOrderTotalVol'] = superBuyOrderPriceSpreadOrderVol
            res['superBuyOrderPriceSpreadOrderMeanPrice'] = superBuyOrderPriceSpreadOrderPrice

    # 大买单统计指标
    largeBuyOrderID = buyOrderAmt[buyOrderAmt >= largeBuyOrderQuantile].reset_index()[['BuyOrderID']]
    if largeBuyOrderID.empty:
        pass
    else:
        largeBuyOrder = pd.merge(transData, largeBuyOrderID, on=['BuyOrderID'], how='inner')
        largeBuyOrderTotalNum = largeBuyOrderID.shape[0]
        # 大买单交易笔数指标
        largeBuyOrderEachTotalTransNumDf = largeBuyOrder.groupby(['Type', 'BuyOrderID'])[
            'Volume'].count().reset_index().rename(columns={'Volume': 'count'})
        largeBuyOrderTotalTransNum = largeBuyOrderEachTotalTransNumDf['count'].sum()
        largeBuyOrderSellTransNum = \
            largeBuyOrderEachTotalTransNumDf[largeBuyOrderEachTotalTransNumDf['Type'] == 'S']['count'].sum()
        largeBuyOrderBuyTransNum = \
            largeBuyOrderEachTotalTransNumDf[largeBuyOrderEachTotalTransNumDf['Type'] == 'B']['count'].sum()
        # 大买单成交额指标
        largeBuyOrderEachTotalAmtDf = largeBuyOrder.groupby(['Type', 'BuyOrderID'])['Amount'].sum().reset_index()
        largeBuyOrderTotalAmt = largeBuyOrderEachTotalAmtDf['Amount'].sum()
        largeBuyOrderSellAmt = largeBuyOrderEachTotalAmtDf[largeBuyOrderEachTotalAmtDf['Type'] == 'S'][
            'Amount'].sum()
        largeBuyOrderBuyAmt = largeBuyOrderEachTotalAmtDf[largeBuyOrderEachTotalAmtDf['Type'] == 'B'][
            'Amount'].sum()
        # 汇总结果
        res['largeBuyOrderTotalNum'] = largeBuyOrderTotalNum
        res['largeBuyOrderTotalTransNum'] = largeBuyOrderTotalTransNum
        res['largeBuyOrderSellTransNum'] = largeBuyOrderSellTransNum
        res['largeBuyOrderBuyTransNum'] = largeBuyOrderBuyTransNum
        res['largeBuyOrderTotalAmt'] = largeBuyOrderTotalAmt
        res['largeBuyOrderSellAmt'] = largeBuyOrderSellAmt
        res['largeBuyOrderBuyAmt'] = largeBuyOrderBuyAmt
        # 大买单成交价差指标
        largeBuyOrderEachPriceSpreadSe = largeBuyOrder.groupby('BuyOrderID')['Price'].apply(
            lambda x: x.iloc[0] - x.iloc[-1])
        largeBuyOrderNonZeroPriceSpreadDf = largeBuyOrderEachPriceSpreadSe[
            largeBuyOrderEachPriceSpreadSe != 0].reset_index().rename(columns={'Price': 'PriceSpread'})
        if largeBuyOrderNonZeroPriceSpreadDf.empty:
            pass
        else:
            largeBuyOrderPriceSpreadOrderDf = pd.merge(transData, largeBuyOrderNonZeroPriceSpreadDf,
                                                       on=['BuyOrderID'], how='inner')
            largeBuyOrderPriceSpreadOrderTotalAmt = largeBuyOrderPriceSpreadOrderDf['Amount'].sum()
            largeBuyOrderPriceSpreadOrderDiffAmt = (
                    largeBuyOrderPriceSpreadOrderDf['Volume'] * largeBuyOrderPriceSpreadOrderDf['PriceSpread']).sum()
            largeBuyOrderPriceSpreadOrderVol = largeBuyOrderPriceSpreadOrderDf['Volume'].sum()
            largeBuyOrderPriceSpreadOrderPrice = largeBuyOrderPriceSpreadOrderDiffAmt / largeBuyOrderPriceSpreadOrderVol
            # 汇总结果
            res['largeBuyOrderPriceSpreadOrderTotalAmt'] = largeBuyOrderPriceSpreadOrderTotalAmt
            res['largeBuyOrderPriceSpreadOrderDiffAmt'] = largeBuyOrderPriceSpreadOrderDiffAmt
            res['largeBuyOrderPriceSpreadOrderTotalVol'] = largeBuyOrderPriceSpreadOrderVol
            res['largeBuyOrderPriceSpreadOrderMeanPrice'] = largeBuyOrderPriceSpreadOrderPrice
    res = pd.Series(res)
    return res
