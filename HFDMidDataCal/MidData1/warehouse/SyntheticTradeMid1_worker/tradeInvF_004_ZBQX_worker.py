# -*-coding:utf-8-*-
# @Time:   2021/4/14 14:29
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd


def tradeInvF_004_ZBQX_worker(data,
                              code: str,
                              date: str,
                              **kwargs) -> pd.Series(float):
    '''
    将每天个股的逐笔交易按从最低价到最高价，等距换分为五个区间，分析各价格区间内股票的交易特征，
    尤其是成交额占比最大的价格区间（若该价格区间内，成交额占比大至一定程度，可理解为该区间内的
    成交量加权均价为力量最为强大的控盘价位（参与交易人员的心里价位）），用于衍生计算如下因子：
        1.成交额占比最大区间的成交额相对全天成交额占比
        2.成交额占比最大区间的成交量相对全天成交量占比
        3.成交额占比最大区间的成交笔数相对全天成交笔数占比
        4.成交额占比最大区间的成交量加权均价相对于与全天均价涨跌幅；
        5.成交额占比最大区间的成交量加权均价相对于与收盘价的涨跌幅；
        6.成交额占比最大区间的成交量加权均价相对于与收盘价的涨跌幅；
        7.成交额占比最大区间的成交量加权均价相对于与最低价的涨跌幅；
        8.成交额占比最大区间的成交量加权均价相对于与最高价的涨跌幅；
        9.各区间成交量加权均价标准差
       10.各区间成交额标准差；
    该算法需要在逐笔成交数据统计如下特征：
        1.各价格区间内，成交量、成交额、成交笔数、Vwap、成交价格标准差
        2.全天量、额、笔数、Vwap、成交价格标准差、收盘价
    '''
    res = {'code': code, 'date': date}
    layerNum = 5
    #  数据准备
    data['Amount'] = data['Volume'] * data['Price']
    transData = data[data['Time'] >= '09:30:00'].copy()
    maxPriceDiff = transData['Price'].max() - transData['Price'].min()
    # 全天交易价差必须大于0.05元
    if maxPriceDiff <= 0.05:
        pass
    else:
        transData['layerLable'] = transData[['Price']].apply(
            lambda x: pd.cut(x, bins=layerNum, labels=list(range(1, layerNum + 1, 1))))
        # 统计全天成交情况
        totalVol = transData['Volume'].sum()
        totalAmt = transData['Amount'].sum()
        totalNum = transData['Amount'].count()
        totalVwap = totalAmt / totalVol
        totalStd = (transData['Price'] / totalVwap).std()
        # 统计各价格分组特征
        layerVolInd = list(map(lambda x: 'layerVol_%s' % x, range(1, layerNum + 1, 1)))
        layerAmtInd = list(map(lambda x: 'layerAmt_%s' % x, range(1, layerNum + 1, 1)))
        layerVol = transData.groupby('layerLable')[['Volume']].sum().T.rename(
            columns={1: 'layerVol_1', 2: 'layerVol_2', 3: 'layerVol_3', 4: 'layerVol_4',
                     5: 'layerVol_5'}).T.reindex(index=layerVolInd)
        layerAmt = transData.groupby('layerLable')[['Amount']].sum().T.rename(
            columns={1: 'layerAmt_1', 2: 'layerAmt_2', 3: 'layerAmt_3', 4: 'layerAmt_4',
                     5: 'layerAmt_5'}).T.reindex(index=layerAmtInd)
        layerNum = transData.groupby('layerLable')[['Amount']].count().T.rename(
            columns={1: 'layerNum_1', 2: 'layerNum_2', 3: 'layerNum_3', 4: 'layerNum_4', 5: 'layerNum_5'}).T
        layerVwap = pd.Series(np.array(layerAmt['Amount']) / np.array(layerVol['Volume']),
                              index=['layerVwap_1', 'layerVwap_2', 'layerVwap_3', 'layerVwap_4', 'layerVwap_5'])
        # layerStd = transData.groupby('layerLable')[['Price']].std().rename(columns = {'Price':'layerPriceStd'})
        layerStd = transData.groupby('layerLable')[['Price']].apply(
            lambda x: (x[['Price']] / totalVwap - 1).std()).T.rename(
            columns={1: 'layerStd_1', 2: 'layerStd_2', 3: 'layerStd_3', 4: 'layerStd_4', 5: 'layerStd_5'}).T
        # 保存计算结果
        res['totalVol'] = totalVol
        res['totalAmt'] = totalAmt
        res['totalNum'] = totalNum
        res['totalVwap'] = totalVwap
        res['totalStd'] = totalStd

        if layerVol.empty:
            pass
        else:
            res.update(layerVol['Volume'].to_dict())

        if layerAmt.empty:
            pass
        else:
            res.update(layerAmt['Amount'].to_dict())

        if layerNum.empty:
            pass
        else:
            res.update(layerNum['Amount'].to_dict())

        if layerVwap.empty:
            pass
        else:
            res.update(layerVwap.to_dict())

        if layerStd.empty:
            pass
        else:
            res.update(layerStd['Price'].to_dict())
    res = pd.Series(res)
    return res
