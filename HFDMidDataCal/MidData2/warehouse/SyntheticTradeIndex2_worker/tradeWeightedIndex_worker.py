# -*-coding:utf-8-*-
# @Time:   2021/4/15 13:35
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd

priceColumns = ['open', 'high', 'low', 'close']
tradeColumns = ['amount', 'buyamount', 'buytradenum', 'buyvolume', 'tradenum', 'volume']


def tradeWeightedIndex_worker(data: pd.DataFrame,
                              weight: pd.DataFrame,
                              **kwargs) -> pd.DataFrame:
    dataSub = pd.merge(data, weight[['code', 'weight']], on='code', how='inner')
    dataSub = dataSub.set_index(['time', 'code'])
    # 填充收盘价
    dataSub[priceColumns] = dataSub[priceColumns].replace({0: np.nan})
    dataClose = dataSub['close'].unstack().ffill().bfill().stack()
    dataClose.name = 'close'
    dataNew = pd.merge(dataSub[dataSub.columns.difference(['close'])],
                       dataClose,
                       left_index=True,
                       right_index=True,
                       how='right')
    dataNew[priceColumns] = dataNew[priceColumns].bfill(axis=1)

    # 填充权重
    dataNew['weight'] = dataNew.groupby('code')['weight'].ffill().bfill()

    # 其余填充0
    dataNew[tradeColumns] = dataNew[tradeColumns].fillna(0)

    # 新列生成
    dataNew['sellamount'] = dataNew['amount'] - dataNew['buyamount']
    dataNew['sellvolume'] = dataNew['volume'] - dataNew['buyvolume']
    dataNew['selltradenum'] = dataNew['tradenum'] - dataNew['buytradenum']

    # 新价格生成
    firsts = dataNew.groupby('code')[priceColumns].transform('first')
    dataNew[priceColumns] = dataNew[priceColumns] / firsts

    # 加权合成指数
    res = dataNew[dataNew.columns.difference(['weight'])].mul(dataNew['weight'], axis=0)
    res = res.groupby('time').sum()
    return res
