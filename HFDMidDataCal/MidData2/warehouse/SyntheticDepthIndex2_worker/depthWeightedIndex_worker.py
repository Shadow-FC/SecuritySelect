# -*-coding:utf-8-*-
# @Time:   2021/4/15 13:35
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd

priceColumns = ['open', 'high', 'low', 'close']
depthColumns = ['amount', 'tradenum', 'volume', 'bidvolume1', 'askvolume1',
                'askvolume5sum', 'bidvolume5sum', 'askvolume10sum', 'bidvolume10sum']


def depthWeightedIndex_worker(data: pd.DataFrame,
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
    dataNew[depthColumns] = dataNew[depthColumns].fillna(0)

    # 新列生成
    dataNew['bidamount1'] = dataNew['bidvolume1'] * dataNew['bidprice1']
    dataNew['askamount1'] = dataNew['askvolume1'] - dataNew['askprice1']
    dataNew['bidamount5sum'] = dataNew['bidvolume5sum'] * (dataNew['bidprice1'] - 0.025)
    dataNew['askamount5sum'] = dataNew['askvolume5sum'] * (dataNew['askprice1'] + 0.025)
    dataNew['bidamount10sum'] = dataNew['bidvolume10sum'] * (dataNew['bidprice1'] - 0.05)
    dataNew['askamount10sum'] = dataNew['askvolume10sum'] * (dataNew['askprice1'] + 0.05)

    # 新价格生成
    firsts = dataNew.groupby('code')[priceColumns].transform('first')
    dataNew[priceColumns] = dataNew[priceColumns] / firsts

    # 加权合成指数
    res = dataNew[dataNew.columns.difference(['weight', 'bidprice1', 'askprice1'])].mul(dataNew['weight'], axis=0)
    res = res.groupby('time').sum()
    return res
