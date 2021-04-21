# -*-coding:utf-8-*-
# @Time:   2021/4/15 13:35
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd

priceColumns = ['open', 'high', 'low', 'close']
tradeColumns = ['amount', 'buyamount', 'buytradenum', 'buyvolume', 'tradenum', 'volume']


def tradeEqualIndex_worker(data: pd.DataFrame,
                           weight: pd.DataFrame,
                           **kwargs) -> pd.DataFrame:
    dataSub = data[data['code'].isin(weight['code'])]
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
    res = dataNew.groupby('time').mean()
    return res
