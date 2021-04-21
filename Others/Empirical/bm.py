# -*-coding:utf-8-*-
# @Time:   2021/2/24 17:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict


if __name__ == '__main__':
    BM = pd.read_csv(r'B:\Database\StockPool.csv')
    liq_MV = pd.read_csv(r'B:\Database\AStockData.csv', usecols=['date', 'code', 'liqMv'],
                         index_col=['date', 'code'])
    bm = BM[BM['stockPool']][['date', 'code', 'stockWeight']]
    bm = bm.set_index(['date', 'code'])
    bm['ratio'] = bm.groupby('date', group_keys=False).apply(lambda x: x['stockWeight'] / sum(x['stockWeight']))
    res = pd.merge(bm, liq_MV, left_index=True, right_index=True, how='left')
    res['group'] = res.groupby('date', group_keys=False).apply(lambda x: pd.cut(x['liqMv'].rank(), bins=10, labels=False) + 1)

    res_weight = res.groupby(['date', 'group'])['ratio'].sum().unstack()
    res_mv = res.groupby(['date', 'group'])['liqMv'].sum().unstack()

    res_weight_ratio = res_weight.apply(lambda x: x / sum(x), axis=1)
    res_mv_ratio = res_mv.apply(lambda x: x / sum(x), axis=1)

    res_mv_ratio.plot.area()
    plt.show()
