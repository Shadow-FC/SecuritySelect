# -*-coding:utf-8-*-
# @Time:   2021/4/14 16:58
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd


"""
该文件下函数为计算子函数，例如计算因子所需要的中间过程计算函数，即该函数需要结合特定情形使用
"""


# 箱体上下边缘过滤
def boxEdge(data: pd.Series(float),
            edgeWay: str = 'up') -> pd.Series:
    if data.empty:
        return data
    data_Down, data_Up = data.quantile(0.25), data.quantile(0.75)

    if edgeWay == 'up':
        data_EdgeUp = data_Up + 1.5 * (data_Up - data_Down)
        res = data[data > data_EdgeUp]

    elif edgeWay == 'down':
        data_EdgeDown = data_Down - 1.5 * (data_Up - data_Down)
        res = data[data < data_EdgeDown]
    else:
        print(f'Param edgeWay error: {edgeWay}')
        res = data

    return res


def entropy(x: pd.Series(float), bottom: int = 2):
    """
    离散熵
    空值不剔除
    :param x:
    :param bottom:
    :return:
    """
    Probability = (x.groupby(x).count()).div(len(x))
    log2 = np.log(Probability) / np.log(bottom)
    result = - (Probability * log2).sum()
    return result


def func_M_sqrt(data: pd.DataFrame):
    # 可能存在分钟线丢失
    data['S'] = abs(data['close'].pct_change()) / np.sqrt(data['volume'])
    VWAP = (data['close'] * data['volume'] / (data['volume']).sum()).sum()
    data = data.sort_values('S', ascending=False)
    data['cum_volume_R'] = data['volume'].cumsum() / (data['volume']).sum()
    data_ = data[data['cum_volume_R'] <= 0.2]
    res = (data_['close'] * data_['volume'] / (data_['volume']).sum()).sum() / VWAP

    return res


def func_M_ln(data: pd.DataFrame):
    data['S'] = abs(data['close'].pct_change()) / np.log(data['volume'])
    VWAP = (data['close'] * data['volume'] / (data['volume']).sum()).sum()
    data = data.sort_values('S', ascending=False)
    data['cum_volume_R'] = data['volume'].cumsum() / (data['volume']).sum()
    data_ = data[data['cum_volume_R'] <= 0.2]
    res = (data_['close'] * data_['volume'] / (data_['volume']).sum()).sum() / VWAP
    return res


def func_Structured_reversal(data: pd.DataFrame,
                             ratio: float):
    data = data.sort_values('volume', ascending=True)
    data['cum_volume'] = data['volume'].cumsum() / data['volume'].sum()
    # momentum
    data_mom = data[data['cum_volume'] <= ratio]
    rev_mom = (data_mom['ret'] * (1 / data_mom['volume'])).sum() / (1 / data_mom['volume']).sum()
    # Reverse
    data_rev = data[data['cum_volume'] > ratio]
    rev_rev = (data_rev['ret'] * (data_rev['volume'])).sum() / (data_rev['volume']).sum()

    rev_struct = rev_rev - rev_mom
    return rev_struct
