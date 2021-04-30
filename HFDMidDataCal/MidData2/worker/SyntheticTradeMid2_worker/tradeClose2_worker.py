# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import time_std


def tradeClose2_worker(data: pd.DataFrame,
                       date: str,
                       **kwargs) -> pd.DataFrame:
    codeID = data['code'].drop_duplicates()
    dataSub = data.set_index(['time', 'code'])

    info925 = dataSub.loc['09:25:00']
    info935 = dataSub.loc['09:35:00']

    # 09:25:00和09:35:00收盘价
    if not info925.empty:
        close925 = pd.Series(info925['close'], name='close925')
    else:
        close925 = pd.Series(index=codeID, name='close925')
    if not info935.empty:
        close935 = pd.Series(info935['close'], name='close925')
    else:
        close935 = pd.Series(index=codeID, name='close935')

    firsts = dataSub.groupby('code')['close'].transform('first')
    dataSub['retCum'] = dataSub['close'] / firsts - 1
    dataSub['closeTick'] = dataSub['close'] - firsts
    dataSub['sign'] = np.sign(dataSub['close'] - dataSub['open'])
    retCum = dataSub[dataSub['retCum'] >= 0.09]
    closeTick = dataSub[dataSub['closeTick'] >= 4.5]

    # 收益率大于0.09或收盘价超过4.5后的第一根一分钟阴线的收盘价和时间
    if retCum.empty:
        retCumRes = pd.DataFrame(columns=['ret9DownTime', 'ret9DownClose'], index=codeID)
    else:
        retCumRes = retCum[retCum['sign'] < 0].reset_index('time').groupby('code').first()[['time', 'close']]
        retCumRes = retCumRes.rename(columns={'time': 'ret9DownTime', 'close': 'ret9DownClose'})

    if closeTick.empty:
        closeTickRes = pd.DataFrame(columns=['close45DownTime', 'close45DownClose'], index=codeID)
    else:
        closeTickRes = closeTick[closeTick['sign'] < 0].reset_index('time').groupby('code').first()[['time', 'close']]
        closeTickRes = closeTickRes.rename(columns={'time': 'close45DownTime', 'close': 'close45DownClose'})
    res = pd.concat([close925, close935, retCumRes, closeTickRes], axis=1).reset_index()
    res['date'] = date
    return res
