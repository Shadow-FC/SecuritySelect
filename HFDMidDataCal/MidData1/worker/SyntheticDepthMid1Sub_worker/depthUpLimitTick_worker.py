# -*-coding:utf-8-*-
# @Time:   2021/4/29 15:18
# @Author: FC
# @Email:  18817289038@163.com

import time
import numpy as np
import pandas as pd
from typing import Dict, Any


def depthUpLimitTick_worker(data: pd.DataFrame,
                            code: str,
                            date: str,
                            **kwargs) -> pd.Series(float):

    data = data.sort_values(['时间'])
    data[['date', 'Time']] = data['时间'].str.split(' ').tolist()

    dataSub = data[data['Time'] >= '09:25:00'].copy()
    HighPrice = max(data['最新价'])  # 涨停价

    # 封板
    HighPriceData = dataSub[dataSub['挂买价1'] == HighPrice].copy()
    UpEndInfo = HighPriceData.iloc[0]

    # 判断封板发生的时间段
    if '09:00:00' <= UpEndInfo['Time'] < '09:30:00':
        UpLimitFlag = 'CallAM'
    elif '09:30:00' <= UpEndInfo['Time'] < '11:31:00':
        UpLimitFlag = 'TradeAM'
    elif '13:00:00' <= UpEndInfo['Time'] < '14:57:00':
        UpLimitFlag = 'TradePM'
    else:
        UpLimitFlag = 'CallPM'

    # 集合竞价阶段封板不考虑
    if UpLimitFlag not in ['TradeAM', 'TradePM']:
        return

    oldInfo = dataSub[dataSub['Time'] <= UpEndInfo['Time']]
    if oldInfo.shape[0] >= 2:
        res = oldInfo.iloc[-2]
        res['code'] = code
        return res
