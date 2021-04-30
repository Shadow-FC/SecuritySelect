# -*-coding:utf-8-*-
# @Time:   2021/4/29 15:18
# @Author: FC
# @Email:  18817289038@163.com

import time
import numpy as np
import pandas as pd

res = {
    "date": np.nan,  # 日期
    "code": np.nan,  # 股票代码

    "UpLimitFlag": np.nan,  # 封板区间段
    "UpLimitTime": np.nan,  # 封板时时间

    "Last2Price": np.nan,  # 封板前一个Tick的成交价
    "Last2PriceSpread": np.nan,  # 封板前一个Tick与最高价差值
    "Last2PriceRatio": np.nan,  # 封板前一个Tick与最高价所差百分比

    "BreakLimitTime": 0,  # 开板标记：没有开板为0，开板则为开板时间
    "LimitToBreakTime": np.nan  # 封板到开板耗时

}


def depthUpLimitInfo_worker(data: pd.DataFrame,
                            code: str,
                            date: str,
                            **kwargs) -> pd.Series(float):
    data = data.sort_values(['时间'])
    res.update({"date": date, "code": code})
    data['Time'] = data['时间'].map(lambda x: x.split(" ")[-1])

    dataSub = data[data['Time'] >= '09:25:00'].copy()
    HighPrice = max(data['最新价'])  # 涨停价

    # 封板
    HighPriceData = dataSub[dataSub['挂买价1'] == HighPrice].copy()
    UpEndInfo = HighPriceData.iloc[0]
    res['UpLimitTime'] = UpEndInfo['Time']

    # 判断封板发生的时间段
    if '09:00:00' <= UpEndInfo['Time'] < '09:30:00':
        res['UpLimitFlag'] = 'CallAM'
    elif '09:30:00' <= UpEndInfo['Time'] < '11:31:00':
        res['UpLimitFlag'] = 'TradeAM'
    elif '13:00:00' <= UpEndInfo['Time'] < '14:57:00':
        res['UpLimitFlag'] = 'TradePM'
    else:
        res['UpLimitFlag'] = 'CallPM'

    # 集合竞价阶段封板不考虑： 必须返回，不然尾盘集合竞价会干扰
    if res['UpLimitFlag'] not in ['TradeAM', 'TradePM']:
        return pd.Series(res)

    # 封板前一个Tick
    oldInfo = dataSub[dataSub['Time'] <= res['UpLimitTime']]
    if oldInfo.shape[0] >= 2:
        res["Last2Price"] = oldInfo.iloc[-2]['最新价']
        res["Last2PriceSpread"] = HighPrice - res["Last2Price"]
        res["Last2PriceRatio"] = res["Last2PriceSpread"] / HighPrice

    # 开板
    newInfo = dataSub[dataSub['Time'] >= res['UpLimitTime']]
    if min(newInfo['挂买价1']) < HighPrice:
        breakLimitT = newInfo[newInfo['挂买价1'] < HighPrice].iloc[0]['Time']
    else:
        breakLimitT = None

    if breakLimitT is not None:
        timeRage = np.dot(np.array(breakLimitT.split(':'), dtype=int), [3600, 60, 1]) - \
                   np.dot(np.array(res['UpLimitTime'].split(':'), dtype=int), [3600, 60, 1])
        res['BreakLimitTime'] = breakLimitT
        res['LimitToBreakTime'] = timeRage

    return pd.Series(res)
