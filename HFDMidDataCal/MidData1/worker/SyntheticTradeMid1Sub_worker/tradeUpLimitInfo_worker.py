# -*-coding:utf-8-*-
# @Time:   2021/4/28 14:14
# @Author: FC
# @Email:  18817289038@163.com

import time
import numpy as np
import pandas as pd
from typing import Dict, Any
"""
1.全部为主买单没有主卖单

1)封板后不存在交易，没有开板；
2)封板后买一单(9.5)全部撤单，有人按照买一价(9.5)挂卖一，但没有主动卖出，市场依然主动买入，整个过程只存在主买，理论上算是开板，但从逐笔无法判断，那么该情况下最后一笔才为封板交易（我认为算是存在开板）；
3)封板后买一单(9.5)和买二单(9.4)全部撤单，有人按照买二价(9.4)挂卖一，但依然没有主动卖出，只有主动买入，整个过程只存在主买，此时按照第一笔9.4价格交易时间判定为开板时间点

2.存在主卖单的封板
1)封板后买一单(9.5)被主卖单吃完且在卖一(9.5)挂单没有继续吃单，后续没有交易，判定为没有开板，因为从逐笔交易无法判断什么时候买一单被主动吃完；
2)封板后买一单(9.5)被主卖单吃完且在卖一(9.5)挂单没有继续吃单，后续存在主买吃单，判定主卖单后的第一笔主买单(9.5)为开板交易；
3)在上述条件都不满足的情况下，价格发生变化的第一笔判定为开板单

"""
res = {
    "date": np.nan,  # 日期
    "code": np.nan,  # 股票代码

    "UpLimitFlag": np.nan,  # 封板区间段
    "UpLimitTime": np.nan,  # 封板时时间
    "UpLimitJudge": np.nan,  # 封板判断依据

    "BreakLimitTime": 0,  # 开板标记：没有开板为0，开板则为开板时间
    "LimitToBreakTime": np.nan,  # 封板到开板耗时

    "Last2OrderPriceRange": np.nan,  # 促使封板的倒数第二个订单的价格与最高价差值
    "Last2OrderPriceRatio": np.nan,  # 促使封板的倒数第二个订单的价格与最高价所差百分比

    "Last1SecToHighRange": np.nan,  # 封板前1秒成交价与最高价就差值
    "Last1SecToHighRatio": np.nan,  # 封板前1秒成交价与最高价所差百分比
    "Last1SecToHighOrderNum": np.nan,  # 封板前1秒至封板期间主买单数

    "Last3SecToHighRange": np.nan,  # 封板前3秒成交价与最高价就差值
    "Last3SecToHighRatio": np.nan,  # 封板前3秒成交价与最高价所差百分比
    "Last3SecToHighOrderNum": np.nan,  # 封板前3秒至封板期间主买单数

    "Last5SecToHighRange": np.nan,  # 封板前5秒成交价与最高价就差值
    "Last5SecToHighRatio": np.nan,  # 封板前5秒成交价与最高价所差百分比
    "Last5SecToHighOrderNum": np.nan,  # 封板前5秒至封板期间主买单数

    "Last1PriceToHighTime": np.nan,  # 封板前1跳价格至封板耗时
    "Last2PriceToHighTime": np.nan,  # 封板前2跳价格至封板耗时
    "Last3PriceToHighTime": np.nan,  # 封板前3跳价格至封板耗时
    "Last4PriceToHighTime": np.nan,  # 封板前4跳价格至封板耗时
    "Last5PriceToHighTime": np.nan,  # 封板前5跳价格至封板耗时
    "Last6PriceToHighTime": np.nan,  # 封板前6跳价格至封板耗时
    "Last7PriceToHighTime": np.nan,  # 封板前7跳价格至封板耗时
    "Last8PriceToHighTime": np.nan,  # 封板前8跳价格至封板耗时
    "Last9PriceToHighTime": np.nan,  # 封板前9跳价格至封板耗时
    "Last10PriceToHighTime": np.nan,  # 封板前10跳价格至封板耗时
}


def tradeUpLimitInfo_worker(data: pd.DataFrame,
                            code: str,
                            date: str,
                            **kwargs) -> Dict[str, Any]:
    res.update({'date': date, 'code': code})

    dataSub = data[(data['Price'] != 0) & ('09:15:00' < data['Time'])]
    HighPrice = max(dataSub['Price'])  # 涨停价
    HighPriceData = dataSub[dataSub['Price'] == HighPrice].copy()  # 涨停价交易信息

    # 第一次封板发生时的信息:先判断是否存在主卖，没有则判断最后一笔成交，若小于14:57:00，则判定为封板(假设一定存在尾盘撮合交易)
    if np.isin('S', HighPriceData['Type']):
        UpEndInfo = HighPriceData[HighPriceData['Type'] == 'S'].iloc[0]
        res['UpLimitJudge'] = "Sell"
    else:
        UpEndInfo = HighPriceData[HighPriceData['Time'] < '14:57:00'].iloc[-1]
        res['UpLimitJudge'] = "Time"
    res['UpLimitTime'] = UpEndInfo['Time']  # 封板时间

    FirstUpInfo = dataSub.loc[: UpEndInfo.name].iloc[:-1]  # 第一次封板前交易信息

    # 判断封板发生的时间段
    if '09:00:00' <= UpEndInfo['Time'] < '09:30:00':
        res['UpLimitFlag'] = 'CallAM'
    elif '09:30:00' <= UpEndInfo['Time'] < '11:31:00':
        res['UpLimitFlag'] = 'TradeAM'
    elif '13:00:00' <= UpEndInfo['Time'] < '14:56:00':
        res['UpLimitFlag'] = 'TradePM'
    else:
        res['UpLimitFlag'] = 'CallPM'

    # 集合竞价阶段封板不考虑
    if res['UpLimitFlag'] not in ['TradeAM', 'TradePM']:
        return pd.Series(res)

    # 倒数第二笔订单:导致封板的最后一笔交易一定是主买订单，按照买单ID去重取倒数第二个买单最后一笔交易价格
    UpOrderInfo = FirstUpInfo.drop_duplicates(subset=['BuyOrderID'], keep='last')
    if UpOrderInfo.shape[0] >= 2:
        res['Last2OrderPriceRange'] = HighPrice - UpOrderInfo.iloc[-2]['Price']
        res['Last2OrderPriceRatio'] = res['Last2OrderPriceRange'] / HighPrice

    # 封板前1,3,5秒对应的成交价距离最高价有几跳以及区间内的主买订单数
    timeSec = np.dot(np.array(res['UpLimitTime'].split(':'), dtype=int), [3600, 60, 1])
    for timeRange in [1, 3, 5]:
        timeBefore = time.strftime("%H:%M:%S", time.gmtime(timeSec - timeRange))
        if '11:30:00' < timeBefore < '13:00:00':  # 处理中午时间段
            timeBefore = time.strftime("%H:%M:%S", time.gmtime(timeSec - timeRange - 5400))

        oldInfoT = dataSub[(dataSub['Time'] >= timeBefore) & (dataSub['Time'] <= res['UpLimitTime'])]
        oldInfoFirstT = dataSub[dataSub['Time'] == timeBefore]

        if not oldInfoFirstT.empty:
            res[f'Last{timeRange}SecToHighRange'] = HighPrice - oldInfoFirstT.iloc[0]['Price']  # 价格出现的第一笔交易
            res[f'Last{timeRange}SecToHighRatio'] = res[f'Last{timeRange}SecToHighRange'] / HighPrice

        if not oldInfoT.empty:
            res[f'Last{timeRange}SecToHighOrderNum'] = oldInfoT[oldInfoT['Type'] == 'B'].drop_duplicates(subset=['BuyOrderID']).shape[0]

    # 封板前1~10跳对应的成交价距离封板所需的时间:出现该价格的最后一笔
    for priceTick in range(1, 11):
        oldPrice = HighPrice - priceTick
        oldInfoP = dataSub[(dataSub['Price'] == oldPrice) & (dataSub['Time'] <= res['UpLimitTime'])]
        if not oldInfoP.empty:
            staTime = np.dot(np.array(oldInfoP['Time'].iloc[-1].split(':'), dtype=int), [3600, 60, 1])
            res[f'Last{priceTick}PriceToHighTime'] = timeSec - staTime

    # 开板判断
    newInfo = dataSub[dataSub['Time'] >= res['UpLimitTime']]
    if not newInfo.empty:
        if res['UpLimitJudge'] == "Sell" and np.isin('B', newInfo['Type']):
            breakLimitT = newInfo[newInfo['Type'] == 'B'].iloc[0]['Time']
        elif min(newInfo['Price']) < HighPrice:
            breakLimitT = newInfo[newInfo['Price'] < HighPrice].iloc[0]['Time']
        else:
            breakLimitT = None

        if breakLimitT is not None:
            timeRage = np.dot(np.array(breakLimitT.split(':'), dtype=int), [3600, 60, 1]) - \
                       np.dot(np.array(res['UpLimitTime'].split(':'), dtype=int), [3600, 60, 1])
            res['BreakLimitTime'] = breakLimitT
            res['LimitToBreakTime'] = timeRage

    return pd.Series(res)
