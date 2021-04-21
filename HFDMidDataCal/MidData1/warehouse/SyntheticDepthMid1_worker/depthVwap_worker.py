import pandas as pd
from typing import Dict, Any


# 十档Vwap计算公式
def depthVwap_worker(data: pd.DataFrame,
                     code: str,
                     date: str,
                     *args,
                     **kwargs) -> Dict[str, Any]:
    data['time'] = data['时间'].apply(lambda x: x[-8:])
    l = data['time'] >= '09:25:00'
    data = data[l].copy()

    res = {'date': date, 'code': code}

    l = data['最新价'] != 0
    res['open'] = data.loc[l, '最新价'].iloc[0]
    res['close'] = data.loc[l, '最新价'].iloc[-1]
    res['low'] = data.loc[l, '最新价'].min()
    res['high'] = data.loc[l, '最新价'].max()
    res['volume'] = data['总量'].iloc[-1]
    res['amount'] = data['总金额'].iloc[-1]

    data['amountdiff'] = data['总金额'] - data['总金额'].shift(1)
    data['bid'], data['ask'] = 0, 0
    for i in range(1, 11):
        data['bid'] = data['bid'] + data['挂买价' + str(i)] * data['挂买量' + str(i)] * (1.1 - 0.1 * i)
        data['ask'] = data['bid'] + data['挂卖价' + str(i)] * data['挂卖量' + str(i)] * (1.1 - 0.1 * i)
    data['spread'] = (data['bid'] - data['ask']) / (data['bid'] + data['ask'])
    l_t = (data['time'] >= '09:30:00') & (data['time'] <= '15:00:00')
    l = l_t & (data['amountdiff'] <= data.loc[l_t, 'amountdiff'].quantile(0.5))
    res['Speard'] = data.loc[l, 'spread'].mean()
    res['AmountMean'] = res['amount'] / data['总成交笔数'].iloc[-1]
    price_dict = {'1h': '10:30:00', '2h': '11:30:00', '3h': '14:00:00', '4h': '15:00:00'}
    for key, value in price_dict.items():
        l = data['time'] <= value
        if sum(l) > 0:
            res[key + 'Price'] = data.loc[l, '最新价'].iloc[-1]

    vwap_dict = {'0min': '09:29:50', '1min': '09:31:00', '3min': '09:33:00', '5min': '09:35:00',
                 '10min': '09:40:00',
                 '30min': '10:00:00', '60min': '10:30:00', '90min': '11:00:00', '120min': '11:30:00',
                 '150min': '13:30:00', '180min': '14:00:00', '210min': '14:30:00', '240min': '15:00:00'}
    for key, value in vwap_dict.items():
        l = data['time'] <= value
        if sum(l) > 0:
            amount = data.loc[l, '总金额'].iloc[-1]
            volume = data.loc[l, '总量'].iloc[-1]
            res[key + 'Amount'] = amount
            res[key + 'Volume'] = volume
    return res
