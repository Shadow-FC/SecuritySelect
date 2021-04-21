import numpy as np
import pandas as pd


# 逐笔1min中间过程(用来计算中间过程2)
def trade1min_worker(data: pd.DataFrame,
                     code: str,
                     *args,
                     **kwargs) -> pd.DataFrame:
    data['time'] = data['Time'].apply(lambda x: x[:6] + '00')

    l = data['time'] < '09:30:00'
    data.loc[l, 'time'] = '09:25:00'

    l = data['time'] == '11:30:00'
    data.loc[l, 'time'] = '11:29:00'

    l_1 = (data['time'] <= '15:00:00') & (data['time'] >= '13:00:00')
    l_2 = data['time'] <= '11:30:00'
    data = data[l_1 | l_2]

    data['Amount'] = data['Price'] * data['Volume']
    data['isbuy'] = data['Type'] == 'B'
    data['buynum'] = np.where(data['isbuy'], 1, 0)
    data['buyamount'] = np.where(data['isbuy'], data['Amount'], 0)
    data['buyvolume'] = np.where(data['isbuy'], data['Volume'], 0)

    group = data.groupby('time')
    df_res = group['Price'].ohlc()
    df_res['time'] = df_res.index
    df_res['code'] = code
    df_res['volume'] = group['Volume'].sum()
    df_res['amount'] = group['Amount'].sum()
    df_res['tradenum'] = group['Price'].count()
    df_res['buyvolume'] = group['buyvolume'].sum()
    df_res['buyamount'] = group['buyamount'].sum()
    df_res['buytradenum'] = group['buynum'].sum()
    if '09:25:00' in df_res.index:
        df_res.loc['09:25:00', 'buyvolume'] = round(df_res.loc['09:25:00', 'volume'] / 2)
        df_res.loc['09:25:00', 'buyamount'] = df_res.loc['09:25:00', 'amount'] / 2
        df_res.loc['09:25:00', 'buytradenum'] = round(df_res.loc['09:25:00', 'tradenum'] / 2)
    if '15:00:00' in df_res.index:
        df_res.loc['15:00:00', 'buyvolume'] = round(df_res.loc['15:00:00', 'volume'] / 2)
        df_res.loc['15:00:00', 'buyamount'] = df_res.loc['15:00:00', 'amount'] / 2
        df_res.loc['15:00:00', 'buytradenum'] = round(df_res.loc['15:00:00', 'tradenum'] / 2)
    result = df_res
    return result
