import pandas as pd


# 十档1min中间过程
def depth1min_worker(data: pd.DataFrame,
                     code: str,
                     *args,
                     **kwargs) -> pd.DataFrame:
    data['time'] = data['时间'].apply(lambda x: x[-8:-2] + '00')
    l = data['time'] >= '09:25:00'
    data = data[l].copy()

    l = data['time'] < '09:30:00'
    data.loc[l, 'time'] = '09:25:00'

    l = data['time'] >= '15:00:00'
    data.loc[l, 'time'] = '15:00:00'
    if sum(l) >= 1:
        data = data.loc[~l, :].append(data.loc[l, :].iloc[[0], :])

    l = data['time'] == '11:30:00'
    data.loc[l, 'time'] = '11:29:00'

    l = (data['time'] <= '11:30:00') | (data['time'] >= '13:00:00')
    data = data[l]

    group = data.groupby('time')
    df_temp = group.last()

    df_res = group['最新价'].ohlc()
    df_res['time'] = df_res.index
    df_res['code'] = code

    df_res['volume'] = df_temp['总量'] - df_temp['总量'].shift(1)
    df_res.loc[df_res.index[0], 'volume'] = df_temp['总量'].iloc[0]

    df_res['amount'] = df_temp['总金额'] - df_temp['总金额'].shift(1)
    df_res.loc[df_res.index[0], 'amount'] = df_temp['总金额'].iloc[0]

    df_res['tradenum'] = df_temp['总成交笔数'] - df_temp['总成交笔数'].shift(1)
    df_res.loc[df_res.index[0], 'tradenum'] = df_temp['总成交笔数'].iloc[0]

    df_res['bidprice1'] = df_temp['挂买价1']
    df_res['bidvolume1'] = df_temp['挂买量1']
    df_res['askprice1'] = df_temp['挂卖价1']
    df_res['askvolume1'] = df_temp['挂卖量1']

    df_res['askvolume5sum'] = 0
    df_res['bidvolume5sum'] = 0
    for i in range(1, 6):
        df_res['askvolume5sum'] += df_temp['挂卖量' + str(i)]
        df_res['bidvolume5sum'] += df_temp['挂买量' + str(i)]

    df_res['askvolume10sum'] = df_res['askvolume5sum']
    df_res['bidvolume10sum'] = df_res['bidvolume5sum']
    for i in range(1, 6):
        df_res['askvolume10sum'] += df_temp['挂卖量' + str(i)]
        df_res['bidvolume10sum'] += df_temp['挂买量' + str(i)]

    result = df_res
    return result
