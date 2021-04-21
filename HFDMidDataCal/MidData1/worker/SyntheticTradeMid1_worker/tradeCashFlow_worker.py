import pandas as pd
from typing import Dict, Any


# 逐笔资金流向中间过程
def tradeCashFlow_worker(data: pd.DataFrame,
                         code: str,
                         date: str,
                         *args,
                         **kwargs) -> Dict[str, Any]:
    res = {'code': code, 'date': date}

    data['Amount'] = data['Price'] * data['Volume']
    res['high'] = data['Price'].max()
    res['open'] = data['Price'].iloc[0]
    res['close'] = data['Price'].iloc[-1]
    res['low'] = data['Price'].min()
    res['volume'] = data['Volume'].sum()
    res['amount'] = data['Amount'].sum()

    res['BuyOrderVolume'] = data.groupby('BuyOrderID')['BuyOrderVolume'].first().sum()
    res['SaleOrderVolume'] = data.groupby('SaleOrderID')['SaleOrderVolume'].first().sum()

    l_t = (data['Time'] >= '09:30:00') & (data['Time'] <= '14:57:00')
    l_isbuy = data['Type'] == 'B'

    res['AmountMean'] = data.loc[l_t, 'Amount'].mean()
    res['AmountStd'] = data.loc[l_t, 'Amount'].std()
    res['BuyMean'] = data.loc[l_t & l_isbuy, 'Amount'].mean()
    res['SaleMean'] = data.loc[l_t & (~l_isbuy), 'Amount'].mean()
    for i in range(1, 10):
        res['AmountQuantile' + '_' + str(i)] = data.loc[l_t, 'Amount'].quantile(i * 0.1)
        res['BuyQuantile' + '_' + str(i)] = data.loc[l_t & l_isbuy, 'Amount'].quantile(i * 0.1)
        res['SaleQuantile' + '_' + str(i)] = data.loc[l_t & (~l_isbuy), 'Amount'].quantile(i * 0.1)

    BigOrderMeanStd, BigOrderPercentile = res['AmountMean'] + res['AmountStd'], res['AmountQuantile_9']
    l_BigOrderMeanStd = data['Amount'] >= BigOrderMeanStd
    l_BigOrderPercentile = data['Amount'] >= BigOrderPercentile

    time_dict = {'AM_30min': ['09:30:00', '10:00:00'], 'AM_60min': ['09:30:00', '10:30:00'],
                 'AM_120min': ['09:30:00', '11:30:00'],
                 'PM_30min': ['14:30:00', '14:57:00'], 'PM_60min': ['14:00:00', '14:57:00'],
                 'PM_120min': ['13:00:00', '14:57:00'], }

    for key, value in time_dict.items():
        l_t = (data['Time'] >= value[0]) & (data['Time'] <= value[1])
        res['BuyAll' + '_' + key] = data.loc[l_t & l_isbuy, 'Amount'].sum()
        res['SaleAll' + '_' + key] = data.loc[l_t & (~l_isbuy), 'Amount'].sum()
        res['BuyBigOrderMeanStd' + '_' + key] = data.loc[l_t & l_isbuy & l_BigOrderMeanStd, 'Amount'].sum()
        res['SaleBigOrderMeanStd' + '_' + key] = data.loc[l_t & (~l_isbuy) & l_BigOrderMeanStd, 'Amount'].sum()
        res['BuyBigOrderPercentile' + '_' + key] = data.loc[l_t & l_isbuy & l_BigOrderPercentile, 'Amount'].sum()
        res['SaleBigOrderPercentile' + '_' + key] = data.loc[l_t & (~l_isbuy) & l_BigOrderPercentile, 'Amount'].sum()
    return res
