# -*-coding:utf-8-*-
# @Time:   2021/2/25 14:19
# @Author: FC
# @Email:  18817289038@163.com

import os
import time
import json
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as st
import multiprocessing as mp
from itertools import zip_longest
from typing import Callable, List, Union, Dict, Any, Iterable

from Data.GetData import SQL

warnings.filterwarnings('ignore')

"""中间过程1"""


class HFDBase(object):
    CPU = mp.cpu_count() - 1

    MethodMapping = {"Trade": {},
                     "Depth": {}}

    def __init__(self, savePath: Dict[str, str]):
        """

        Args:
            savePath (): 存储路径字典
        """
        self.savePath: Dict[str, str] = savePath
        self.sql: type = SQL()
        self.stockIDMapping: Dict[str, str] = {}

        self.init()

    def init(self):
        data = self.sql.query(self.sql.stockID())
        self.stockIDMapping = data.set_index('CODE_SIMPLE').to_dict()['CODE']

    # 数据存储
    def data_to_csv(self, Data: Dict[str, pd.DataFrame]):

        for fileName, dataValue in Data.items():
            if fileName.endswith('1min'):
                res1.to_csv(os.path.join(self.savePath[fileName], f'{self.date}'), index=False)
            else:
                res1.to_csv(os.path.join(self.savePath[fileName], f'{fileName}.csv'), index=False)

    def filePath(self, **kwargs):
        pass

    def process(self, **kwargs):
        pass

    # 记事本
    def to_txt(self, Title: str, content: str):
        with open(os.path.join(os.getcwd(), f"{Title}.txt"), mode='a', encoding='UTF-8') as f:
            f.writelines(content + "\n")


class HFDTrade(HFDBase):
    """
    逐笔委托数据处理
    """

    def __init__(self, path: str, savePath: Dict[str, str]):
        super(HFDTrade, self).__init__(savePath)

        self.path: str = path

    # 获取逐笔委托文件(csv)所在路径
    def filePath(self) -> List[str]:
        subFolders1 = os.listdir(self.path)

        subFolderPath = []
        for subFolder in subFolders1:
            subFolders2 = os.listdir(os.path.join(self.path, subFolder))
            subFolderPath += [os.path.join(os.path.join(self.path, subFolder), x) for x in subFolders2]

        return subFolderPath

    def errorInfo(self, info):
        print(info)

    def multiProcess(self):
        filePath = sorted(self.filePath())
        ileGroup = list(zip_longest(*[iter(filePath)] * int(len(filePath) / self.CPU)))

        pool = mp.Pool(self.CPU)
        for group in ileGroup:  # fileGroup
            pool.apply_async(func=self.onCallFunc, args=(group, self.process,), error_callback=self.errorInfo)
        pool.close()
        pool.join()

    def singleProcess(self):
        filePath = sorted(self.filePath())
        ileGroup = list(zip_longest(*[iter(filePath)] * int(len(filePath) / self.CPU)))

        for group in ileGroup:  # [['Y:\\逐笔全息\\2019\\2019-09-05']
            self.onCallFunc(group, self.process)

    # 高频数据处理
    def onCallFunc(self, filePathList: Iterable, func: Callable):
        for path_ in filePathList:
            if path_ is None:
                continue
            func(path_)

    # 逐笔数据处理
    def process(self, path):
        files = os.listdir(path)
        _, date = os.path.split(path)

        tradeCashFlow, trade1min = [], []
        flag = 0
        for file_ in files:
            flag += 1
            if file_[:-4] not in self.stockIDMapping.keys():
                continue
            sta = time.time()
            try:
                data = pd.read_csv(os.path.join(path, file_), encoding='GBK')
            except Exception as e:
                print(f"tradeReadFile-{date}-{file_}: {e}")
                self.to_txt("Trade", f"{dt.datetime.now()}: readFile {date} {file_}")
            else:
                try:
                    tradeCashFlow.append(
                        pd.Series(self.tradeCashFlow(data, self.stockIDMapping[file_[:-4]], date)))  # 逐笔资金流向计算
                except Exception as e:
                    print(f"tradeCashFlow-{date}-{file_}: {e}")
                    self.to_txt("Trade", f"{dt.datetime.now()}: tradeCashFlow {date} {file_}")

                try:
                    trade1min.append(self.trade1min(data, self.stockIDMapping[file_[:-4]]))  # 逐笔1min计算
                except Exception as e:
                    print(f"trade1min-{date}-{file_}: {e}")
                    self.to_txt("Trade", f"{dt.datetime.now()}: trade1min {date} {file_}")

            if flag % 200 == 0:
                print(f"逐笔数据-{date}：{flag:>4}/{len(files):>4}-{file_}, 耗时{round(time.time() - sta, 4)}")

        res1 = pd.concat(tradeCashFlow, axis=1).T
        res2 = pd.concat(trade1min)

        res1.to_csv(os.path.join(self.savePath['tradeCashFlow'], f'CashFlow-{date}.csv'), index=False)
        res2.to_csv(os.path.join(self.savePath['trade1min'], date + '.csv'), index=False)

    # 逐笔资金流向中间过程
    def tradeCashFlow(self, df_data, code, date):
        res = {'code': code, 'date': date}
        df_data['Amount'] = df_data['Price'] * df_data['Volume']
        res['high'] = df_data['Price'].max()
        res['open'] = df_data['Price'].iloc[0]
        res['close'] = df_data['Price'].iloc[-1]
        res['low'] = df_data['Price'].min()
        res['volume'] = df_data['Volume'].sum()
        res['amount'] = df_data['Amount'].sum()

        res['BuyOrderVolume'] = df_data.groupby('BuyOrderID')['BuyOrderVolume'].first().sum()
        res['SaleOrderVolume'] = df_data.groupby('SaleOrderID')['SaleOrderVolume'].first().sum()

        l_t = (df_data['Time'] >= '09:30:00') & (df_data['Time'] <= '14:57:00')
        l_isbuy = df_data['Type'] == 'B'

        res['AmountMean'] = df_data.loc[l_t, 'Amount'].mean()
        res['AmountStd'] = df_data.loc[l_t, 'Amount'].std()
        res['BuyMean'] = df_data.loc[l_t & l_isbuy, 'Amount'].mean()
        res['SaleMean'] = df_data.loc[l_t & (~l_isbuy), 'Amount'].mean()
        for i in range(1, 10):
            res['AmountQuantile' + '_' + str(i)] = df_data.loc[l_t, 'Amount'].quantile(i * 0.1)
            res['BuyQuantile' + '_' + str(i)] = df_data.loc[l_t & l_isbuy, 'Amount'].quantile(i * 0.1)
            res['SaleQuantile' + '_' + str(i)] = df_data.loc[l_t & (~l_isbuy), 'Amount'].quantile(i * 0.1)

        BigOrderMeanStd, BigOrderPercentile = res['AmountMean'] + res['AmountStd'], res['AmountQuantile_9']
        l_BigOrderMeanStd = df_data['Amount'] >= BigOrderMeanStd
        l_BigOrderPercentile = df_data['Amount'] >= BigOrderPercentile

        time_dict = {'AM_30min': ['09:30:00', '10:00:00'], 'AM_60min': ['09:30:00', '10:30:00'],
                     'AM_120min': ['09:30:00', '11:30:00'],
                     'PM_30min': ['14:30:00', '14:57:00'], 'PM_60min': ['14:00:00', '14:57:00'],
                     'PM_120min': ['13:00:00', '14:57:00'], }

        for key, value in time_dict.items():
            l_t = (df_data['Time'] >= value[0]) & (df_data['Time'] <= value[1])
            res['BuyAll' + '_' + key] = df_data.loc[l_t & l_isbuy, 'Amount'].sum()
            res['SaleAll' + '_' + key] = df_data.loc[l_t & (~l_isbuy), 'Amount'].sum()
            res['BuyBigOrderMeanStd' + '_' + key] = df_data.loc[l_t & l_isbuy & l_BigOrderMeanStd, 'Amount'].sum()
            res['SaleBigOrderMeanStd' + '_' + key] = df_data.loc[l_t & (~l_isbuy) & l_BigOrderMeanStd, 'Amount'].sum()
            res['BuyBigOrderPercentile' + '_' + key] = df_data.loc[l_t & l_isbuy & l_BigOrderPercentile, 'Amount'].sum()
            res['SaleBigOrderPercentile' + '_' + key] = df_data.loc[
                l_t & (~l_isbuy) & l_BigOrderPercentile, 'Amount'].sum()
        return res

    # 逐笔1min中间过程(用来计算中间过程2)
    def trade1min(self, df_data, code):
        df_data['time'] = df_data['Time'].apply(lambda x: x[:6] + '00')

        l = df_data['time'] < '09:30:00'
        df_data.loc[l, 'time'] = '09:25:00'

        l = df_data['time'] == '11:30:00'
        df_data.loc[l, 'time'] = '11:29:00'

        l_1 = (df_data['time'] <= '15:00:00') & (df_data['time'] >= '13:00:00')
        l_2 = df_data['time'] <= '11:30:00'
        df_data = df_data[l_1 | l_2]

        df_data['Amount'] = df_data['Price'] * df_data['Volume']
        df_data['isbuy'] = df_data['Type'] == 'B'
        df_data['buynum'] = np.where(df_data['isbuy'], 1, 0)
        df_data['buyamount'] = np.where(df_data['isbuy'], df_data['Amount'], 0)
        df_data['buyvolume'] = np.where(df_data['isbuy'], df_data['Volume'], 0)

        group = df_data.groupby('time')
        df_res = group['Price'].ohlc()
        df_res['time'] = df_res.index
        df_res['code'] = code
        df_res['volume'] = group['Volume'].sum()
        df_res['amount'] = group['Amount'].sum()
        df_res['tradenum'] = group['TranID'].count()
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


class HFDDepth(HFDBase):
    """
    十档数据处理
    """

    def __init__(self, path: str, savePath: Dict[str, str]):
        super(HFDDepth, self).__init__(savePath)

        self.path = path

    # 获取十档文件(csv)所在路径
    def filePath(self) -> List[str]:
        subFolders1 = os.listdir(self.path)

        subFolderPath = []
        for subFolder in subFolders1:
            subFolders2 = os.listdir(os.path.join(self.path, subFolder))
            subFolderPath += [os.path.join(os.path.join(self.path, subFolder), x) for x in subFolders2]

        return subFolderPath

    def errorInfo(self, info):
        print(info)

    def multiProcess(self):
        filePath = sorted(self.filePath())
        ileGroup = list(zip_longest(*[iter(filePath)] * int(len(filePath) / self.CPU)))

        pool = mp.Pool(self.CPU)
        for group in ileGroup:  # fileGroup
            pool.apply_async(func=self.onCallFunc, args=(group, self.process,), error_callback=self.errorInfo)
        pool.close()
        pool.join()

    def singleProcess(self):
        filePath = sorted(self.filePath())
        ileGroup = list(zip_longest(*[iter(filePath)] * int(len(filePath) / self.CPU)))

        for group in ileGroup:
            self.onCallFunc(group, self.process)

    # 高频数据处理
    def onCallFunc(self, filePathList: Iterable, func: Callable):
        for path_ in filePathList:
            if path_ is None:
                continue
            func(path_)

    # 深度数据中间过程1计算
    def process(self, path: str):
        """
        need input a folder path
        Args:
            path ():

        Returns:

        """
        files = os.listdir(path)
        _, date_raw = os.path.split(path)
        date = date_raw[:4] + '-' + date_raw[4:6] + '-' + date_raw[-2:]

        depth1min, depthVwap = [], []
        flag = 0
        for file_ in files:
            flag += 1
            if file_[2:-4] not in self.stockIDMapping.keys():
                continue
            sta = time.time()
            try:
                data = pd.read_csv(os.path.join(path, file_), encoding='GBK')
            except Exception as e:
                print(f"depthReadFile-{date}-{file_}: {e}")
                self.to_txt("Depth", f"{dt.datetime.now()}: readFile {date} {file_}")
            else:
                try:
                    depthVwap.append(pd.Series(self.depthVwap(data, f"{file_[2:-4]}.{file_[:2].upper()}", date)))  # 十档Vwap
                except Exception as e:
                    print(f"depthVwap-{date}-{file_}: {e}")
                    self.to_txt("Depth", f"{dt.datetime.now()}: depthVwap {date} {file_}")

                try:
                    depth1min.append(self.depth1min(data, f"{file_[2:-4]}.{file_[:2].upper()}"))  # 十档1min
                except Exception as e:
                    print(f"depth1min-{date}-{file_}: {e}")
                    self.to_txt("Depth", f"{dt.datetime.now()}: depth1min {date} {file_}")

            if flag % 200 == 0:
                print(f"十档数据-{date}：{flag:>5}/{len(files):>5}-{file_}, 耗时{round(time.time() - sta, 4)}")

        res1 = pd.concat(depthVwap, axis=1).T
        res2 = pd.concat(depth1min)

        res1.to_csv(os.path.join(self.savePath['depthVwap'], f'VwapFactor-{date}.csv'), index=False)
        res2.to_csv(os.path.join(self.savePath['depth1min'], f'{date}.csv'), index=False)

    # 十档Vwap中间过程
    def depthVwap(self, df_data: pd.DataFrame, code: str, date: str) -> Dict[str, Any]:
        df_data['time'] = df_data['时间'].apply(lambda x: x[-8:])
        l = df_data['time'] >= '09:25:00'
        df_data = df_data[l].copy()

        res = {'date': date, 'code': code}

        l = df_data['最新价'] != 0
        res['open'] = df_data.loc[l, '最新价'].iloc[0]
        res['close'] = df_data.loc[l, '最新价'].iloc[-1]
        res['low'] = df_data.loc[l, '最新价'].min()
        res['high'] = df_data.loc[l, '最新价'].max()
        res['volume'] = df_data['总量'].iloc[-1]
        res['amount'] = df_data['总金额'].iloc[-1]

        df_data['amountdiff'] = df_data['总金额'] - df_data['总金额'].shift(1)
        df_data['bid'], df_data['ask'] = 0, 0
        for i in range(1, 11):
            df_data['bid'] = df_data['bid'] + df_data['挂买价' + str(i)] * df_data['挂买量' + str(i)] * (1.1 - 0.1 * i)
            df_data['ask'] = df_data['bid'] + df_data['挂卖价' + str(i)] * df_data['挂卖量' + str(i)] * (1.1 - 0.1 * i)
        df_data['spread'] = (df_data['bid'] - df_data['ask']) / (df_data['bid'] + df_data['ask'])
        l_t = (df_data['time'] >= '09:30:00') & (df_data['time'] <= '15:00:00')
        l = l_t & (df_data['amountdiff'] <= df_data.loc[l_t, 'amountdiff'].quantile(0.5))
        res['Speard'] = df_data.loc[l, 'spread'].mean()
        res['AmountMean'] = res['amount'] / df_data['总成交笔数'].iloc[-1]
        price_dict = {'1h': '10:30:00', '2h': '11:30:00', '3h': '14:00:00', '4h': '15:00:00'}
        for key, value in price_dict.items():
            l = df_data['time'] <= value
            if sum(l) > 0:
                res[key + 'Price'] = df_data.loc[l, '最新价'].iloc[-1]

        vwap_dict = {'0min': '09:29:50', '1min': '09:31:00', '3min': '09:33:00', '5min': '09:35:00',
                     '10min': '09:40:00',
                     '30min': '10:00:00', '60min': '10:30:00', '90min': '11:00:00', '120min': '11:30:00',
                     '150min': '13:30:00', '180min': '14:00:00', '210min': '14:30:00', '240min': '15:00:00'}
        for key, value in vwap_dict.items():
            l = df_data['time'] <= value
            if sum(l) > 0:
                amount = df_data.loc[l, '总金额'].iloc[-1]
                volume = df_data.loc[l, '总量'].iloc[-1]
                res[key + 'Amount'] = amount
                res[key + 'Volume'] = volume

        return res

    # 十档1min中间过程(用来计算中间过程2)
    def depth1min(self, df_data: pd.DataFrame, code: str) -> pd.DataFrame:
        df_data['time'] = df_data['时间'].apply(lambda x: x[-8:-2] + '00')
        l = df_data['time'] >= '09:25:00'
        df_data = df_data[l].copy()

        l = df_data['time'] < '09:30:00'
        df_data.loc[l, 'time'] = '09:25:00'

        l = df_data['time'] >= '15:00:00'
        df_data.loc[l, 'time'] = '15:00:00'
        if sum(l) >= 1:
            df_data = df_data.loc[~l, :].append(df_data.loc[l, :].iloc[[0], :])

        l = df_data['time'] == '11:30:00'
        df_data.loc[l, 'time'] = '11:29:00'

        l = (df_data['time'] <= '11:30:00') | (df_data['time'] >= '13:00:00')
        df_data = df_data[l]

        group = df_data.groupby('time')
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


if __name__ == '__main__':
    HFDTrade("", {})
    HFDDepth("", {})
