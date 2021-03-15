# -*-coding:utf-8-*-
# @Time:   2020/12/28 17:04
# @Author: FC
# @Email:  18817289038@163.com

import os
import time
import json
import numpy as np
import collections
import pandas as pd
import datetime as dt

import scipy.stats as st
from typing import Callable, List, Union, Dict
import multiprocessing as mp

"""
高频数据计算注意事项：
1.如果不做特殊说明，所有因子的计算不包含集合竞价数据
2.日内第一根K线为09:30:00，最后一根K线为15:00:00
3.对于深度数据的统计，默认剔除14:57:00到15:00:00的数据，方便统计
4.缺失分钟数据不填充，当根K线无成交，默认成交量为0
5.在进行差分或者收益率计算时，先划分区间再进行计算，例如，先剔除集合竞价后再计算收益率
6.数据进过差分或者收益率变换后，在按照收益率或者差分后的数据对样本进行划分时，考虑首根K线因为为空，将其划分为0的区间内
7.不同时间段计算的因子是否需要包含时间段两端详见算法
"""


def entropy(x: pd.Series, bottom: int = 2):
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
    if np.isnan(rev_struct):
        print("Nan error!")
    return rev_struct


#  基于逐笔一分钟数据构建的因子中间数据
class MidData(object):

    pathIn = {"Trade": "",  # 高频分钟数据
              "Depth": ""  # 高频十档分钟数据
              }

    MethodMapping = {"Trade": {},
                     "Depth": {}}

    close_price = {
        '0h': '09:30:00',
        '0.5h': '10:00:00',
        '1h': '10:30:00',
        '1.5h': '11:00:00',
        '2h': '11:30:00',
        '2.5h': '13:30:00',
        '3h': '14:00:00',
        '3.5h': '14:30:00',
        '4h': '15:00:00'
    }

    time_AM = {"call": "09:30:00",
               "5min": "09:35:00",
               "10min": "09:40:00",
               "15min": "09:45:00",
               "30min": "10:00:00",
               "60min": "10:30:00",
               "all": "11:30:00"}

    time_PM = {"5min": "14:55:00",
               "10min": "14:50:00",
               "15min": "14:45:00",
               "30min": "14:30:00",
               "60min": "14:00:00",
               "all": "13:00:00"}

    time_std = {"all": ["09:30:00", "15:00:00"],
                "open": ["09:30:00", "10:00:00"],
                "between": ["10:00:00", "14:30:00"],
                "close": ["14:30:00", "15:00:00"]}

    CPU = mp.cpu_count() - 1

    json_name_ID = 'CheckedID'
    json_name_Error = 'ErrorID'

    json_path = os.path.join(os.getcwd(), f'{json_name_ID}.json')

    def __init__(self, pathIn: str, pathOut: str):
        self.pathOut: str = pathOut
        self.pathIn: str = pathIn

        self.file_name = 'Depth'
        self.lock = mp.Manager().Lock()

        self.path_files = {}
        self.flag = 0

        self.get_func()

    def range_T(self, data: pd.DataFrame):
        return (data['time'] >= '09:30:00') & (data['time'] < '15:00:00')

    def get_func(self):
        method_name = self.__dir__()
        for key_ in self.MethodMapping.keys():
            self.MethodMapping[key_] = \
                {''.join(list(map(lambda x: x.capitalize(), i.split('_')))): self.__getattribute__(i)
                 for i in method_name if i.startswith(f'_{key_.lower()}')}

    # Generator file(csv) path
    def get_file_path_list(self) -> List[str]:

        files = os.listdir(self.pathIn)
        path_files = [os.path.join(self.pathIn, file_name) for file_name in files if file_name != '1min']

        return path_files

    def callfunc(self,
                 data: pd.DataFrame,
                 date: str,
                 func_name: str,
                 func: Callable):
        try:
            res = data.groupby('code').apply(func, date)
        except Exception as e:
            print(f"{func_name} {date}: {e}")
            with self.lock:
                self.save_json(json_name=self.json_name_Error, data_name=func_name, data_save=date)
            res = pd.DataFrame()
        return res

    # 收益率相关中间过程
    def _trade_ret(self, d: pd.DataFrame, date: str) -> pd.Series:
        d_sub = d[self.range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        retDiff = d_sub['ret'].diff(1)
        retData = {
            "retDiffMean": retDiff.mean(),  # 收益率差分均值
            "retDiffAbsMean": abs(retDiff).mean(),  # 收益率差分绝对值均值

            "retMean": d_sub['ret'].mean(),  # 收益率均值
            "retAbsMean": abs(d_sub['ret']).mean(),  # 收益率绝对值均值

            "ret2Up_0": pow(d_sub['ret'][d_sub['ret'] > 0], 2).sum(),  # 收益率大于0的平方和
            "ret2Down_0": pow(d_sub['ret'][d_sub['ret'] < 0], 2).sum(),  # 收益率小于0的平方和

            "ret3Up_0": pow(d_sub['ret'][d_sub['ret'] > 0], 3).sum(),  # 收益率大于0的三次方和
            "ret3Down_0": pow(d_sub['ret'][d_sub['ret'] < 0], 3).sum(),  # 收益率小于0的三次方和

            "retVolWeight": (d_sub['ret'] * d_sub['volume']).sum() / d_sub['volume'].sum(),  # 成交量加权收益

            "retVar": d_sub['ret'].var(),  # 收益率方差
            "retSkew": d_sub['ret'].skew(),  # 收益率偏度
            "retKurt": d_sub['ret'].kurt(),  # 收益率峰度

            "date": date
        }
        return pd.Series(retData)

    # 成交量相关中间过程
    def _trade_vol(self, d: pd.DataFrame, date: str) -> pd.Series:

        d['volPerTrade'] = d['volume'] / d['tradenum']
        d_sub = d[self.range_T]

        volDiff = d_sub['volume'].diff(1)
        volPerDiff = d_sub['volPerTrade'].diff(1)

        vol_data = pd.Series({

            "volDiffMean": volDiff.mean(),  # 成交量差分均值
            "volDiffStd": volDiff.std(),  # 成交量差分标准差

            "volDiffAbsMean": abs(volDiff).mean(),  # 成交量差分绝对值均值
            "volDiffAbsStd": abs(volDiff).std(),  # 成交量差分绝对值标准差

            "volPerMean": d_sub['volPerTrade'].mean(),  # 每笔成交量均值
            "volPerStd": d_sub['volPerTrade'].std(),  # 每笔成交量标准差

            "volPerDiffMean": volPerDiff.mean(),  # 每笔成交量差分均值
            "volPerDiffStd": volPerDiff.std(),  # 每笔成交量差分标准差

            "volPerDiffAbsMean": abs(volPerDiff).mean(),  # 每笔成交量差分绝对值均值
            "volPerDiffAbsStd": abs(volPerDiff).std(),  # 每笔成交量差分绝对值标准差

            "date": date
        })

        return pd.Series(vol_data)

    # 成交笔数相关中间过程
    def _trade_trade_num(self, d: pd.DataFrame, date: str) -> pd.Series:
        d_sub = d[self.range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        tradeNumDiff = d_sub['tradenum'].diff(1)

        tradeNumData = {
            "tradeNumRetUpSum_0": d_sub[d_sub['ret'] > 0]['tradenum'].sum(),  # 收益率大于0的笔数和
            "tradeNumRetDownSum_0": d_sub[d_sub['ret'] < 0]['tradenum'].sum(),  # 收益率小于0的笔数和
            "tradeNumRetEqualSum_0": d_sub[np.isnan(d_sub['ret']) | (d_sub['ret'] == 0)]['tradenum'].sum(),
            # 收益率等于0的笔数和(包含收益率为空的数据)

            "tradeNumDiffMean": tradeNumDiff.mean(),  # 成交笔数差分均值
            "tradeNumDiffStd": tradeNumDiff.std(),  # 成交笔数差分标准差

            "tradeNumDiffAbsMean": abs(tradeNumDiff).mean(),  # 成交笔数差分绝对值均值
            "tradeNumDiffAbsStd": abs(tradeNumDiff).std(),  # 成交笔数差分绝对值标准差

            "date": date  # 日期
        }

        return pd.Series(tradeNumData)

    # 不同时间段总成交额和
    def _trade_amt_sum(self, d: pd.DataFrame, date: str) -> pd.Series:
        d_sub = d[self.range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        amtSumAM = {f"amtAM_{time_}": d[d['time'] < right_time]['amount'].sum()
                    for time_, right_time in self.time_AM.items()}  # 开盘不同时间段成交量和

        amtSumPM = {f"amtPM_{time_}": d[d['time'] >= left_time]['amount'].sum()
                    for time_, left_time in self.time_PM.items()}  # 尾盘不同时间段成交量和

        amtSumSp = {
            "amtRetUpSum_0": d_sub[d_sub['ret'] > 0]['amount'].sum(),  # 收益率大于0的成交额和
            "amtRetDownSum_0": d_sub[d_sub['ret'] < 0]['amount'].sum(),  # 收益率小于0的成交额和
            "amtRetEqualSum_0": d_sub[np.isnan(d_sub['ret']) | (d_sub['ret'] == 0)]['amount'].sum(),  # # 收益率等于0的成交额和
            "date": date  # 日期
        }

        amtSum = {**amtSumAM, **amtSumPM, **amtSumSp}

        return pd.Series(amtSum)

    # 不同时间段主买额和
    def _trade_buy_amt_sum(self, d: pd.DataFrame, date: str) -> pd.Series:
        buyAmtSumAM = {f"buyAmtSumAM_{t_}": d[d['time'] < T_r]['buyamount'].sum()
                       for t_, T_r in self.time_AM.items()}  # 开盘不同时间段主买额和

        buyAmtSumPM = {f"buyAmtSumPM_{t_}": d[d['time'] >= T_l]['buyamount'].sum()
                       for t_, T_l in self.time_PM.items()}  # 尾盘不同时间段主买额和

        buyAmtSum = {**buyAmtSumAM, **buyAmtSumPM, **{"date": date}}

        return pd.Series(buyAmtSum)

    # 不同时间段主卖额和
    def _trade_sell_amt_sum(self, d: pd.DataFrame, date: str) -> pd.Series:
        d['sellAmount'] = d['amount'] - d['buyamount']

        sellAmtSumAM = {f"sellAmtSumAM_{t_}": d[d['time'] < T_r]['sellAmount'].sum()
                        for t_, T_r in self.time_AM.items()}  # 开盘不同时间段主卖额和

        sellAmtSumPM = {f"sellAmtSumPM_{t_}": d[d['time'] >= T_l]['sellAmount'].sum()
                        for t_, T_l in self.time_PM.items()}  # 尾盘不同时间段主卖额和

        sellAmtSum = {**sellAmtSumAM, **sellAmtSumPM, **{"date": date}}

        return pd.Series(sellAmtSum)

    # 不同时间段成交额标准差
    def _trade_amt_std(self, d: pd.DataFrame, date: str) -> pd.Series:
        d['sellAmount'] = d['amount'] - d['buyamount']
        d['netAmount'] = d['buyamount'] - d['sellAmount']

        buyAmtStd = {f"buyAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['buyamount'].std()
                     for t_, T_ in self.time_std.items()}  # 不同时间段主买额标准差
        sellAmtStd = {f"sellAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['sellAmount'].std()
                      for t_, T_ in self.time_std.items()}  # 不同时间段主卖额标准差
        netAmtStd = {f"netAmtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['netAmount'].std()
                     for t_, T_ in self.time_std.items()}  # 不同时间段净主买额标准差

        allAmtAtd = {f"amtStd_{t_}": d[(d['time'] >= T_[0]) & (d['time'] < T_[1])]['amount'].std()
                     for t_, T_ in self.time_std.items()}  # 不同时间段成交额标准差

        amtStd = {**buyAmtStd, **sellAmtStd, **allAmtAtd, **netAmtStd, **{"date": date}}

        return pd.Series(amtStd)

    # 收盘价相关中间过程
    def _trade_close(self, d: pd.DataFrame, date: str) -> pd.Series:
        d_sub = d[self.range_T]

        # 收盘价相关中间过程
        closeData = {"close" + t_: 0 if d[d['time'] <= T_r].tail(1)['close'].empty
        else d[d['time'] <= T_r].tail(1)['close'].values[0]
                     for t_, T_r in self.close_price.items()}  # 不同时间截面收盘价

        closeData.update({
            "closeMean": d_sub['close'].mean(),  # 收盘价均值
            "closeStd": d_sub['close'].std(),  # 收盘价标准差
            "closeAmtWeight": (d_sub['close'] * d_sub['amount']).sum() / d_sub['amount'].sum(),  # 成交量加权收盘价
            "date": date  # 日期
        })

        return pd.Series(closeData)

    # 逐笔特殊因子1
    def _trade_special1(self, d: pd.DataFrame, date: str) -> pd.Series:
        d['amtPerTrade'] = d['amount'] / d['tradenum']
        d_sub = d[self.range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        d_sub1 = d_sub[d_sub['amtPerTrade'] >= d_sub['amtPerTrade'].quantile(0.8)]
        d_sub_inflow, d_sub_outflow = d_sub1[d_sub1['ret'] > 0], d_sub1[d_sub1['ret'] < 0]

        specialData = {
            "corCloseVol": d_sub[['close', 'volume']].corr().iloc[0, 1],  # 收盘价和成交量pearson相关系数
            "corRetVol": d_sub[['ret', 'volume']].corr().iloc[0, 1],  # 收益率和成交量pearson相关系数

            "closeVolWeightSkew": (pow((d_sub['close'] - d_sub['close'].mean()) / d_sub['close'].std(), 3) * (
                    d_sub['volume'] / d_sub['volume'].sum())).sum(),  # 加权收盘价偏度

            "AMTInFlowBigOrder": d_sub_inflow['amount'].sum(),  # 单笔成交量在前20%的成交量收益率大于零的和(大单流入)
            "AMTOutFlowBigOrder": d_sub_outflow['amount'].sum(),  # 单笔成交量在前20%的成交量收益率小于零的和(大单流出)

            "CashFlow": (np.sign(d_sub['close'].diff(1)) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 资金流向(成交量加权收盘价差分和)

            "MOMBigOrder": (d_sub[d_sub['amtPerTrade'] >= d_sub['amtPerTrade'].quantile(0.8)]['ret'] + 1).prod(
                min_count=1),  # 大单驱动涨幅

            "retD": np.log(1 + abs(np.log(d_sub['close'] / d_sub['close'].shift(1)))).sum(),
            # 轨迹非流动因子分母
            "RevStruct": func_Structured_reversal(d_sub, 0.1),  # 结构化反转因子

            "date": date  # 日期
        }

        return pd.Series(specialData)

    # 逐笔特殊因子2
    def _trade_special2(self, d: pd.DataFrame, date: str) -> pd.Series:
        d_sub = d[self.range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        specialData = {
            "volEntropy": entropy(d_sub['close'] * d_sub['volume']),  # 单位一成交量占比熵
            "amtEntropy": entropy(d_sub['amount']),  # 成交额占比熵

            "naiveAmtR": (st.t.cdf(d_sub['close'].diff(1) / d_sub['close'].diff(1).std(), len(d_sub) - 1) * d_sub[
                'amount']).sum() / d_sub['amount'].sum(),  # 朴素主动占比因子
            "TAmtR": (st.t.cdf(d_sub['ret'] / d_sub['ret'].std(), len(d_sub) - 1) * d_sub['amount']).sum() / d_sub[
                'amount'].sum(),  # T分布主动占比因子
            "NAmtR": (st.norm.cdf(d_sub['ret'] / d_sub['ret'].std()) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 正态分布主动占比因子
            "CNAmtR": (st.norm.cdf(d_sub['ret'] / 0.1 * 1.96) * d_sub['amount']).sum() / d_sub['amount'].sum(),
            # 置信正态分布主动占比因子
            "EventAmtR": ((d_sub["ret"] - 0.1) / 0.2 * d_sub['amount']).sum() / d_sub['amount'].sum(),  # 均匀分布主动占比因子

            "SmartQ": func_M_sqrt(d_sub),  # 聪明钱因子
            "SmartQln": func_M_ln(d_sub),  # 聪明钱因子改进

            "date": date  # 日期
        }

        return pd.Series(specialData)

    # 5档盘口委买委卖量和
    def _depth5_vol_sum(self, data: pd.DataFrame, date: str) -> pd.Series:
        dataSub = data[((data['time'] < '14:57:00') | (data['time'] == '15:00:00')) & (
                (data['bidvolume1'] != 0) | (data['askvolume1'] != 0))]

        bid5VolSum = {f"bid5VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['bidvolume5sum'].tail(1)).sum()
                      for t_, T_r in self.close_price.items()}  # 不同时间点5挡委买量和
        ask5VolSum = {f"ask5VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['askvolume5sum'].tail(1)).sum()
                      for t_, T_r in self.close_price.items()}  # 不同时间点5挡委卖量和

        depth5Sum = {**bid5VolSum, **ask5VolSum, **{"date": date}}

        return pd.Series(depth5Sum)

    # 10档盘口委买委卖量和
    def _depth10_vol_sum(self, data: pd.DataFrame, date: str) -> pd.Series:
        dataSub = data[((data['time'] < '14:57:00') | (data['time'] == '15:00:00')) & (
                (data['bidvolume1'] != 0) | (data['askvolume1'] != 0))]

        bid10VolSum = {f"bid10VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['bidvolume10sum'].tail(1)).sum()
                       for t_, T_r in self.close_price.items()}  # 不同时间点10挡委买量和
        ask10VolSum = {f"ask10VolSum_{t_}": np.array(dataSub[dataSub['time'] <= T_r]['askvolume10sum'].tail(1)).sum()
                       for t_, T_r in self.close_price.items()}  # 不同时间点10挡委卖量和

        depth10Sum = {**bid10VolSum, **ask10VolSum, **{"date": date}}

        return pd.Series(depth10Sum)

    # save to csv
    def to_csv(self, key: str, data: List[pd.DataFrame]):
        df = pd.concat(data)
        file_path = os.path.join(self.pathOut, key)
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        df.to_csv(os.path.join(file_path, f"{key}-{int(dt.datetime.now().strftime('%Y%m%d%H%M%S'))}.csv"))

    # 获取ID
    def get_json(self, json_name: str):
        json_path = os.path.join(os.getcwd(), f'{json_name}.json')

        if os.path.exists(json_path):
            with open(json_path, mode="r", encoding="UTF-8") as f:
                data = json.load(f)
        else:
            data = collections.defaultdict(list)
        return data

    # 存储ID
    def save_json(self, json_name: str, data_name: str, data_save: Union[List[str], str]):
        json_path = os.path.join(os.getcwd(), f'{json_name + self.file_name}.json')

        data = self.get_json(json_name + self.file_name)
        with open(json_path, mode="w+", encoding="UTF-8") as f:
            data_new = [f"{data_save}"] if not isinstance(data_save, list) else data_save
            data[data_name] = list(set(data.get(data_name, []) + data_new))
            json.dump(data,
                      f,
                      indent=4,
                      ensure_ascii=False)

    def errorInfo(self, data):
        print(data)

    def multiProcess(self, Name: str):
        self.file_name = Name

        P = mp.Pool(self.CPU)
        for func_name, func in self.MethodMapping[self.file_name].items():
            # self.process_data(func_name, func)
            P.apply_async(func=self.process_data, args=(func_name, func,), error_callback=self.errorInfo)
        P.close()
        P.join()

    def singleProcess(self, Name: str):
        self.file_name = Name

        for func_name, func in self.MethodMapping[self.file_name].items():
            self.process_data(func_name, func)

    def process_data(self, func_name: str, func: Callable):
        filePathList = self.get_file_path_list()
        data_set, res_list, ID_checked, flag, process = [], [], [], 0, 0
        length = len(filePathList)

        for file_path in filePathList:
            T = 0
            _, file_sub = os.path.split(file_path)
            # if process_file != '2017-03-30.csv':
            #     continue
            try:
                data_sub = pd.read_csv(file_path)
            except Exception as e:
                print(f"{func_name:<15}{file_sub}:{e}")
            else:
                sta = time.time()
                res_df = self.callfunc(data_sub, file_path[-14: -4], func_name, func)
                T = time.time() - sta
                res_list.append(res_df)
                flag += 1
            finally:
                ID_checked.append(file_sub)
                process += 1
                print(f"{func_name:<15}{file_sub}:{process:>4}/{length}-{round(process / length * 100, 2)}%, "
                      f"Consuming:{round(T, 2):>6}sec")

                if flag >= 100 or process == length:
                    self.to_csv(func_name, res_list)  # 存储结果
                    with self.lock:
                        self.save_json(self.json_name_ID, func_name, ID_checked)  # 存储ID
                    # 重置
                    res_list, flag = [], 0


if __name__ == '__main__':
    path = r'C:\Users\Administrator\Desktop\高频因子计算\DataBase\输出数据\MidData'
    for name in ['Depth', 'Trade']:
        A = MidData({}, path)
        A.multiProcess(name)
