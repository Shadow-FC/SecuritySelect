# -*-coding:utf-8-*-
# @Time:   2021/3/9 13:01
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
from typing import List, Callable, Iterable, Dict, Union
from itertools import zip_longest
import multiprocessing as mp

from HFDMidData.HFDMidData1.MidData1 import HFDTrade, HFDDepth, HFDBase
from HFDMidData.HFDMidData2.MidData2Copy import MidData

# 原始数据
RawDataPath = {
    "depthPath": r'Y:\十档',
    "callPath": r'Y:\集合竞价',
    "tradePath": r'Y:\逐笔全息'
}

# 数据存储地址
SavePath = {
    "depth1min": r'D:\补充数据\十档1min',
    "depthVwap": r'D:\补充数据\十档Vwap',

    "trade1min": r'D:\补充数据\逐笔1min',
    "tradeCashFlow": r'D:\补充数据\逐笔资金流',
    "factorMid": r'D:\补充数据\因子中间过程'
}


class CalMidData(object):
    CPU = mp.cpu_count()

    def __init__(self):
        self.lock = mp.Manager().Lock()

        self.onProcess: Dict[str, Union[HFDBase, object]] = {
            "Depth": None,
            "Trade": None,
            "MIDDepth": None,
            "MIDTrade": None
        }

        self.init()

    def init(self):
        self.onProcess['Depth'] = HFDDepth(RawDataPath['depthPath'], SavePath)
        self.onProcess['Trade'] = HFDTrade(RawDataPath['tradePath'], SavePath)
        self.onProcess['MIDDepth'] = MidData(SavePath['depth1min'], SavePath['factorMid'])
        self.onProcess['MIDTrade'] = MidData(SavePath['trade1min'], SavePath['factorMid'])

    def cal(self, Name: str):

        # 计算中间过程1
        # self.onProcess[Name].singleProcess()

        # 计算中间过程2
        # self.onProcess[f'MID{Name}'].singleProcess(Name)

        # 计算中间过程1
        self.onProcess[Name].CPU = 2
        self.onProcess[Name].multiProcess()

        # 计算中间过程2
        self.onProcess[f'MID{Name}'].CPU = 2
        self.onProcess[f'MID{Name}'].multiProcess(Name)


if __name__ == '__main__':
    A = CalMidData()
    for name in ['Trade']:
        A.cal(name)
