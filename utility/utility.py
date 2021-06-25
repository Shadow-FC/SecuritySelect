# -*-coding:utf-8-*-
# @Time:   2021/3/18 14:39
# @Author: FC
# @Email:  18817289038@163.com

import os
import re
import time
import importlib
import pandas as pd
import datetime as dt
import pickle5 as pickle
from typing import Callable, Dict, List, Tuple, Union, Any

from Object import (
    DataInfo
)
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


# 因子计算封装类
class Process(object):
    def __init__(self,
                 funcType: str = ""):
        self.funcType = funcType

    def __call__(self, func):
        def inner(*args, **kwargs):
            func_name = func.__name__
            data = kwargs['data'].set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
            kwargs['data'] = data.sort_index()

            res = func(*args, **kwargs, name=func_name)
            F = DataInfo(data=res['data'],
                         data_name=res['name'],
                         data_type=self.funcType,
                         data_category=func.__str__().split(" ")[1].split('.')[0])
            return F

        return inner


def timer(func):
    def wrapper(*args, **kwargs):
        func_name = func.__name__

        sta = time.time()

        res = func(*args, **kwargs)

        rang_time = round((time.time() - sta) / 60, 4)

        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: It takes\033[0m "
              f"\033[1;33m{rang_time:<6}Min\033[0m "
              f"\033[1;31mto run func\033[0m "
              f"\033[1;33m\'{func_name}\'\033[0m")
        return res

    return wrapper


# 因子值入库
@timer
def factor_to_pkl(fact: DataInfo):
    file_path = os.path.join(FPN.Fact_dataSet.value, fact.data_category)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    data_path = os.path.join(file_path, fact.data_name + '.pkl')
    fact.data.to_pickle(data_path)

