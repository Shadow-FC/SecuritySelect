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
    SpecialName as SN,
    DBName as DBN
)

from mapping import readMapping as readM

REPLACE = "(_read)|(_worker)|(_switch)|(.py)"


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
    file_path = os.path.join(FPN.FactorDataSet.value, fact.data_category)

    if not os.path.exists(file_path):
        os.makedirs(file_path)

    data_path = os.path.join(file_path, fact.data_name + '.pkl')
    fact.data.to_pickle(data_path)


# 方法搜寻函数
def searchFunc(pathIn: str,
               folderName: str) -> Dict[str, Callable]:
    files = os.listdir(pathIn)

    subPath = pathIn.split(f'SecuritySelect{os.sep}')[-1].replace(os.sep, '.')
    mudPath = os.path.split(pathIn)[-1]

    funcDict = {}

    for file in files:
        if file in [mudPath.split('.')[-1] + '.py', '__init__.py']:  # 母函数脚本无需再次导入
            continue

        # 对文件夹进行判断
        if os.path.isdir(os.path.join(pathIn, file)):
            if file.startswith('Synthetic'):       # 母函数文件夹则导入母函数
                mud = importlib.import_module(f'{subPath}.{file}.{file}')
                funcDict[re.sub(REPLACE, '', file)] = getattr(mud, file)
            else:                                  # 非母函数文件夹导入文件夹下py文件
                fileSubs = os.listdir(os.path.join(pathIn, file))
                for fileSub in fileSubs:
                    if fileSub.endswith('.py') and fileSub not in ['__init__.py']:
                        mud = importlib.import_module(f'{subPath}.{file}.{fileSub[:-3]}')
                        funcDict[re.sub(REPLACE, '', fileSub)] = getattr(mud, fileSub[:-3])
        else:
            if file.endswith(f'{folderName}.py'):  # 如果是_{name}.py脚本，加载脚本中的子函数
                mud = importlib.import_module(f'{subPath}.{file[:-3]}')
                funcDict[re.sub(REPLACE, '', file)] = getattr(mud, file[:-3])

    return funcDict


# 股票代码转换
def switchCode(code: Union[str, int]) -> str:
    codeStr = str(code)
    if codeStr[0] in ['6', '9']:
        return code + '.SH'
    elif codeStr[0] in ['0', '2', '3']:
        return code + '.SZ'
    else:
        return code


# 有效股票代码
def stockCode() -> List[str]:
    data = pd.read_csv(r'Y:\DataBase\AStockData.csv', usecols=['code'])
    stockID = data.drop_duplicates()['code'].to_list()
    return stockID


# 额外数据调取
def getData(filePath: str, fileType: DBN) -> Union[Any, None]:
    if fileType == DBN.CSV:
        res = pd.read_csv(filePath)
    elif fileType == DBN.PKL:
        with open(filePath, mode='rb') as f:
            res = pickle.load(f)
    else:
        res = None
    return res

