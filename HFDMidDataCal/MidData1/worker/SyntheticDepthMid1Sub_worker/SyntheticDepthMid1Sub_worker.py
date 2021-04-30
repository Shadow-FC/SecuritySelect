import os
import time
import warnings
import numpy as np
import pandas as pd
import datetime as dt
import pickle5 as pickle
import multiprocessing as mp
from itertools import zip_longest
from collections import defaultdict
from typing import Callable, List, Dict, Any, Tuple, Iterable

from utility.utility import searchFunc, stockCode
from mapping import (
    saveMapping as saveM,
    CPU
)
from HFDMidDataCal.save.saveData import saveData, readJson

warnings.filterwarnings('ignore')

parentPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
folderName = os.path.splitext(os.path.basename(__file__))[0]

calFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}worker{os.sep}{folderName}', 'worker')
readFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}read', 'read')
switchFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}switch', 'switch')


# 函数类别
funcClass = 'DepthMid1Sub'

DepthPath = r"D:\Test\十档\Stk_Tick10_{year}{month}\{year}{month}{day}\{exchange}{codeID}.pkl"


# 准备数据
def prepare(filePath: str) -> List[Tuple[str, str]]:
    with open(filePath, 'rb') as f:
        UpLimitStock = pickle.load(f)
    data = (UpLimitStock['date'] + '_' + UpLimitStock['stock_id']).to_list()

    # 已处理文件
    dataOld = readJson(saveM['Record']['Path'], funcClass)

    # 数据参数
    UpLimitStock_ID = list(set(data).difference(set(dataOld)))

    return UpLimitStock_ID


def SyntheticDepthMid1Sub_worker(readFunc: Callable,
                                 filePath: str,
                                 **kwargs):
    if calFuncs != {}:
        dataParams = prepare(filePath)
        ileGroup = list(zip_longest(*[iter(dataParams)] * int(np.ceil(len(dataParams) / CPU))))

        pool = mp.Pool(CPU)
        for group in ileGroup:  # [[r'Y:\十档\Stk_Tick10_201309\20130910']]
            pool.apply_async(func=onCallFunc, args=(group, readFunc), error_callback=onErrorInfo)
            # onCallFunc(group, readFunc)
        pool.close()
        pool.join()
    else:
        print(f"{funcClass} {dt.datetime.now()}: No function to cal!")


def onCallFunc(dataParams: Iterable, readFunc: Callable):
    # 进程名
    dataParams = list(filter(None, dataParams))
    pid = os.getpid()

    flag = 0
    resDict = defaultdict(list)  # 数据容器
    sta = time.time()
    # 循环处理
    for sample in dataParams:
        date, code = sample.split('_')
        flag += 1
        year, month, day = date.split('-')
        codeID, exchange = code.split('.')

        depthPath = DepthPath.format(year=year,
                                     month=month,
                                     day=day,
                                     exchange=exchange,
                                     codeID=codeID)

        # 1.Read
        try:
            data = readFunc(depthPath)
        except Exception as e:
            print(f"Read file error: {date}, {code}, {e}")
            saveData(DBName=saveM['TxT']['DBName'],
                     position=saveM['TxT']['Path'],
                     data=f"Read file error {dt.datetime.now()}: {date}, {code}, {e}",
                     fileName=f"{funcClass}Error")
            continue  # 读取出错后跳出该次循环
        else:
            # 2.Cal
            for funcName, func in calFuncs.items():
                try:
                    calRes = func(data=data.copy(), code=code, date=date)
                except Exception as e:
                    print(f"{funcName} error: {date}, {code}, {e}")
                    saveData(DBName=saveM['TxT']['DBName'],
                             position=saveM['TxT']['Path'],
                             data=f"{funcName} error {dt.datetime.now()}: {date}, {code}, {e}",
                             fileName=f"{funcClass}Error")
                    continue  # 函数计算出错后跳出该次循环
                # None值不存储
                if calRes is not None:
                    resDict[funcName].append(calRes)
            resDict['ID'].append(sample)

        if flag >= 1000 or flag == len(dataParams):

            # 3.Save
            for resName, resValue in resDict.items():  # 对子函数返回值进行格式转化
                if resName == 'ID':
                    continue
                resSwitch = switchFuncs[resName](resValue)

                if resSwitch is not None:  # 子函数无返回值不进行存储操作
                    saveData(DBName=saveM[resName]['DBName'],
                             position=saveM[resName]['Path'],
                             data=resSwitch,
                             fileName=f"Process-{pid}-{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}")

            # 记录已处理数据
            with mp.Lock():
                saveData(DBName=saveM['Record']['DBName'],
                         position=saveM['Record']['Path'],
                         data=resDict['ID'],
                         fileName=funcClass)
            resDict['ID'].clear()
            print(f"{funcClass} {pid:<5}: {flag:>3}/{len(dataParams):>3}, 耗时{round(time.time() - sta, 4)}")


def onErrorInfo(info: Any):
    print(info)
