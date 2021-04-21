import os
import time
import pandas as pd
import datetime as dt
import multiprocessing as mp
from itertools import zip_longest
from collections import defaultdict
from typing import Callable, List, Dict, Any, Tuple, Iterable

from utility.utility import searchFunc
from mapping import (
    saveMapping as saveM,
    CPU
)
from HFDMidDataCal.save.saveData import saveData, readJson

parentPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")
folderName = os.path.splitext(os.path.basename(__file__))[0]

calFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}worker{os.sep}{folderName}', 'worker')
# readFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}read', 'read')
switchFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}switch', 'switch')

# 函数类别
funcClass = 'DepthMid2'


# 准备数据
def prepare(filePath: str) -> List[str]:
    files = os.listdir(filePath)
    # 已处理文件
    dataOld = readJson(saveM['Record']['Path'], funcClass)
    filesNew = list(set(files).difference(set(dataOld)))
    return filesNew


def SyntheticDepthMid2_worker(readFunc: Callable,
                              filePath: str,
                              **kwargs):
    if calFuncs != {}:
        files = prepare(filePath)
        ileGroup = list(zip_longest(*[iter(files)] * int(len(files) / CPU)))

        pool = mp.Pool(CPU)
        for group in ileGroup:  # ileGroup
            pool.apply_async(func=onCallFunc, args=(filePath, group, readFunc), error_callback=onErrorInfo)
            # onCallFunc(filePath, group, readFunc)
        pool.close()
        pool.join()
    else:
        print(f"{funcClass} {dt.datetime.now()}: No function to cal!")


def onCallFunc(filePath: str, fileNames: Iterable, readFunc: Callable):

    # 进程名
    pid = os.getpid()

    flag = 0
    resDict = defaultdict(list)  # 每日数据容器

    for fileSub in fileNames:
        flag += 1
        if fileSub is None:
            continue
        sta = time.time()
        date = fileSub.split('.')[0]
        # 1.Read
        try:
            data = readFunc(filePath, fileSub)
        except Exception as e:
            print(f"Read file error: {fileSub}, {e}")
            saveData(DBName=saveM['TxT']['DBName'],
                     position=saveM['TxT']['Path'],
                     data=f"Read file error {dt.datetime.now()}: {fileSub}, {e}",
                     fileName=f"{funcClass}Error")
            continue      # 读取出错后跳出该次循环
        else:
            # 2.Cal
            for funcName, func in calFuncs.items():

                try:
                    calRes = func(data=data.copy(), date=date)
                except Exception as e:
                    print(f"{funcName} error: {fileSub}, {e}")
                    saveData(DBName=saveM['TxT']['DBName'],
                             position=saveM['TxT']['Path'],
                             data=f"{funcName} error {dt.datetime.now()}: {fileSub}, {e}",
                             fileName=f"{funcClass}Error")
                    continue  # 函数计算出错后跳出该次循环
                # None值不存储
                if calRes is not None:
                    resDict[funcName].append(calRes)

        # 3.Save
        for resName, resValue in resDict.items():  # 对子函数返回值进行格式转化
            resSwitch = switchFuncs[resName](resValue)

            if resSwitch is not None:  # 子函数无返回值不进行存储操作
                saveData(DBName=saveM[resName]['DBName'],
                         position=saveM[resName]['Path'],
                         data=resSwitch,
                         fileName=f"{resName}-{date}")

        # 记录已处理数据
        with mp.Lock():
            saveData(DBName=saveM['Record']['DBName'],
                     position=saveM['Record']['Path'],
                     data=fileSub,
                     fileName=funcClass)

        print(f"{funcClass} {pid:<5}:{date} {flag:>4}/{len(fileNames):>3}, 耗时{round(time.time() - sta, 4)}")


def onErrorInfo(info: Any):
    print(info)
