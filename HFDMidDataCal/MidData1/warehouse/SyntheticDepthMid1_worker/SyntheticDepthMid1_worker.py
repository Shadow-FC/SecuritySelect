import os
import time
import warnings
import numpy as np
import pandas as pd
import datetime as dt
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

# 有效股票ID
effectID = stockCode()

# 函数类别
funcClass = 'DepthMid1'


# 准备数据
def prepare(filePath: str) -> List[str]:
    folderFiles = os.listdir(filePath)
    # 已处理文件
    dataOld = readJson(saveM['Record']['Path'], funcClass)
    # 过滤文件夹
    folderFiles = [file_ for file_ in folderFiles if len(file_.split('.')) == 1]

    # 生成文件夹
    subFolderPath = []
    for subFolder in folderFiles:
        subFolders2 = os.listdir(os.path.join(filePath, subFolder))
        # 过滤已处理文件
        subFoldersE = list(set(subFolders2).difference(set(dataOld)))
        subFolderPath += [os.path.join(os.path.join(filePath, subFolder), x) for x in subFoldersE]

    return subFolderPath


def SyntheticDepthMid1_worker(readFunc: Callable,
                              filePath: str,
                              **kwargs):
    if calFuncs != {}:
        subFolderPath = prepare(filePath)
        ileGroup = list(zip_longest(*[iter(subFolderPath)] * int(np.ceil(len(subFolderPath) / CPU))))

        pool = mp.Pool(CPU)
        for group in ileGroup:  # [[r'Y:\十档\Stk_Tick10_201309\20130910']]
            pool.apply_async(func=onCallFunc, args=(group, readFunc), error_callback=onErrorInfo)
            # onCallFunc(group, readFunc)
        pool.close()
        pool.join()
    else:
        print(f"{funcClass} {dt.datetime.now()}: No function to cal!")


def onCallFunc(filePathList: Iterable, readFunc: Callable):
    # 进程名
    pid = os.getpid()
    flag = 0
    for folderPathSub in filePathList:
        flag += 1
        if folderPathSub is None:
            continue

        resDict = defaultdict(list)  # 每日数据容器
        folder_date = os.path.split(folderPathSub)[-1]
        date = folder_date[:4] + '-' + folder_date[4:6] + '-' + folder_date[-2:]  # 日期

        fileNames = os.listdir(folderPathSub)
        sta = time.time()
        # 循环处理
        for fileSub in fileNames:

            code = f"{fileSub[2:-4]}.{fileSub[:2].upper()}"

            if code not in effectID:  # 剔除非股票数据
                continue

            # if fileSub not in ["sz000010.csv", "sz000010.csv", "sh600000.csv"]:
            #     continue
            # 1.Read
            try:
                data = readFunc(folderPathSub, fileSub)
            except Exception as e:
                print(f"Read file error: {date}, {fileSub}, {e}")
                saveData(DBName=saveM['TxT']['DBName'],
                         position=saveM['TxT']['Path'],
                         data=f"Read file error {dt.datetime.now()}: {date}, {fileSub}, {e}",
                         fileName=f"{funcClass}Error")
                continue  # 读取出错后跳出该次循环
            else:
                # 2.Cal
                for funcName, func in calFuncs.items():

                    try:
                        calRes = func(data=data.copy(), code=code, date=date)
                    except Exception as e:
                        print(f"{funcName} error: {date}, {fileSub}, {e}")
                        saveData(DBName=saveM['TxT']['DBName'],
                                 position=saveM['TxT']['Path'],
                                 data=f"{funcName} error {dt.datetime.now()}: {date}, {fileSub}, {e}",
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
                         fileName=date)

        # 记录已处理数据
        with mp.Lock():
            saveData(DBName=saveM['Record']['DBName'],
                     position=saveM['Record']['Path'],
                     data=folder_date,
                     fileName=funcClass)

        print(f"{funcClass} {pid:<5}:{date} {flag:>4}/{len(filePathList):>3}, 耗时{round(time.time() - sta, 4)}")


def onErrorInfo(info: Any):
    print(info)
