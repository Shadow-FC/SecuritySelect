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

from utility.utility import searchFunc, switchCode, stockCode
from mapping import (
    saveMapping as saveM,
    CPU
)
from HFDMidDataCal.save.saveData import saveData, readJson

warnings.filterwarnings('ignore')

parentPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep+".")
folderName = os.path.splitext(os.path.basename(__file__))[0]

calFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}worker{os.sep}{folderName}', 'worker')
readFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}read', 'read')
switchFuncs = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}switch', 'switch')

# 有效股票ID
effectID = stockCode()

# 函数类别
funcClass = 'TradeMid1'


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


def SyntheticTradeMid1_worker(readFunc: Callable,
                              filePath: str,
                              **kwargs):
    if calFuncs != {}:

        subFolderPath = prepare(filePath)
        ileGroup = list(zip_longest(*[iter(subFolderPath)] * int(np.ceil(len(subFolderPath) / CPU))))
        sta = time.time()
        pool = mp.Pool(CPU)
        for group in ileGroup:  # ileGroup
            # sta = time.time()
            pool.apply_async(func=onCallFunc, args=(group, readFunc), error_callback=onErrorInfo)
            # onCallFunc(group, readFunc)
            # end = time.time() - sta
            # print(f"ALL:{end}")
        pool.close()
        pool.join()
        end = time.time() - sta
        print(end)
    else:
        print(f"{funcClass} {dt.datetime.now()}: No function to cal!")


def onCallFunc(filePathList: Iterable, readFunc: Callable):
    Time = defaultdict(list)
    # 进程名
    pid = os.getpid()
    flag = 0
    for folderPathSub in filePathList:
        flag += 1
        if folderPathSub is None:
            continue

        resDict = defaultdict(list)  # 每日数据容器
        date = os.path.split(folderPathSub)[-1]
        fileNames = os.listdir(folderPathSub)

        sta = time.time()
        # 循环处理
        for fileSub in fileNames:
            print(f"{pid}-{fileSub}")
            code = switchCode(fileSub[:-4])

            if code not in effectID:  # 剔除非股票数据
                continue
            # if code != '688321.SH':
            #     continue
            try:
                data = readFunc(folderPathSub, fileSub)
            except Exception as e:
                print(f"Read file error: {date}, {fileSub}, {e}")
                saveData(DBName=saveM['TxT']['DBName'],
                         position=saveM['TxT']['Path'],
                         data=f"Read file error {dt.datetime.now()}: {date}, {fileSub}, {e}",
                         fileName=f"{funcClass}Error")
                continue     # 读取出错后跳出该次循环
            else:
                for funcName, func in calFuncs.items():

                    try:
                        staC = time.time()
                        calRes = func(data=data.copy(), code=code, date=date)
                        endC = time.time() - staC
                        Time[funcName].append(endC)
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
                     data=date,
                     fileName=funcClass)

        print(f"{funcClass} {pid:<5}:{date} {flag:>4}/{len(filePathList):>3}, 耗时{round(time.time() - sta, 4)}")
        dataRes = pd.DataFrame(Time)
        res = dataRes.describe()
        dataRes.to_csv(f'C:\\Users\\Administrator\\Desktop\\Test\\{pid}.csv')
        res.to_csv(f'C:\\Users\\Administrator\\Desktop\\Test\\{pid}_res.csv')


def onErrorInfo(info: Any):
    print(info)
