# -*-coding:utf-8-*-
# @Time:   2021/4/12 10:50
# @Author: FC
# @Email:  18817289038@163.com

import os
import datetime as dt

from HFDMidDataCal.save.saveData import saveData
from utility.utility import searchFunc
from mapping import (
    saveMapping as saveM,
    readMapping as readM
)

parentPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep+".")

# 函数检索
readFuncsMid1 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}read', 'read')
calFuncsMid1 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}worker', 'worker')
# switchFuncsMid1 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData1{os.sep}switch', 'switch')

readFuncsMid2 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}read', 'read')
calFuncsMid2 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}worker', 'worker')
# switchFuncsMid2 = searchFunc(parentPath + f'{os.sep}HFDMidDataCal{os.sep}MidData2{os.sep}switch', 'switch')


def RunMidData1():

    for funcName, func in calFuncsMid1.items():
        try:
            print(f"\033[1;31m{dt.datetime.now()}:Start calculate {funcName}\033[0m")
            if funcName.startswith('SyntheticTrade'):  # 母函数计算
                calFuncsMid1[funcName](readFuncsMid1[funcName], readM[funcName])
        except Exception as e:
            print(f"RunMidData1 Error: {e}")


def RunMidData2():

    for funcName, func in calFuncsMid2.items():
        try:
            print(f"\033[1;31m{dt.datetime.now()}:Start calculate {funcName}\033[0m")
            if funcName.startswith('Synthetic'):  # 母函数计算
                calFuncsMid2[funcName](readFuncsMid2[funcName], readM[funcName])
        except Exception as e:
            print(f"RunMidData2 Error: {e}")


if __name__ == '__main__':
    RunMidData1()
    # RunMidData2()
