# -*-coding:utf-8-*-
# @Time:   2021/3/26 16:42
# @Author: FC
# @Email:  18817289038@163.com

import os
import datetime as dt
import pandas as pd

pathZIP = r"Z:\冯晨\十档备份\Stk_Tick10_201310_NEW\Stk_Tick10_201310"
pathO = r'Y:\十档\Stk_Tick10_201310'


def to_csv(data: pd.Series):
    header = False if os.path.exists(os.path.join('Z:\\冯晨\\十档备份', 'result.csv')) else True
    data.to_frame(str(flag)).T.to_csv(os.path.join('Z:\\冯晨\\十档备份', 'result.csv'), mode='a+', header=header)


flag = 0
folders = os.listdir(pathZIP)
for folder in folders:
    print(f"{dt.datetime.now()}-{folder}")
    filePath = pathRead = os.path.join(pathZIP, folder)
    files = os.listdir(filePath)
    for fileSub in files:
        flag += 1
        try:
            dfOld = pd.read_csv(os.path.join(filePath, fileSub), encoding='GBK')
            dfNew = pd.read_csv(os.path.join(os.path.join(pathO, folder), fileSub), encoding='GBK')
        except Exception as e:
            res = pd.Series([folder, fileSub, '-', '-', e],
                            index=['folderName', 'fileName', 'OLD', 'NEW', 'judge'])
        else:
            # 比较
            Judge = dfOld == dfNew
            if Judge.all().all():
                res = pd.Series([folder, fileSub, dfOld.shape, dfNew.shape, 'T'],
                                index=['folderName', 'fileName', 'OLD', 'NEW', 'judge'])
            else:
                if dfOld[~Judge].isna().all().all():
                    res = pd.Series([folder, fileSub, dfOld.shape, dfNew.shape, 'NaN'],
                                    index=['folderName', 'fileName', 'OLD', 'NEW', 'judge'])
                else:
                    res = pd.Series([folder, fileSub, dfOld.shape, dfNew.shape, 'F'],
                                    index=['folderName', 'fileName', 'OLD', 'NEW', 'judge'])
        finally:
            to_csv(res)
print('s')
