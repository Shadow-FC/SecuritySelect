# -*-coding:utf-8-*-
# @Time:   2021/3/25 15:58
# @Author: FC
# @Email:  18817289038@163.com

import time
import numpy as np
import os
import pandas as pd
import pickle5 as pickle


# os.listdir()
# pathRead = r'Y:\十档\Stk_Tick10_202012'
# pathOut = r'B:\Test'
#
#
# # 记录
# def to_csv(data: pd.Series):
#     header = False if os.path.exists(os.path.join('B:\\', 'result.csv')) else True
#     data.to_frame(str(flag)).T.to_csv(os.path.join('B:\\', 'result.csv'), mode='a+', header=header)
#
#
# """
# DataInput: [foldeName, fileSub, O_Shape, N_Shape, Error]
# """
#
# folders = os.listdir(pathRead)
# [os.makedirs(os.path.join(pathOut, i)) for i in folders if not os.path.exists(os.path.join(pathOut, i))]
#
# flag = 0
# for folder in folders:
#     print(folder)
#     folderPath = os.path.join(pathRead, folder)
#     files = os.listdir(folderPath)
#     for fileSub in files:
#
#         flag += 1
#         try:
#             df = pd.read_csv(os.path.join(folderPath, fileSub), encoding='GBK')
#         except Exception as e:
#             res = pd.Series([folder, fileSub, '-', '-', e],
#                             index=['folderName', 'fileName', 'csvShape', 'pklShape', 'judge'])
#         else:
#
#             df.to_pickle(os.path.join(os.path.join(pathOut, folder), fileSub[:-4] + '.pkl'))
#
#             # re-read
#             df = pd.read_csv(os.path.join(folderPath, fileSub), encoding='GBK')
#             df_pkl = pd.read_pickle(os.path.join(os.path.join(pathOut, folder), fileSub[:-4] + '.pkl'))
#             # 比较
#             Judge = df == df_pkl
#             if Judge.all().all():
#                 res = pd.Series([folder, fileSub, df.shape, df_pkl.shape, 'T'],
#                                 index=['folderName', 'fileName', 'csvShape', 'pklShape', 'judge'])
#             else:
#                 if df_pkl[~Judge].isna().all().all():
#                     res = pd.Series([folder, fileSub, df.shape, df_pkl.shape, 'NaN'],
#                                     index=['folderName', 'fileName', 'csvShape', 'pklShape', 'judge'])
#                 else:
#                     res = pd.Series([folder, fileSub, df.shape, df_pkl.shape, 'F'],
#                                     index=['folderName', 'fileName', 'csvShape', 'pklShape', 'judge'])
#         finally:
#             to_csv(res)

def t1():
    pathIn = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\TechnicalBehaviorFactor'
    files = os.listdir(pathIn)
    for fileSub in files:
        fileName = fileSub.split('.')[0]
        dataCsv = pd.read_csv(os.path.join(pathIn, fileSub))
        dataCsv.to_pickle(os.path.join(pathIn, fileName + '.pkl'))
    print('s')


def t2():
    pathIn = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet'
    m = []
    folders = os.listdir(pathIn)
    for folder in folders:
        if folder.startswith('High'):
            pathFile = os.path.join(pathIn, folder)
            files = os.listdir(pathFile)
            for fileSub in files:
                with open(os.path.join(pathFile, fileSub), 'rb') as f:
                    data = pickle.load(f)
                    dataSub = data[(data['code'] == '600000.SZ') & (data['date'] == '2020-04-28')]
                    dataSub = dataSub.set_index(['date', 'code'])
                m.append(dataSub)
    res = pd.concat(m).stack()
    print(res)


if __name__ == '__main__':

    t2()
    pathTest = r'B:'
    # files = os.listdir(pathTest)
    m = []
    for _ in range(3000):
        sta = time.time()
        data = pd.read_csv(os.path.join(pathTest, '000001.csv'))
        # data = pd.read_pickle(os.path.join(pathTest, '000001_4.pkl'))
        # with open(os.path.join(pathTest, '000001_5.pkl'), mode='rb') as f:
        #     data = pickle.load(f)
        end = time.time() - sta
        m.append(end)
    print('s')
    path = r'D:\DataBase'
    date = '2020-01-02'
    folders = os.listdir(path)
    d = []
    for folder in folders:
        # if folder not in ['HighFrequencyDistributionFactor']:
        #     continue
        pathSub = os.path.join(path, folder)
        files = os.listdir(pathSub)
        for file in files:
            print(file)
            data = pd.read_csv(os.path.join(pathSub, file))
            dataSub = data[data['date'] == date]
            dataSub = dataSub.set_index(['date', 'code'])
            d.append(dataSub)

        d_out = pd.concat(d, axis=1)
    print(d_out)
