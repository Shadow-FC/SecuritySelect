# -*-coding:utf-8-*-
# @Time:   2021/4/26 16:55
# @Author: FC
# @Email:  18817289038@163.com


import os
import pandas as pd
import pickle5 as pickle

from functools import reduce
from typing import Dict, List, Union

from DataAPI.LoadData import LoadData
from Analysis.ReturnForecast import ReturnForecast as RetF
# from Analysis.RiskForecast import RiskForecast as RiskF

from constant import (
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    PriceVolumeName as PVN
)

API = LoadData()

effectFacts = ['Distribution004_1min_1days', 'Distribution006_1min_1days', 'Distribution007_1min_1days']

# 因子路径
def factParma(pathExp: str) -> Dict[str, str]:
    # 因子暴露路径
    expFolders = os.listdir(pathExp)
    res = {}
    for folder in expFolders:
        subPath = os.path.join(pathExp, folder)
        factFiles = os.listdir(subPath)
        for factName in factFiles:
            res[factName.split('.')[0]] = os.path.join(subPath, factName)
    return res


def getData(factPathDict: Dict[str, str]) -> Dict[str, Union[pd.DataFrame, pd.Series]]:
    dataSets = {}
    # 获取因子暴露数据
    # factExp = []
    # for factName in effectFacts:
    #     with open(factPathDict[factName], 'rb') as f:
    #         dataD = pickle.load(f).set_index(['date', 'code'])
    #         if not factName.startswith('Syn'):
    #             # 原始因子需要滚动N日取平均
    #             dataD = dataD.groupby('code').apply(lambda x: x.rolling(5, min_periods=1).mean())
    #         factExp.append(dataD)
    # dataSets['factExp'] = pd.concat(exps, axis=1)

    LabelData = API.getLabelPoolData('portfolioLabel')

    # 获取其他数据

    # m = A.get_factor()
    parameters = {"fac_exp": input_data['fac_exp'],  # 因子暴露
                  "stock_ret": input_data['stock_ret'].iloc[:, 0],  # 个股收益
                  "ind_exp": input_data['ind_exp'].iloc[:, 0],  # 行业暴露
                  "mv": input_data['mv'].iloc[:, 0],  # 个股市值

                  "ind_mv": input_data['ind_mv'].iloc[:, 0],  # 基准行业市值
                  "ind_weight": input_data['ind_weight'].iloc[:, 0],
                  "hp": 20
                  }
    return dataSets


def Forecast():
    factExp = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet'
    factDict = factParma(factExp)
    InputData = getData(factDict)
    Ret = RetF()
    # Risk = RiskF()
    pass


if __name__ == '__main__':
    Forecast()
    pass
