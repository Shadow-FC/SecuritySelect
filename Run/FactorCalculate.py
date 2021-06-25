# -*-coding:utf-8-*-
# @Time:   2020/10/12 14:23
# @Author: FC
# @Email:  18817289038@163.com

from utility.utility import factor_to_pkl
from Analysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

API = LoadData()


# 因子计算存储
def cal_factor(params_dict: dict):
    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {params_dict['fact_name']}\033[0m")
    factData = API.getFactorData(**params_dict)
    factor_to_pkl(factData)


def singleCal(factName: str = None, factP: Dict[str, Any] = {}):
    fact_params = {"minute": 1,
                   "R": 0.1,
                   "q": 0.2,
                   "n": 1}
    fact_params.update(factP)
    Factor = {"fact_name": factName,
              "fact_params": fact_params,
              "fact_value": None
              }

    cal_factor(Factor)


def multipleCal():
    factor_names = [f"Distribution{i:03}" for i in [1, 30, 31] + list(range(4, 10)) + list(range(11, 28)) + [28, 29]] + \
                   [f"FundFlow{i:03}" for i in
                    list(range(1, 7)) + [18, 19, 20, 25, 26, 29, 32, 33, 34, 35, 39, 40, 46]] + \
                   [f"VolPrice{i:03}" for i in list(range(11, 21)) + [8, 9]]

    for N in factor_names:
        factorPa = {"minute": 1,
                    "R": 0.1,
                    "q": 0.2,
                    "n": 1}

        if N in ['Distribution028', 'Distribution029',
                 'FundFlow032', 'FundFlow039', 'FundFlow040',
                 'VolPrice018', 'VolPrice019', 'VolPrice020']:
            continue

        if N in ['Distribution028', 'Distribution029', "FundFlow039", "FundFlow040"]:
            factorPa['n'] = 20
        elif N in ["FundFlow032"]:
            factorPa['n'] = 21
        else:
            factorPa['n'] = 1

        if N in ['FundFlow033', 'FundFlow034']:
            for s in ['5', '10', '15', '30', '60']:
                factorPa['x_min'] = s
                fact_dict = {
                    "fact_name": N,
                    "fact_params": factorPa,
                    'fact_value': None
                }
                cal_factor(fact_dict)

        elif N in ['FundFlow018', 'FundFlow019', 'FundFlow020', 'FundFlow046']:
            for s in ['all', 'between', 'close', 'open']:
                factorPa['period'] = s
                fact_dict = {
                    "fact_name": N,
                    "fact_params": factorPa,
                    'fact_value': None
                }
                cal_factor(fact_dict)

        elif N in ['VolPrice018', 'VolPrice019']:
            for s in [5, 10]:
                factorPa['depth'] = s
                fact_dict = {
                    "fact_name": N,
                    "fact_params": factorPa,
                    'fact_value': None
                }
                cal_factor(fact_dict)
        else:
            factor_dict = {"fact_name": N,
                           "fact_params": factorPa,
                           'fact_value': None
                           }
            cal_factor(factor_dict)


if __name__ == '__main__':
    factName = 'VolPrice020'
    factP = {"n": 1}

    singleCal(factName, factP)
    # for i in ['Strength001', 'HighFreq002', 'HighFreq003', 'HighFreq004']:
    #     singleCal(factName=i)
    # multipleCal()
