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

#
# def cal(fact_list, pamas):
#     for fac in fact_list:
#         factor_dict = {"factor_name": fac,
#                        "factor_params": pamas,
#                        'factor': None,
#                        'cal': True
#                        }
#         main1(factor_dict)


# def cal_pa(fact_list, pa):
#     for fac in fact_list:
#         for p_ in pa:
#             factor_dict = {"factor_name": fac,
#                            "factor_params": {"period": p_,
#                                              "minute": 1,
#                                              "n": 1},
#                            'factor': None,
#                            'cal': True
#                            }
#             print(p_)
#             main1(factor_dict)


# def main_M():
#     factor_name = [f"Distribution{i:03}" for i in [1, 30, 31] + list(range(4, 10)) + list(range(11, 28)) + [28, 29]] + \
#                   [f"FundFlow{i:03}" for i in
#                    list(range(1, 7)) + list(range(9, 14)) + list(range(25, 30)) + [32, 35, 39, 40]] + \
#                   [f"VolPrice{i:03}" for i in list(range(11, 18)) + [8, 9]]
#
#     factorPa = {"minute": 1,
#                 "method": 'mid',
#                 "R": 0.1,
#                 "q": 0.2,
#                 "n": 1}
#
#     for N in factor_name:
#         if N in ['Distribution028', 'Distribution029', 'Distribution030', 'Distribution031',
#                  'FundFlow039', 'VolPrice017', 'VolPrice018', 'VolPrice019', 'VolPrice020']:
#             continue
#         if N in ['Distribution028', 'Distribution029', "FundFlow039", "FundFlow040"]:
#             factorPa['n'] = 20
#         elif N in ["FundFlow032"]:
#             factorPa['n'] = 21
#         else:
#             factorPa['n'] = 1
#         # if N not in ['Distribution018', 'Distribution025', 'Distribution028', 'Distribution030'] + \
#         #         ['FundFlow001', 'FundFlow006', 'FundFlow013', 'FundFlow028', 'FundFlow033', 'FundFlow034',
#         #          'FundFlow039'] + \
#         #         ['VolPrice012', 'VolPrice013', 'VolPrice017']:
#         #     continue
#         # if N.startswith('Distribution'):
#         #     continue
#         factor_dict = {"fact_name": N,
#                        "fact_params": factorPa,
#                        'fact_value': None
#                        }
#         try:
#             cal_factor(factor_dict)
#         except Exception as e:
#             print(f"{N}: {e}")
#     # pool = Pool(processes=4)
#     # for key_, value_ in fac_dict.items():
#     #     pool.apply_async(cal, (value_, pam))
#     # # pool.apply_async(cal_pa, (dd, pam))
#     # pool.close()
#     # pool.join()


# def single():
#     for factor in [f"FundFlow{i:03}" for i in [46, 47, 48]] + [f"VolPrice{i:03}" for i in [18, 19, 20]]:
#         if factor.startswith('FundFlow'):
#             for s in ['all', 'open', 'close', 'between']:  # ['all', 'open', 'close', 'between']
#                 fact_dict = {
#                     "factor_name": factor,
#                     "factor_params": {"minute": 1,
#                                       "x_min": s,
#                                       "method": 'mid',
#                                       "period": s,
#                                       "depth": 5,
#                                       "ratio": 0.1,
#                                       "q": 0.2,
#                                       "n": 1},
#                     'cal': True
#                 }
#                 cal_factor(fact_dict)
#         if factor.startswith("VolPrice"):
#
#             for s in [5, 10]:
#                 fact_dict = {
#                     "fact_name": factor,
#                     "fact_params": {"minute": 1,
#                                     "x_min": s,
#                                     "method": 'mid',
#                                     "period": s,
#                                     "depth": s,
#                                     "ratio": 0.1,
#                                     "q": 0.2,
#                                     "n": 1},
#                     'fact_value': None
#                 }
#                 cal_factor(fact_dict)


def singleCal(factName: str = None):
    Factor = {"fact_name": factName,
              "fact_params": {"minute": 1,
                              "method": 'mid',
                              "R": 0.1,
                              "q": 0.2,
                              "n": 1},
              "fact_value": None
              }

    cal_factor(Factor)


def multipleCal():
    # factor_names = [f"Distribution{i:03}" for i in [1, 30, 31] + list(range(4, 10)) + list(range(11, 28)) + [28, 29]] + \
    #               [f"FundFlow{i:03}" for i in
    #                list(range(1, 7)) + list(range(9, 14)) + list(range(25, 30)) + [32, 35, 39, 40]] + \
    #               [f"VolPrice{i:03}" for i in list(range(11, 18)) + [8, 9]]
    #
    # factorPa = {"minute": 1,
    #             "method": 'mid',
    #             "R": 0.1,
    #             "q": 0.2,
    #             "n": 1}
    #
    # for N in factor_names:
    #     if N in ['Distribution028', 'Distribution029', 'Distribution030', 'Distribution031',
    #              'FundFlow039', 'VolPrice017', 'VolPrice018', 'VolPrice019', 'VolPrice020']:
    #         continue
    #     if N in ['Distribution028', 'Distribution029', "FundFlow039", "FundFlow040"]:
    #         factorPa['n'] = 20
    #     elif N in ["FundFlow032"]:
    #         factorPa['n'] = 21
    #     else:
    #         factorPa['n'] = 1
    #
    #     factor_dict = {"fact_name": N,
    #                    "fact_params": factorPa,
    #                    'fact_value': None
    #                    }
    #     try:
    #         cal_factor(factor_dict)
    #     except Exception as e:
    #         print(f"{N}: {e}")

    # factor_names = [f"FundFlow{i:03}" for i in [18, 19, 20]] + [f"VolPrice{i:03}" for i in [18, 19, 20]]
    factor_names = [f"FundFlow{i:03}" for i in [33, 34]]
    for factor in factor_names:
        if factor.startswith('FundFlow'):
            for s in ['5', '10', '15', '30', '60']:
                fact_dict = {
                    "fact_name": factor,
                    "fact_params": {"minute": 1,
                                    "x_min": s,
                                    "method": 'mid',
                                    "period": '',
                                    "depth": 5,
                                    "ratio": 0.1,
                                    "q": 0.2,
                                    "n": 1},
                    'fact_value': None
                }
                cal_factor(fact_dict)
        if factor.startswith("VolPrice"):

            for s in [5, 10]:
                fact_dict = {
                    "fact_name": factor,
                    "fact_params": {"minute": 1,
                                    "x_min": s,
                                    "method": 'mid',
                                    "period": s,
                                    "depth": s,
                                    "ratio": 0.1,
                                    "q": 0.2,
                                    "n": 1},
                    'fact_value': None
                }
                cal_factor(fact_dict)


if __name__ == '__main__':
    # single()
    # main()
    for i in ['Strength001', 'HighFreq002', 'HighFreq003', 'HighFreq004']:
        singleCal(factName=i)
    # multipleCal()
