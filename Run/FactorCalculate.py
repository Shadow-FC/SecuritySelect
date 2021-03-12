# -*-coding:utf-8-*-
# @Time:   2020/10/12 14:23
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
import time
from multiprocessing import Pool
from FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}


# 因子计算存储
def cal_factor(params_dict: dict):
    A = FactorValidityCheck()

    factor_name = params_dict['factor_name']
    factor_params = params_dict['factor_params']

    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}\033[0m")

    A.load_factor(fact_name=factor_name,
                  factor_params=factor_params,
                  cal=params_dict['cal'])

    A.factor_to_csv()


def main():
    factors_name = {
        FCN.Val.value: ['EP_ttm', 'EP_LR', 'EP_cut_ttm', 'E2P_ttm', 'PEG_ttm', 'BP_LR', 'BP_ttm', 'SP_ttm',
                        'SP_LR', 'NCFP_ttm', 'OCFP_ttm', 'FCFP_LR', 'FCFP_ttm', 'DP_ttm'],
        FCN.Gro.value: ['BPS_G_LR', 'EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR', 'TA_G_ttm', 'LA_G_LR', 'LA_G_ttm',
                        'ILA_G_LR', 'ILA_G_ttm', 'TA_G_LR_std', 'TA_G_ttm_std', 'LA_G_LR_std', 'LA_G_ttm_std',
                        'ILA_G_LR_std', 'ILA_G_ttm_std', 'NP_Acc', 'NP_Stable', 'NP_SD', 'OP_Acc', 'OP_Stable',
                        'OP_SD', 'OR_Acc', 'OR_Stable', 'OR_SD'],
        FCN.Pro.value: ['ROA_ttm', 'DPR_ttm', 'NP', 'NP_ttm', 'OPM', 'OPM_ttm'],
        FCN.Sol.value: ['Int_to_Asset', 'ShortDebt1_CFPA', 'ShortDebt2_CFPA', 'ShortDebt3_CFPA',
                        'ShortDebt1_CFPA_qoq', 'ShortDebt2_CFPA_qoq', 'ShortDebt3_CFPA_qoq',
                        'ShortDebt1_CFPA_qoq_abs', 'ShortDebt2_CFPA_qoq_abs', 'ShortDebt3_CFPA_qoq_abs',
                        'ShortDebt1_CFPA_std', 'ShortDebt2_CFPA_std', 'ShortDebt3_CFPA_std',
                        'IT_qoq_Z', 'PTCF_qoq_Z', 'OT_qoq_Z', 'OT2NP_qoq_Z', 'PT2NA_Z'],

        FCN.Ope.value: ['RROC_N', 'OCFA', 'TA_Turn_ttm'],
        FCN.EQ.value: ['CSR', 'CSRD', 'APR', 'APRD']}

    for j, j_v in factors_name.items():
        if j in [FCN.Val.value]:
            continue
        print(f"开始计算{j}因子")
        for v_ in j_v:
            # if v_ in ['EPS_G_ttm', 'ROA_G_ttm', 'TA_G_LR']:
            #     continue
            factor_dict = {"factor_category": j,
                           "factor_name": v_,
                           "factor_params": {"switch": False},
                           'factor': None,
                           'cal': True,
                           'save_type': 'raw'  # 保存原始因子数据， switch:保留频率转换后的数据
                           }

            print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_dict['factor_name']}\033[0m")
            db = 'Fin'
            cal_factor(factor_dict)


def main1(fact_dict):
    # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: "
    #       f"{factor_dict['factor_name']}-{factor_dict['factor_params']['n']}\033[0m")
    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {fact_dict['factor_name']}\033[0m")
    cal_factor(fact_dict)

    # factor_dict = {"factor_category": factor_category,
    #                "factor_name": factor,
    #                "factor_params": {},
    #                'factor': None,
    #                'cal': True,
    #                'save_type': 'switch'  # 保存原始因子数据， switch:保留频率转换后的数据
    #                }
    #
    # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_dict['factor_name']}\033[0m")
    # db = 'Fin'
    # cal_factor(factor_dict, db)


def cal(fact_list, pamas):
    for fac in fact_list:
        factor_dict = {"factor_name": fac,
                       "factor_params": pamas,
                       'factor': None,
                       'cal': True
                       }
        main1(factor_dict)


def cal_pa(fact_list, pa):
    for fac in fact_list:
        for p_ in pa:
            factor_dict = {"factor_name": fac,
                           "factor_params": {"period": p_,
                                             "minute": 1,
                                             "n": 1},
                           'factor': None,
                           'cal': True
                           }
            print(p_)
            main1(factor_dict)


def main_M():
    factor_name = [f"Distribution{i:03}" for i in [1, 30, 31] + list(range(4, 10)) + list(range(11, 28)) + [28, 29]] + \
                  [f"FundFlow{i:03}" for i in list(range(1, 7)) + list(range(9, 14)) + list(range(25, 30)) + [32, 35, 39, 40]] + \
                  [f"VolPrice{i:03}" for i in list(range(11, 18)) + [8, 9]]

    factorPa = {"minute": 1,
                "method": 'mid',
                "R": 0.1,
                "q": 0.2,
                "n": 1}

    for N in factor_name:
        if N in ['Distribution028', 'Distribution029', "FundFlow039", "FundFlow040"]:
            factorPa['n'] = 20
        elif N in ["FundFlow032"]:
            factorPa['n'] = 21
        else:
            factorPa['n'] = 1
        if N not in ['Distribution018', 'Distribution025', 'Distribution028', 'Distribution030'] + \
                ['FundFlow001', 'FundFlow006', 'FundFlow013', 'FundFlow028', 'FundFlow033', 'FundFlow034', 'FundFlow039'] + \
                ['VolPrice012', 'VolPrice013', 'VolPrice017']:
            continue
        # if N.startswith('Distribution'):
        #     continue
        factor_dict = {"factor_name": N,
                       "factor_params": factorPa,
                       'factor': None,
                       'cal': True
                       }
        try:
            cal_factor(factor_dict)
        except Exception as e:
            print(f"{N}: {e}")
    # pool = Pool(processes=4)
    # for key_, value_ in fac_dict.items():
    #     pool.apply_async(cal, (value_, pam))
    # # pool.apply_async(cal_pa, (dd, pam))
    # pool.close()
    # pool.join()


def single():
    for factor in [f"FundFlow{i:03}" for i in [46, 47, 48]] + [f"VolPrice{i:03}" for i in [18, 19, 20]]:
        if factor.startswith('FundFlow'):
            continue
            for s in ['all', 'open', 'close', 'between']:  # ['all', 'open', 'close', 'between']
                fact_dict = {
                    "factor_name": factor,
                    "factor_params": {"minute": 1,
                                      "x_min": s,
                                      "method": 'mid',
                                      "period": s,
                                      "depth": 5,
                                      "ratio": 0.1,
                                      "q": 0.2,
                                      "n": 1},
                    'cal': True
                }
                cal_factor(fact_dict)
        if factor.startswith("VolPrice"):

            for s in [5, 10]:
                fact_dict = {
                    "factor_name": factor,
                    "factor_params": {"minute": 1,
                                      "x_min": s,
                                      "method": 'mid',
                                      "period": s,
                                      "depth": s,
                                      "ratio": 0.1,
                                      "q": 0.2,
                                      "n": 1},
                    'cal': True
                }
                cal_factor(fact_dict)


if __name__ == '__main__':
    # single()
    main()
