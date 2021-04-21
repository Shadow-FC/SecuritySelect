# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

import pickle5 as pickle
from Analysis.FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

API = LoadData()


# # 单因子测试
# def Single_factor_test(params: dict,
#                        process: dict,
#                        hp: int = 1,
#                        save: bool = False):
#     """
#
#     :param params:因子参数
#     :param process: 因子处理参数
#     :param hp: 持有周期
#     :param save: 是否保存检验结果
#     :return:
#     """
#     A = FactorValidityCheck(hp=hp,
#                             stock_pool='StockPoolZD',
#                             label_pool='LabelPool')
#
#     # load pool data
#     A.load_pool_data()
#
#     # load factor data
#     A.load_factor(**params)
#
#     A.integration(**process)
#
#     # Factor validity test
#     A.effectiveness(save=save, group_num=10)
#
#
# def main1(factor_name,
#           hp,
#           save: bool = False):
#     df = pd.read_csv(f"A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorRawData\\TechnicalHighFrequencyFactor\\"
#                      f"{factor_name}.csv", header=None)
#     df.columns = ['date', 'stock_id', factor_name]
#     factor_p = {"fact_name": factor_name,
#                 "factor_params": {"switch": False},
#                 'db': 'HFD',
#                 'factor_value': df,
#                 'cal': False}
#     factor_process = {"outliers": 'mad',  # mad
#                       "neu": 'mv+industry',  # mv+industry
#                       "stand": 'mv',  # mv
#                       # "switch_freq": False,
#                       # "limit": 120
#                       }
#
#     print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}\033[0m")
#
#     Single_factor_test(params=factor_p,
#                        process=factor_process,
#                        hp=hp,
#                        save=save)
#
#
# def main2(factor_name, hp, save: bool = False):
#     fact_value = None
#
#     factor_p = {"fact_name": factor_name,
#                 "factor_params": {"n": 21},
#                 'db': 'HFD',
#                 'factor_value': fact_value,
#                 'cal': True}
#
#     factor_process = {"outliers": '',  # mad
#                       "neu": '',  # mv+industry
#                       "stand": '',  # mv
#                       "switch_freq": False,
#                       "limit": 120}
#     # factor_process1 = {"outliers": 'mad',  # mad
#     #                    "neu": 'mv+industry',  # mv+industry
#     #                    "stand": 'mv',  # mv
#     #                    "switch_freq": False,
#     #                    "limit": 120}
#
#     # print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: "
#     #       f"{factor_name}-{factor_p['factor_params']['n']}-{hp}days\033[0m")
#     print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factor_name}-{hp}days\033[0m")
#     Single_factor_test(params=factor_p,
#                        process=factor_process,
#                        hp=hp,
#                        save=save)
#
#     # Single_factor_test(params=factor_p,
#     #                    worker=factor_process1,
#     #                    hp=hp,
#     #                    save=save)


# def HighFreqFactorTest(f_name: str, f: pd.DataFrame, hp: int = 5, save: bool = True):
#     factor_p = {"fact_name": f_name,
#                 "factor_params": {"switch": False},
#                 'db': 'TEC',
#                 'factor_value': f,
#                 'cal': False}
#     factor_process = {"outliers": 'mad',  # mad
#                       "neu": '',  # mv+industry
#                       "stand": 'z_score',  # mv
#                       }
#
#     print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {f_name}\033[0m")
#     Single_factor_test(params=factor_p,
#                        process=factor_process,
#                        hp=hp,
#                        save=save)


def singleTest():
    # 1.数据部署
    file_path = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\TechnicalBehaviorFactor\CGO_Ret_20days.pkl'
    with open(file_path, mode='rb') as f:
        fact_value = pickle.load(f)
    dataParams = {
        "Factor": {"fact_name": "CGO_Ret_20days",
                   "fact_value": fact_value,
                   "fact_params": {"": ""}
                   },
        "StockPool": "StockPoolZD",
        "LabelPool": "strategyLabel"
    }

    dataInput = {
        "factPoolData": API.getFactorData(**dataParams['Factor']),
        "stockPoolData": API.getStockPoolData(dataParams['StockPool']),
        "labelPoolData": API.getLabelPoolData(dataParams['LabelPool']),
    }

    Analysis = FactorValidityCheck()
    Analysis.set_data(**dataInput)

    # 2.检验参数设置
    Params = {
        "fact_name": dataInput['factPoolData'].data_name,
        "hp": 5,
        "groupNum": 10,
        "retName": "retOpen",

        "RO": {"method": "mad", "p": {}},  #
        "Neu": {"method": "", "p": {"mvName": "liqMv", "indName": "indexCode"}},  # industry+mv
        "Sta": {"method": "mv", "p": {"mvName": "liqMv"}},
    }

    Analysis.set_params(**Params)

    # 3.整合数据
    Analysis.integration()

    testParams = {"plot": True,
                  "save": False}
    # 4.进行因子检验
    Analysis.effectiveness(**testParams)


if __name__ == '__main__':

    # file_path = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\TechnicalBehaviorFactor\CGO_Ret_20days.csv'
    # fact_value = pd.read_csv(file_path)
    singleTest()
    print('Finish!')

    # factor_p = {"fact_name": "CGO_Ret_20days",
    #             "fact_params": {"n": 20},
    #             'fact_value': fact_value}
    #
    # fact_pro = {"outliers": 'mad',  # mad
    #             "neu": '',  # mv+industry
    #             "stand": 'mv',  # mv
    #             }
    # Single_factor_test(factor_p, fact_pro, 5, True)
    # #
    # factor_file_path = r'D:\DataBase'
    # factor_files = [folder for folder in os.listdir(factor_file_path) if folder.startswith('HighFrequency')]
    # for HighFreqFolder in factor_files:
    #     HighFreqPath = os.path.join(factor_file_path, HighFreqFolder)
    #     factor_list = os.listdir(HighFreqPath)
    #     for factor in factor_list:
    #         if factor not in ['Distribution008_1min_1days.csv', 'Distribution010_1min_1days.csv',
    #                           'Distribution015_1min_1days.csv',
    #                           'FundFlow003_1days.csv', 'FundFlow004_1days.csv', 'FundFlow006_0.2q_1days.csv',
    #                           'FundFlow012_1days.csv',
    #                           'FundFlow026_1days.csv', 'FundFlow027_1days.csv', 'FundFlow034_10min_C_1days.csv',
    #                           'FundFlow039_20days.csv',
    #                           'FundFlow040_20days.csv', 'VolPrice008_0.2q_1days.csv', 'VolPrice009_1days.csv',
    #                           'VolPrice013_1min_1days.csv',
    #                           'VolPrice017_1days.csv']:
    #             continue
    #
    #         factor_name = factor[:-4]
    #         f_value = pd.read_csv(os.path.join(HighFreqPath, factor))
    #         if '_1days' in factor_name:
    #             f_value = f_value.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
    #             f_value = f_value.groupby(KN.STOCK_ID.value,
    #                                       group_keys=False).rolling(hp, min_periods=round(hp * 0.8)).mean()
    #             f_value = f_value.reset_index()
    #             factor_name_new = factor_name.replace("_1days", f"_{hp}days")
    #             f_value = f_value.rename(columns={factor_name: factor_name_new})
    #         else:
    #             factor_name_new = factor_name
    #         try:
    #             HighFreqFactorTest(factor_name_new, f_value, hp, True)
    #         except Exception as e:
    #             print(e)
    #         print('Stop')
