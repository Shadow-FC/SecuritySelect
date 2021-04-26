# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

from Analysis.FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

API = LoadData()


def singleTest(factName: str = None, PathIn: str = ''):
    # 1.数据部署
    pathRead = os.path.join(PathIn, factName + '.pkl')
    with open(pathRead, mode='rb') as f:
        fact_value = pickle.load(f)
    dataParams = {
        "Factor": {"fact_name": factName,
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
        "factName": dataInput['factPoolData'].data_name,
        "hp": 5,
        "groupNum": 10,
        "retName": "retOpen",
        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  #
            "Neu": {"method": "", "p": {"mvName": "liqMv", "indName": "indexCode"}},  # industry+mv
            "Sta": {"method": "z_score", "p": {"mvName": "liqMv"}}
        },
    }

    Analysis.set_params(**Params)

    # 3.整合数据
    Analysis.integration()

    testParams = {"plot": False,
                  "save": True}
    # 4.进行因子检验
    Analysis.effectiveness(**testParams)


if __name__ == '__main__':

    fact_path = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\TechnicalBehaviorFactor'
    factNameList = ['TK_PB_20days', 'TK_PB_abs_20days', 'TK_BP_20days', 'TK_BP_abs_20days']
    for fact_ in factNameList:
        singleTest(factName=fact_, PathIn=fact_path)
    print('Finish!')
