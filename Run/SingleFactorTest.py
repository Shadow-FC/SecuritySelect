# -*-coding:utf-8-*-
# @Time:   2020/9/14 11:26
# @Author: FC
# @Email:  18817289038@163.com

from Analysis.FactorAnalysis import *

API = LoadData()


def singleTest(factName: str = None, PathIn: str = ''):
    # 1.数据部署
    pathRead = os.path.join(PathIn, factName)
    with open(pathRead, mode='rb') as f:
        factValue = pickle.load(f)
    if factName.endswith('_1days.pkl'):
        factValue = factValue.set_index(['date', 'code']).groupby('code').apply(
            lambda x: x.rolling(5, min_periods=1).mean()).reset_index()
        nameNew = factName.split('.')[0].replace('_1days', '_5days')
        factValue = factValue.rename(columns={factName.split('.')[0]: nameNew})
    else:
        nameNew = factName.split('.')[0]

    dataParams = {
        "Factor": {"fact_name": nameNew,
                   "fact_value": factValue,
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

    Analysis = FactorValidityCheckRelative()
    Analysis.set_data(**dataInput)

    # 2.检验参数设置
    Params = {
        "factName": dataInput['factPoolData'].data_name,
        "hp": 5,
        "groupNum": 10,
        "groupMethod": "equalValue",
        "retName": "retOpen",
        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  #
            "Neu": {"method": "industry+mv", "p": {"mvName": "liqMv", "indName": "indexCode"}},  #
            "Sta": {"method": "", "p": {"mvName": "liqMv"}}  # z_score
        },
    }

    Analysis.set_params(**Params)

    # 3.整合数据
    Analysis.integration()

    testParams = {"plot": True,
                  "save": True}
    # 4.进行因子检验
    Analysis.effectiveness(**testParams)


if __name__ == '__main__':
    path = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet'
    name = ['HighFrequencyDistributionFactor', 'HighFrequencyFundFlowFactor', 'HighFrequencyVolPriceFactor']
    for i in name:
        folderPath = os.path.join(path, i)
        factNameList = os.listdir(folderPath)
        for fact_ in factNameList:
            singleTest(factName=fact_, PathIn=folderPath)
        print('Finish!')
