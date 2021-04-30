# -*-coding:utf-8-*-
# @Time:   2020/10/15 16:05
# @Author: FC
# @Email:  18817289038@163.com

from Analysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

email = {"FC": {"user": "18817289038@163.com",
                "password": "PFBMGFCIDJJGRRCK",
                "host": "smtp.163.com"},
         }

API = LoadData()


# 多因子测试
def multipleTest(factPathDict: Dict[str, str]):
    # 1.初始化数据
    dataParams = {
        "StockPool": "StockPoolZD",
        "LabelPool": "strategyLabel"
    }

    dataInput = {
        "stockPoolData": API.getStockPoolData(dataParams['StockPool']),
        "labelPoolData": API.getLabelPoolData(dataParams['LabelPool']),
    }

    Analysis = FactorValidityCheck()
    Analysis.set_data(**dataInput)

    # 2.检验参数设置
    Params = {
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

    # 多因子检验
    for factName, path in factPathDict.items():
        print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {factName}\033[0m")
        with open(path, 'rb') as f:
            factValue = pickle.load(f)
        if '_1days' in factName:

            factValue = factValue.set_index(['date', 'code']).rolling(5, min_periods=1).mean().reset_index()
            nameNew = factName.replace('_1days', '_5days')
            factValue = factValue.rename(columns={factName: nameNew})
        else:
            nameNew = factName

        factParams = {"fact_name": nameNew,
                      "fact_value": factValue,
                      "fact_params": {"": ""}
                      }
        Analysis.set_data(factPoolData=API.getFactorData(**factParams))
        Analysis.set_params(factName=nameNew)

        # 3.整合数据
        Analysis.integration()

        testParams = {"plot": True,
                      "save": True}
        # 4.进行因子检验
        Analysis.effectiveness(**testParams)


def factPath(pathIn: str) -> Dict[str, str]:
    factNames = os.listdir(pathIn)
    res = {factName.split('.')[0]: os.path.join(pathIn, factName) for factName in factNames}
    return res


if __name__ == '__main__':
    for i in ['HighFrequencyDistributionFactor', 'HighFrequencyFundFlowFactor', 'HighFrequencyVolPriceFactor']:

        fact_path = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\{}'.format(i)
        factPaths = factPath(fact_path)
        multipleTest(factPaths)




