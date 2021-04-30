from utility.utility import factor_to_pkl
from Analysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

factorComp = {'Syn1': ['Distribution004_1min_1days', 'Distribution005_1min_1days', 'Distribution006_1min_1days',
                       'Distribution007_1min_1days']}


# 因子路径
def factParma(pathExp: str, pathRet: str) -> Dict[str, str]:
    # 因子暴露路径
    expFolders = os.listdir(pathExp)
    res = defaultdict(dict)
    for folder in expFolders:
        subPath = os.path.join(pathExp, folder)
        factFiles = os.listdir(subPath)
        for factName in factFiles:
            res[factName.split('.')[0]]['expPath'] = os.path.join(subPath, factName)

    # 因子收益路径
    retFiles = os.listdir(pathRet)
    for file in retFiles:
        res[file.split('.')[0]]['retPath'] = os.path.join(pathRet, file)

    return res


def CorTest():
    pathIn = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet\HighFrequencyDistributionFactor'
    m = []
    for fact_ in ['Distribution004_1min_1days', 'Distribution005_1min_1days', 'Distribution006_1min_1days',
                  'Distribution007_1min_1days']:
        with open(os.path.join(pathIn, fact_ + '.pkl'), 'rb') as f:
            dataD = pickle.load(f).set_index(['date', 'code'])
            m.append(dataD)
    factData = pd.concat(m, axis=1)

    Corr = FactorCollinearity()
    Corr.set_data(factPoolData=factData)

    Params = {
        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  #
            "Sta": {"method": "z_score", "p": {}},

            "Cor": {"method": "LinCor", "p": {"corName": "pearson"}}
        },
    }

    Corr.set_params(**Params)

    Corr.Cleaning()
    Corr.correctionTest(plot=True)


def FactSynthetic(factPathDict: Dict[str, str]):
    Corr = FactorCollinearity()

    # 1.参数设置
    Params = {
        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  #
            "Sta": {"method": "z_score", "p": {}},

            "Syn": {"method": "OPT", "p": {"rp": 60, "hp": 5, "retType": "IC_IR"}},
        },

        # "Syn": {"method": "RetWeight", "p": {"rp": 60, "hp": 5, "algorithm": "HalfTime"}},
    }

    Corr.set_params(**Params)

    # 2.加载因子暴露和因子收益
    for compName, compFacts in factorComp.items():
        exps, rets = [], []
        for factSub in compFacts:
            with open(factPathDict[factSub]['expPath'], 'rb') as f:
                dataD = pickle.load(f).set_index(['date', 'code'])
                # 原始因子需要滚动N日取平均
                dataD = dataD.groupby('code').apply(lambda x: x.rolling(5, min_periods=1).mean())
                exps.append(dataD)
            with open(factPathDict[factSub]['retPath'], 'rb') as f:
                dataD = pickle.load(f).set_index('date')['IC']
                dataD.name = factSub
                rets.append(dataD)
        factExps = pd.concat(exps, axis=1)
        factRets = pd.concat(rets, axis=1)

        Corr.set_data(factPoolData=factExps,
                      factWeightData=factRets)

        # 3.清洗数据
        Corr.Cleaning()

        # 4.因子合成
        factValue = Corr.factorSynthetic()

        # 5.因子存储
        F = DataInfo(data=factValue,
                     data_name=compName,
                     data_type='Syn',
                     data_category='SyntheticFactor')
        factor_to_pkl(F)


if __name__ == '__main__':
    factExp = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet'
    factRet = r'A:\DataBase\SecuritySelectData\FactorPool\FactorsTestResult\FactRet'
    # CorTest()

    factDict = factParma(factExp, factRet)
    FactSynthetic(factDict)
    # main()
