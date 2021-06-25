from utility.utility import factor_to_pkl
from Analysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

factorValid = {"HighFrequencyDistributionFactor": ['Distribution006_1min_1days', 'Distribution007_1min_1days',
                                                   'Distribution013_1min_1days',
                                                   'Distribution018_1min_1days', 'Distribution019_1min_1days',
                                                   'Distribution024_1min_1days',
                                                   'Distribution025_1min_1days', 'Distribution026_1min_1days'],
               "HighFrequencyFundFlowFactor": ['FundFlow003_1days', 'FundFlow006_02q_1days', 'FundFlow009_1days',
                                               'FundFlow012_1days',
                                               'FundFlow013_1days', 'FundFlow018_all_1days',
                                               'FundFlow018_between_1days', 'FundFlow018_open_1days',
                                               'FundFlow018_open_1days', 'FundFlow020_open_1days', 'FundFlow026_1days',
                                               'FundFlow034_5min_C_1days',
                                               'FundFlow039_20days', 'FundFlow040_20days',
                                               'FundFlow046_5depth_open_1days'],
               "HighFrequencyVolPriceFactor": ['VolPrice008_02q_1days', 'VolPrice013_1min_1days',
                                               'VolPrice019_10depth_1days']}

factorComp = {'Syn1_5days': ['Distribution004_1min_1days', 'Distribution005_1min_1days', 'Distribution006_1min_1days',
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
    pathIn = r'A:\DataBase\SecuritySelectData\FactorPool\FactorDataSet'
    m = []
    for factType, factNames in factorValid.items():
        factFolder = os.path.join(pathIn, factType)
        facts = os.listdir(factFolder)
        for fact in facts:
            with open(os.path.join(factFolder, fact), 'rb') as f:
                dataD = pickle.load(f).set_index(['date', 'code'])
                m.append(dataD)
    factData = pd.concat(m, axis=1)

    Corr = FactorCollinearity()
    Corr.set_data(factPoolData=factData)

    Params = {
        "methodProcess": {
            "RO": {"method": "mad", "p": {}},  # 异常值处理
            "Sta": {"method": "z_score", "p": {}},  # 标准化

            "Cor": {"method": "LinCor", "p": {"corName": "spearman"}}  # 相关性检验方法
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

            "Cor": {"method": "LinCor", "p": {"corName": "spearman"}},  # 相关性检验方法
            # "Syn": {"method": "PCA", "p": {}},  # "weightAlgo": "IC_IR"
            "Syn": {"method": "OPT", "p": {"weightAlgo": "IC_IR"}},
        },

        "hp": 5,  # 持有周期
        "rp": 20,  # 滚动周期

        "plot": True,
        "save": True
    }

    # 2.加载因子暴露和因子收益
    for compName, compFacts in factorComp.items():
        exps, weights = [], []
        for factSub in compFacts:
            with open(factPathDict[factSub]['expPath'], 'rb') as f:
                dataD = pickle.load(f).set_index(['date', 'code'])

            if factSub.endswith('_1days'):
                nameNew = factSub.replace('_1days', '_5days')
                # 原始因子需要滚动N日取平均
                dataD = dataD.groupby('code').apply(lambda x: x.rolling(5, min_periods=1).mean())
                dataD = dataD.rename(columns={factSub: nameNew})
            else:
                nameNew = factName

            exps.append(dataD)
            dataW = pd.read_csv(factPathDict[nameNew]['retPath']).set_index('date')['IC']
            dataW.name = nameNew
            weights.append(dataW)

        factExps = pd.concat(exps, axis=1)
        factRets = pd.concat(weights, axis=1)

        Params['synName'] = compName
        Corr.set_params(**Params)
        Corr.set_data(factPoolData=factExps,
                      factWeightData=factRets)

        # 3.清洗数据
        Corr.Cleaning()

        # 4.因子合成
        factValue = Corr.factor_Synthetic()

        # 5.因子存储
        F = DataInfo(data=factValue,
                     data_name=compName,
                     data_type='Synthetic',
                     data_category='SyntheticFactor')
        factor_to_pkl(F)


if __name__ == '__main__':
    CorTest()
    factRet = r'A:\DataBase\SecuritySelectData\FactorPool\FactorsTestResult\FactRet'

    factDict = factParma(FPN.Fact_dataSet.value, factRet)
    FactSynthetic(factDict)
