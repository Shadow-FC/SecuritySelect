from utility.utility import factor_to_pkl
from Analysis.FactorAnalysis.FactorAnalysis import *

DATABASE_NAME = {"Group": "分组数据保存",
                 "Fin": "基本面因子保存",
                 "PV": "价量易因子保存",
                 "GenPro": "遗传规划算法挖掘因子保存"}

# factorSynthetic = {FCN.Val.value: {"BP_LR", "BP_ttm", "DP_ttm", "E2P_ttm", "EP_cut_ttm",
#                                  "EP_LR", "EP_ttm", "SP_LR", "SP_ttm"},
#
#                  FCN.Gro.value: {"BPS_G_LR", "EPS_G_ttm", "ROA_G_ttm", "MAR_G",
#                                  "NP_Stable", "OP_Stable", "OR_Stable",
#                                  "ILA_G_ttm_std", "TA_G_LR_std"},
#
#                  FCN.Pro.value: {"NPM_T", "ROA_ttm"},
#
#                  FCN.Ope.value: {"RROC_N", "TA_Turn_ttm_T"},
#
#                  FCN.Sol.value: {"IT_qoq_Z", "OT2NP_qoq_Z",
#                                  "ShortDebt2_CFPA_qoq_abs", "ShortDebt3_CFPA_qoq_abs",
#                                  "ShortDebt3_CFPA_std"},
#                  FCN.EQ.value: {}}
#
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
            "Cor": {"method": "LinCor", "p": {"corName": "pearson"}}
        },
        # "methodSynthetic": {
        #     "method": "RetWeight",
        #     "p": {"rp": 60,
        #           "hp": 5,
        #           "algorithm": "HalfTime"}, },
        "methodSynthetic": {
            "method": "OPT",
            "p": {"rp": 60,
                  "hp": 5,
                  "retType": "IC_IR"}, }
    }

    Corr.set_params(**Params)

    # 2.加载因子暴露和因子收益
    for compName, compFacts in factorComp.items():
        exps, rets = [], []
        for factSub in compFacts:
            with open(factPathDict[factSub]['expPath'], 'rb') as f:
                dataD = pickle.load(f).set_index(['date', 'code'])
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
