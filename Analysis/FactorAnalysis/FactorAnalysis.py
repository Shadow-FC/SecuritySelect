import os
import json
import time
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import datetime as dt
import pickle5 as pickle
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy import stats
from collections import defaultdict
from typing import List, Union, Dict, Any

from DataAPI.LoadData import LoadData
from DataAPI.DataInput.GetData import SQL, CSV

from EvaluationIndicitor.Indicator import Indicator

from utility.FactorUtility import (
    RemoveOutlier as RO,
    Neutralization as Neu,
    Standardization as Sta,
    Correlation as Cor,
    DataSynthesis as Syn,
)
from utility.utility import (
    timer
)

from Object import (
    DataInfo
)
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)

warnings.filterwarnings(action='ignore')

sns.set(font='SimHei', palette="muted", color_codes=True)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

"""
缺失值处理，需要提前定义好子样本
"""


# 因子处理方法合集
class FactorProcess(object):
    # 方法参数必须被继承复写
    methodProcess = {
        "RO": {"method": "", "p": {}},
        "Neu": {"method": "", "p": {}},
        "Sta": {"method": "", "p": {}},
        "Cor": {"method": "pearson", "p": {}}
    }

    methodSynthetic = {
        "method": "EqualWeight",
        "p": {"rp": 60,
              "hp": 5,
              "algorithm": "mean"},

    }

    def __init__(self):
        self.Cor = Cor()
        self.Syn = Syn()
        self.RO = RO()
        self.Neu = Neu()
        self.Sta = Sta()

    # 更新因子处理参数
    def set_params(self, **kwargs):
        """

        Parameters
        ----------
        Returns
        -------
        对于因子处理方法设置因子参数
        """
        for paramName, paramValue in kwargs.items():
            setattr(self, paramName, paramValue)

    def processSingle(self,
                      data: Union[pd.DataFrame, pd.Series],
                      methodN: str,
                      **kwargs) -> Any:
        """
        单一处理方法
        Parameters
        ----------
        data :
        methodN :
        kwargs :

        Returns
        -------

        """
        value = getattr(self, methodN).process(data=data,
                                               method=self.methodProcess[methodN]['method'],
                                               **self.methodProcess[methodN]['p'],
                                               **kwargs)
        return value

    def processSeq(self,
                   data: pd.DataFrame,
                   methodN: List[str],
                   dataName: str,
                   **kwargs):
        """
        连续处理
        Parameters
        ----------
        data :
        methodN :
        dataName :
        kwargs :

        Returns
        -------

        """
        for M in methodN:
            if self.methodProcess[M]['method'] != "":
                value = getattr(self, M).process(data=data,
                                                 method=self.methodProcess[M]['method'],
                                                 dataName=dataName,
                                                 **self.methodProcess[M]['p'],
                                                 **kwargs)
                data[dataName] = value

    def processSyn(self,
                   factData: pd.DataFrame,
                   factWeight: pd.DataFrame,
                   **kwargs
                   ) -> pd.Series(float):
        methodN = self.methodSynthetic['method']
        value = getattr(self, "Syn").process(factData=factData,
                                             factWeight=factWeight,
                                             method=methodN,
                                             **self.methodSynthetic['p'],
                                             **kwargs)
        return value


# 单因子有效性测试
class FactorValidityCheck(FactorProcess):
    """
    对于单因子的有效性检验，我们从以下几个维度进行考量：
    1.因子暴露稳定性
    截面自相关系数(spearman秩相关)

    1.单因子与下期收益率回归：
        1)因子T值序列绝对值平均值--因子有效性是否显著；
        2)因子T值序列绝对值大于2的占比--因子是否稳定(防止部分异常T值干扰均值)；

        3)因子T值序列均值绝对值除以T值序列的标准差--因子有效性(方向，稳定，收益)；
          # 因子T值序列绝对值均值除以标准差--因子有效性
          因子T值序列的T检验--因子方向性

        4)因子收益率序列平均值--因子方向是否一致；
          因子收益率序列大于0占比
        # 5)因子收益率序列平均值零假设T检验--因子收益率是否显著不为零

    2.因子IC值：
        1)因子IC值序列的均值大小--判断因子方向是否一致；
        # 2)因子IC值序列绝对值均值大小--因子有效性；
        3)因子IC值序列的标准差--因子稳定性；
        4)因子IR比率--因子有效性；
        # 5)因子IC值累积曲线--随时间变化效果是否稳定
        6)因子IC值序列大于零的占比--判断因子的方向
    3.分层回测检验单调性-打分法：
        按照基准权重合成各组净值曲线

        普通检验指标
        因子方向：最优组减去最劣组
        时间区间划分：整个回溯区间，近一年，近三年到近一年，三年前
        1)年化收益率；
        2)年化波动率；
        3)年化信息比
        4)最大回撤；
        5)捕获率
        6)曲度

        方差分析分月度和年度：F值
        单调性检验：截面组序列与收益率rank的相关性
        衰减度：各年化收益率差分

        稳定性指标：相关性，IR
        复杂度指标

    """
    hp = 5
    groupNum = 5

    retName = 'retOpen'

    parent_path = os.path.abspath(os.path.dirname(os.getcwd()))

    industryMapping = ["CI005001.WI",
                       "CI005002.WI",
                       "CI005003.WI",
                       "CI005004.WI",
                       "CI005005.WI",
                       "CI005006.WI",
                       "CI005007.WI",
                       "CI005008.WI",
                       "CI005009.WI",
                       "CI005010.WI",
                       "CI005011.WI",
                       "CI005012.WI",
                       "CI005013.WI",
                       "CI005014.WI",
                       "CI005015.WI",
                       "CI005016.WI",
                       "CI005017.WI",
                       "CI005018.WI",
                       "CI005019.WI",
                       "CI005020.WI",
                       "CI005021.WI",
                       "CI005022.WI",
                       "CI005023.WI",
                       "CI005024.WI",
                       "CI005025.WI",
                       "CI005026.WI",
                       "CI005027.WI",
                       "CI005028.WI",
                       "CI005029.WI",
                       "CI005030.WI"]

    def __init__(self):
        super(FactorValidityCheck, self).__init__()

        self.CSV = CSV()

        self.api = LoadData()  # 数据接口

        self.factProc = FactorProcess()  # 因子预处理
        self.ind = Indicator()  # 评价指标的计算

        self.dataSet = {}  # 原始数据集

        self.Res = defaultdict(dict)  # 因子检验结果

        self.td = self.CSV.trade_date_csv()  # 交易日序列

    # 原始数据存储
    def set_data(self,
                 factPoolData: Union[DataInfo, None] = None,
                 stockPoolData: Union[DataInfo, None] = None,
                 labelPoolData: Union[DataInfo, None] = None,
                 **kwargs):
        """
        数据发生变动则替换原值，否则保持原值
        Args:
            factPoolData: 因子
            stockPoolData: 股票池
            labelPoolData: 标签池
            **kwargs:
        Returns:
        """
        if factPoolData is not None:
            self.dataSet['factDirty'] = factPoolData
        if stockPoolData is not None:
            self.dataSet['stockPool'] = stockPoolData
        if labelPoolData is not None:
            self.dataSet['labelPool'] = labelPoolData

    # DataInput Integration
    @timer
    def integration(self):
        """
        1.输入标签可能存在空值，最后结果保存时进行了以因子暴露值为基准的去空处理，
          在后续研究分析中因子暴露值可以保证不为空，但标签依然会存在空值；
        2.基准的合成采用最原始的股票池进行合成；
        3.涨跌停仅在开仓时需要处理，开仓时涨跌停不进行建仓处理，但该股在分析时依然占有一定的权重
        4.为方便研究，数据保存原始数据，去极值后的数据，中性化后的数据
        :return:
        """
        # 1.Get data
        SP, LP, FP = self.dataSet["stockPool"].data, self.dataSet["labelPool"].data, self.dataSet['factDirty'].data

        # 2.Merge factor value and label pool on factor value that after stock pool filter
        self.dataSet['Expos'] = pd.merge(FP.reindex(SP.index),
                                         LP,
                                         left_index=True,
                                         right_index=True,
                                         how='left')

        # 3.Synthetic benchmark return
        self.dataSet['BMRet'] = self.api.getBMData(
            benchMarkName='benchmarkZD',
            data=self.dataSet['Expos'],
            retName=self.retName,
            wName=SN.STOCK_WEIGHT.value)

        # 4.Filter Up Down Limit Stock  TODO
        self.dataSet['ExpClean'] = self.dataSet['Expos'].copy(deep=True)

        # 5.worker factor
        self.processSeq(self.dataSet['ExpClean'], methodN=['RO', 'Neu', 'Sta'], dataName=self.factName)

        # 6.save data
        # self.dataSet['ExpClean'][self.factName] = self.dataSet['factClean'][self.factName]

        # 7.drop Nan by fact value, 试试inplace会不会改变self.dataSet['Expos']
        self.dataSet["ExpClean"] = self.dataSet["ExpClean"].dropna(subset=[self.factName])

    # Factor validity test
    @timer
    def effectiveness(self,
                      plot: bool = True,
                      save: bool = True):

        # 清空之前的记录
        self.Res.clear()

        data_clean = self.dataSet['ExpClean'].copy(deep=True)

        # 检验
        try:

            # 因子暴露的稳定性
            self.factStability(data_clean[self.factName])

            self.fact_ret(data=data_clean)

            self.IC_IR(data=data_clean)

            self.Layered(data=data_clean)

        except Exception as e:
            print(f"Factor test error：{e}")
        else:
            # plot
            if plot:
                self.plotRes(save=save)
            if save:
                factRet = pd.concat([self.Res['reg']['res_reg'], self.Res['IC']['IC_rank']], axis=1).reset_index()
                factRet.to_pickle(os.path.join(FPN.factor_test_res.value, f'FactRet{os.sep}{self.factName}.pkl'))

    # 因子稳定性
    def factStability(self, data: pd.Series):
        """
        因子暴露稳定性，spearman相关性
        Parameters
        ----------
        data :

        Returns
        -------

        """
        fact_df = data.unstack()
        self.Res["Stability"] = fact_df.corrwith(fact_df.shift(1), axis=1, drop=True, method='spearman').sort_index()

    # 单因子与下期收益率回归
    def fact_ret(self,
                 data: pd.DataFrame,
                 **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        data['hpRet'] = self._holding_ret(data[self.retName])
        df_data = data[['hpRet', SN.INDUSTRY_FLAG.value, self.factName, SN.STOCK_WEIGHT.value]]
        df_data = df_data.rename(columns={"hpRet": KN.RETURN.value}).dropna(how='any')

        # Analytic regression result：T Value and Factor Return
        res_reg = df_data.groupby(KN.TRADE_DATE.value).apply(self._reg_fact_ret).dropna(how='all')

        # get Trade date
        res_reg = res_reg.reindex(self.td[(self.td[KN.TRADE_DATE.value] >= res_reg.index[0]) &
                                          (self.td[KN.TRADE_DATE.value] <= res_reg.index[-1])][KN.TRADE_DATE.value])

        # Calculate Indicators
        T_abs_mean = abs(res_reg['retT']).mean()
        T_abs_up_2 = res_reg['retT'][abs(res_reg['retT']) > 2].count() / res_reg.dropna().shape[0]
        T_stable = abs(res_reg['retT'].mean()) / res_reg['retT'].std()

        fact_ret_mean = res_reg['fact_ret'].mean()
        T_ttest = stats.ttest_1samp(res_reg['retT'].dropna(), 0)

        # 最近一年表现
        T_year = res_reg['retT'][-244:]
        T_abs_mean_year = abs(T_year).mean()
        T_abs_up_2_year = T_year[abs(T_year) > 2].count() / T_year.dropna().shape[0]
        T_stable_year = abs(T_year.mean()) / T_year.std()
        T_ttest_year = stats.ttest_1samp(T_year.dropna(), 0)

        indicators = pd.Series([T_abs_mean, T_abs_up_2, T_stable, fact_ret_mean, T_ttest[0],
                                T_abs_mean_year, T_abs_up_2_year, T_stable_year, T_ttest_year[0]],
                               index=['T_abs_mean', 'T_abs_up_2', 'T_stable', 'fact_ret', 'T_ttest',
                                      'T_abs_mean_year', 'T_abs_up_2_year', 'T_stable_year', 'T_ttest_year'],
                               name=self.factName)

        # 因子收益路径依赖处理
        fact_ret_path = self.corr_path(res_reg['fact_ret'])

        # save data to dict
        self.Res['reg'] = {
            "res_reg": res_reg,
            "Indicators": indicators,
            "path_ret": fact_ret_path
        }

    # 因子IC值
    def IC_IR(self,
              data: pd.DataFrame,
              **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        data['hpRet'] = self._holding_ret(data[self.retName])
        df_data = data[['hpRet', self.factName, SN.STOCK_WEIGHT.value]].dropna()
        df_data = df_data.rename(columns={"hpRet": KN.RETURN.value})

        IC_rank = df_data.groupby(KN.TRADE_DATE.value).apply(
            lambda x: self.weight_cor(x[[self.factName, KN.RETURN.value]].rank(), x[SN.STOCK_WEIGHT.value]))
        IC_rank.name = 'IC'

        IC_rank = IC_rank.reindex(self.td[(self.td[KN.TRADE_DATE.value] >= IC_rank.index[0]) &
                                          (self.td[KN.TRADE_DATE.value] <= IC_rank.index[-1])][KN.TRADE_DATE.value])

        IC_mean, IC_std = IC_rank.mean(), IC_rank.std()
        IR, IC_up_0 = IC_mean / IC_std, len(IC_rank[IC_rank > 0]) / IC_rank.dropna().shape[0]

        IC_year = IC_rank[-244:]
        IC_mean_year, IC_std_year = IC_year.mean(), IC_year.std()
        IR_year, IC_up_0_year = IC_mean_year / IC_std_year, len(IC_year[IC_year > 0]) / IC_year.dropna().shape[0]

        indicators = pd.Series([IC_mean, IC_std, IR, IC_up_0,
                                IC_mean_year, IC_std_year, IR_year, IC_up_0_year],
                               index=['IC_mean', 'IC_std', 'IR', 'IC_up_0',
                                      'IC_mean_year', 'IC_std_year', 'IR_year', 'IC_up_0_year'],
                               name=self.factName)
        # Path dependence
        IC_path = self.corr_path(IC_rank)

        # save data to dict
        self.Res['IC'] = {"IC_rank": IC_rank, "Indicators": indicators, "path_IC": IC_path}

    # 分层回测检验
    def Layered(self,
                data: pd.DataFrame,
                **kwargs):
        df_data = data[[self.retName, self.factName, SN.INDUSTRY_FLAG.value, SN.STOCK_WEIGHT.value, PVN.Up_Down.value]]
        df_data = df_data.rename(columns={self.retName: KN.RETURN.value})

        # Grouping
        df_data[SN.GROUP.value] = df_data[self.factName].groupby(KN.TRADE_DATE.value).apply(
            lambda x: pd.cut(x.rank(), bins=self.groupNum, labels=False) + 1)

        # Average group return
        groupRet = self.group_ret(df_data[[KN.RETURN.value, SN.GROUP.value, SN.STOCK_WEIGHT.value]],
                                  weight_name=SN.STOCK_WEIGHT.value)
        ################################################################################################################
        # NAV: Continuous and uninterrupted, unfill Nav
        groupRet = pd.merge(groupRet,
                            self.dataSet['BMRet'].data,
                            left_index=True,
                            right_index=True,
                            how='left').dropna()
        nav = groupRet.add(1).cumprod()
        ex_nav = nav.div(nav[self.dataSet['BMRet'].data_name], axis=0).drop(columns=self.dataSet['BMRet'].data_name)

        # 计算指标
        self.LayeredIndicator(ex_nav, nav)

        self.Res['Layered'].update({"dataMerge": df_data, "exNav": ex_nav})
        # 多头超额收益信息比和最强组各行业表现(超额收益最大组)

    """指标的计算"""

    # cal ind
    def LayeredIndicator(self, ex_nav: pd.DataFrame, nav: pd.DataFrame):
        # 变量集合
        Var = {}
        # ret days
        ex_ret = ex_nav.pct_change(fill_method=None).dropna()
        ret = nav.pct_change(fill_method=None).dropna()
        # Date notes
        lastDate = ex_nav.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        mapping = {"All": ":",
                   "1Y": "date1Y:",
                   "3Y": "date3Y:",
                   "1T3Y": "date3Y:date1Y",
                   "3YBef": ":date3Y"}

        # 基础指标的计算:年化收益率，年化波动率，年化夏普比，最大回撤
        indMethod = [self.ind.return_a, self.ind.std_a, self.ind.shape_a, self.ind.max_retreat]
        for v_ in ['All', '1Y', '3Y', '1T3Y', '3YBef']:
            retSub = eval(f"ret.loc[{mapping[v_]}]")
            Var[f"ex_nav{v_}"] = eval(f"ex_nav.loc[{mapping[v_]}]")
            Var[f"ex_ret{v_}"] = eval(f"ex_ret.dropna().loc[{mapping[v_]}]")
            Var[f"ind{v_}"] = Var[f"ex_nav{v_}"].agg(indMethod)

            # 曲率 TODO 待考究
            Cur = Var[f"ex_nav{v_}"].resample("M").last().rolling(3).apply(lambda y: self.PJCur([1, 2, 3], y.to_list()))
            Var[f"ind{v_}"].loc['Cur'] = Cur.mean()

            # 捕获率
            BMRet = eval(f"retSub[self.dataSet['BMRet'].data_name].loc[{mapping[v_]}]")

            retUp = (1 + retSub[BMRet > 0]).prod() ** (244 / retSub.shape[0]) - 1
            retDown = (1 + retSub[BMRet < 0]).prod() ** (244 / retSub.shape[0]) - 1
            CapUp = retUp.div(retUp[self.dataSet['BMRet'].data_name], axis=0)
            CapDown = retDown.div(retDown[self.dataSet['BMRet'].data_name], axis=0)
            Var[f"ind{v_}"].loc['CapUp'] = CapUp
            Var[f"ind{v_}"].loc['CapDown'] = CapDown
            Var[f"ind{v_}"].loc['Cap'] = CapUp / CapDown

            # 衰减
            Var[f"ind{v_}"].loc['decay'] = Var[f"ex_nav{v_}"].resample("M").agg(self.ind.accumulative_return).mean()
            # 单调性
            Var[f"Monotony"] = Var[f"ex_nav{v_}"].resample(f"{self.hp}D").last().corrwith(
                pd.Series([i for i in range(1, self.groupNum + 1)], index=Var[f"ex_nav{v_}"].columns),
                axis=1,
                method='spearman').mean()
            # 单因素方差分析
            Var["FTest"], Var["F_PValue"] = stats.f_oneway(*Var[f"ex_ret{v_}"].dropna().values.T)

            # 数据合并
            Var[f"ind{v_}"].loc['Monotony'] = Var["Monotony"]
            Var[f"ind{v_}"].loc['FTest'] = Var["FTest"]
            Var[f"ind{v_}"].loc['F_PValue'] = Var["F_PValue"]
            # merge
            self.Res["Layered"][f"Indicators{v_}"] = Var[f"ind{v_}"]

    # Plot
    def plotRes(self, **kwargs):
        """
        Don't edit it
        Parameters
        ----------
        kwargs :

        Returns
        -------

        """

        factExp = self.dataSet['ExpClean'][self.factName]

        exNav = self.Res['Layered']['exNav']
        exRet = exNav.pct_change(fill_method=None).dropna()

        lastDate = exNav.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        factExp_df = factExp.unstack()
        dataPlot = pd.DataFrame(
            {
                "Count": factExp_df.count(axis=1),
                "25%": factExp_df.quantile(0.25, axis=1),
                "50%": factExp_df.quantile(0.5, axis=1),
                "75%": factExp_df.quantile(0.75, axis=1),
            })
        IndG = {}
        for ind_ in ['Cap', 'decay', 'Cur', 'return_a', 'std_a', 'shape_a', 'max_retreat']:
            data = {}
            for period_ in ['All', '3Y', '1Y']:
                data[period_] = self.Res['Layered'][f'Indicators{period_}'].loc[ind_].round(4)
            IndG[ind_] = pd.DataFrame(data).stack().reset_index().rename(columns={'level_0': 'Group',
                                                                                  'level_1': 'Period',
                                                                                  0: ind_})
        IndS = {}
        for period_ in ['All', '3Y', '1Y']:
            data = {}
            for ind_ in ['Monotony', 'F_PValue']:
                data[ind_] = round(self.Res['Layered'][f'Indicators{period_}'].loc[ind_, 'G1'], 4)
            IndS[period_] = pd.Series(data)

        fig = plt.figure(figsize=(20, 16))

        ax1 = fig.add_subplot(4, 4, 1)
        factExp.plot.hist(bins=30, ax=ax1)

        ax2 = fig.add_subplot(4, 4, 2)
        ax2.xaxis.label.set_visible(False)
        dataPlot['Count'].plot(label=False, ax=ax2)

        ax3 = fig.add_subplot(4, 4, 3)
        dataPlot[['25%', '50%', '75%']].plot(ax=ax3)
        ax3.xaxis.label.set_visible(False)

        axT = fig.add_subplot(4, 4, 4)
        self.Res['Stability'].plot(legend=False, ax=axT)
        # plt.text(self.Res['Stability'].index[0],
        #          self.Res['Stability'].min(),
        #          f"MeanRho: {self.Res['Stability'].mean().round(4)}",
        #          color="r",
        #          fontsize=15,
        #          weight='bold')
        axT.set_ylabel('Stability')
        axT.xaxis.label.set_visible(False)

        ax4 = fig.add_subplot(4, 3, 4)
        exNav.plot(legend=False, ax=ax4)
        plt.text(exNav.index[0],
                 exNav.iloc[-1].min(),
                 IndS['All'].to_string(),
                 color="r",
                 fontsize=15,
                 weight='bold')
        ax4.set_ylabel('EX_NAV')
        ax4.xaxis.label.set_visible(False)

        ax5 = fig.add_subplot(4, 3, 5)
        exNav_3 = (exRet.loc[date3Y:] + 1).cumprod()
        exNav_3.plot(legend=False, ax=ax5)
        plt.text(exNav_3.index[0],
                 exNav_3.iloc[-1].min(),
                 IndS['3Y'].to_string(),
                 color="r",
                 fontsize=15,
                 weight='bold')
        ax5.xaxis.label.set_visible(False)

        ax6 = fig.add_subplot(4, 3, 6)
        exNav_1 = (exRet.loc[date1Y:] + 1).cumprod()
        exNav_1.plot(legend=False, ax=ax6)
        plt.text(exNav_1.index[0],
                 exNav_1.iloc[-1].min(),
                 IndS['1Y'].to_string(),
                 color="r",
                 fontsize=15,
                 weight='bold')
        ax6.xaxis.label.set_visible(False)
        plt.legend(bbox_to_anchor=(1.2, 0), loc=4, ncol=1)

        pos = 8
        for indName, IndValue in IndG.items():
            pos += 1
            ax = fig.add_subplot(4, 4, pos)
            sns.barplot(x='Group', y=indName, hue='Period', data=IndValue, ax=ax)
            ax.legend().set_visible(False)
            ax.xaxis.label.set_visible(False)
        plt.legend(bbox_to_anchor=(1.2, 0), loc=4, ncol=1)

        axEnd = fig.add_subplot(4, 4, 16)
        plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        l1 = self.Res['reg']['Indicators'][['T_abs_mean', 'T_abs_mean_year', 'T_ttest', 'T_ttest_year']]
        r1 = self.Res['IC']['Indicators'][['IC_mean', 'IC_mean_year', 'IC_up_0', 'IC_up_0_year']]
        plt.text(0., 0.1,
                 pd.concat([l1, r1]).round(4).to_string(),
                 color="r",
                 fontsize=22,
                 weight='bold')

        plt.suptitle(f"{self.factName}-{self.hp}days", fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95,
                            bottom=0.05, top=0.95,
                            wspace=0.18, hspace=0.18)
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)

        if kwargs['save']:
            plt.savefig(os.path.join(FPN.factor_test_res.value,
                                     f"{self.factName}_nav-{self.hp}days.png"),
                        dpi=500,
                        bbox_inches='tight')

        plt.show()

    """多路径取平均"""

    # 考虑路径依赖，多路径取平均
    def group_ret(self,
                  data: pd.DataFrame,
                  weight_name: str) -> pd.DataFrame:
        """
        :param data:
        :param weight_name:
        :return:
        """
        group_ = data[SN.GROUP.value].unstack()
        group_ = group_.reindex(
            self.td[(self.td[KN.TRADE_DATE.value] >= group_.index[0]) &
                    (self.td[KN.TRADE_DATE.value] <= group_.index[-1])][KN.TRADE_DATE.value])

        # The average in the group and weighting of out-of-group BenchMark, consider return period
        res_cont_, res = [], {}
        for i in range(0, self.hp):
            groupCopy = group_.copy(deep=True)
            data_ = data.copy(deep=True)

            array1 = np.arange(0, groupCopy.shape[0], 1)
            array2 = np.arange(i, groupCopy.shape[0], self.hp)
            row_ = list(set(array1).difference(array2))

            # 非调仓期填为空值 TODO 调仓期涨跌停收益率标记为空
            groupCopy.iloc[row_] = groupCopy.iloc[row_] * np.nan

            groupCopy = groupCopy.ffill(limit=self.hp - 1) if self.hp != 1 else groupCopy

            # 替换原组别并进行收益率的计算
            data_[SN.GROUP.value] = groupCopy.stack()
            data_ = data_.dropna(subset=[KN.RETURN.value, weight_name])

            group_ret = data_.groupby([KN.TRADE_DATE.value, SN.GROUP.value]).apply(
                lambda x: np.average(x[KN.RETURN.value], weights=x[weight_name]))
            res[i] = group_ret
        # 取平均
        res_df = pd.DataFrame(res).mean(axis=1).unstack()
        res_df.columns = [f'G{int(col_)}' for col_ in res_df.columns]  # rename
        res_df.index = pd.DatetimeIndex(res_df.index)

        return res_df

    # 考虑路径依赖，多路径取平均
    def corr_path(self,
                  data: pd.DataFrame,
                  ) -> pd.Series:

        res_cont_, res = [], 0
        for i in range(0, self.hp):
            array1 = np.arange(i, data.shape[0], self.hp)

            # 非调仓期填为空值
            data_sub = data.iloc[list(array1)].reindex(data.index)
            data_sub = data_sub.ffill(limit=self.hp - 1) if self.hp != 1 else data_sub
            res += data_sub

        res = (res / self.hp).fillna(0)

        return res

    def _reg_fact_ret(self, data_: pd.DataFrame) -> pd.Series(float):
        """
        返回回归类
        """
        data_sub = data_.sort_index()
        weight = data_sub[SN.STOCK_WEIGHT.value]
        d_ = data_sub.loc[:, data_sub.columns != SN.STOCK_WEIGHT.value]
        X = pd.get_dummies(d_.loc[:, d_.columns != KN.RETURN.value], columns=[SN.INDUSTRY_FLAG.value])
        Y = d_[KN.RETURN.value]
        reg = sm.WLS(Y, X, weights=weight).fit()
        if np.isnan(reg.rsquared_adj):
            res = pd.Series(index=['retT', 'fact_ret'])
        else:
            res = pd.Series([reg.tvalues[self.factName], reg.params[self.factName]], index=['retT', 'fact_ret'])
        return res

    def _holding_ret(self, ret: pd.Series) -> pd.Series:
        """
        计算持有不同周期的股票收益率
        :param ret: 股票收益率序列
        :return:
        """

        # Holding period return
        ret = ret.add(1)

        ret_label = 1
        for shift_ in range(self.hp):
            ret_label *= ret.groupby(KN.STOCK_ID.value).shift(- shift_)

        ret_label = ret_label.sub(1)

        return ret_label

    # Series additional written
    def to_csv(self, path: str, file_name: str, data_: pd.Series):
        data_path_ = os.path.join(path, file_name + '.csv')
        data_df = data_.to_frame().T

        header = False if os.path.exists(data_path_) else True

        data_df.to_csv(data_path_, mode='a', header=header)

    @staticmethod
    def weight_cor(data: pd.DataFrame,
                   weight: Union[List, pd.Series, np.arange]) -> float:
        """
        加权相关系数计算：加权协方差和加权方差
        """
        data_array, weight_array = np.array(data.T), np.array(weight)
        # calculate the weighted variance
        cov_weight = np.cov(data_array, aweights=weight_array)
        # calculate the weighted covariance
        var_weight_A = np.cov(data_array[0], aweights=weight_array)
        var_weight_B = np.cov(data_array[-1], aweights=weight_array)
        # calculate the weighted correlation
        corr_weight = cov_weight / pow((var_weight_A * var_weight_B), 0.5)

        return corr_weight[0][1]

    @staticmethod
    def PJCur(x, y) -> float:
        """
        下弯(凹)为负: e ** x，上弯(凸)为正: sqrt(x)
        input  : the coordinate of the three point
        output : the curvature and norm direction
        """
        t_a = np.linalg.norm([x[1] - x[0], y[1] - y[0]])
        t_b = np.linalg.norm([x[2] - x[1], y[2] - y[1]])

        M = np.array([
            [1, t_a, t_a ** 2],
            [1, 0, 0],
            [1, -t_b, t_b ** 2]
        ])

        a = np.matmul(np.linalg.inv(M), x)
        b = np.matmul(np.linalg.inv(M), y)

        kappa = 2 * (a[2] * b[1] - b[2] * a[1]) / (a[1] ** 2. + b[1] ** 2.) ** 1.5
        return kappa


# 多因子相关性分析
class FactorCollinearity(FactorProcess):
    """
    目前只考虑线性相关性

    多因子模型中，按照因子的属性类别，将因子划分为大类内因子和大类间因子，
    一般认为大类内因子具有相同的属性，对个股收益有着相似的解释，
    大类间因子具有不同的因子属性和意义，对个股收益解释不同，
    所以基于此：
    1.大类内因子考虑采用合成的方式对相关性强的因子进行复合
        复合方式：等权法，
                 历史收益率加权法（等权，半衰加权），
                 历史信息比率加权法（等权，半衰加权），
                 最大化复合IC/IC_IR加权，主成分分析等
    2.大类间相关性强的因子考虑采用取舍，剔除相关性强的因子
    注：
    1.对于符号相反的因子采用复合方式合成新因子时，需要调整因子的符号使因子的符号保持一致
    2.采用历史收益率或IC对因子进行加权时，默认情况下认为所有交易日都存在，可考虑对因子和权重进行日期重塑，避免数据错位
    3.在进行因子标准化处理时默认采用多个因子取交集的方式，剔除掉因子值缺失的部分
    """

    factCodeMapping = {}
    codeFactMapping = {}

    def __init__(self):
        super(FactorCollinearity, self).__init__()

        self.dataSet: Dict[str, Any] = {
            "factDirty": None,
            "factClean": None,
        }

    # 原始数据存储
    def set_data(self,
                 factPoolData: Union[pd.DataFrame, None] = None,
                 factWeightData: Union[pd.DataFrame, None] = None,
                 **kwargs):
        """
        数据发生变动则替换原值，否则保持原值
        Parameters
        ----------
        factPoolData : 因子池
        factWeightData: 因子加权系数
        kwargs :

        Returns
        -------

        """
        # 输入空值默认不替换原值
        if factPoolData is not None:
            self.dataSet['factDirty'] = factPoolData
            Col = factPoolData.columns
            self.factCodeMapping = {Col[pos]: chr(ord('A') + pos) for pos in range(len(Col))}
            self.codeFactMapping = {value_: key_ for key_, value_ in self.factCodeMapping.items()}
        if factWeightData is not None:
            self.dataSet['factWeight'] = factWeightData

    def Cleaning(self):
        self.dataSet['factClean'] = self.dataSet['factDirty'].copy()
        # 清洗因子
        # for factName in self.dataSet['factClean'].columns:
        #     self.processSeq(self.dataSet['factClean'], methodN=["RO", "Sta"], dataName=factName)

    # 相关性检验
    def correctionTest(self,
                       plot: bool = True,
                       save: bool = False):
        # 相关性相关指标计算
        corRes = self.processSingle(self.dataSet['factClean'], 'Cor')

        if plot:
            self.plotCor(corRes, save)

        # 数据存储
        if save:
            pass

    def plotCor(self,
                data: Dict[str, pd.DataFrame],
                save: bool = False):
        fig = plt.figure(figsize=(10, 10))
        pos = 1
        for corName, corValue in data.items():
            corValueNew = corValue.rename(columns=self.factCodeMapping, index=self.factCodeMapping)
            ax = fig.add_subplot(2, 2, pos)
            sns.heatmap(corValueNew, annot=True, cmap="YlGnBu", ax=ax)
            ax.set_title(corName)
            pos += 1
        title = "_".join([self.methodP['Cor']['method'], self.methodP['Cor']['p'].get('corName', '4')])
        plt.suptitle(title, fontsize=18)
        if save:
            plt.savefig(os.path.join(FPN.factor_test_res.value, f"{title}.png"),
                        dpi=500,
                        bbox_inches='tight')

        plt.show()

    #  因子合成
    def factorSynthetic(self, **kwargs):
        """
        因子复合需要考虑不同因子在不同的截面数据缺失情况，对于当期存在缺失因子时，复合因子取有效因子进行加权，而不是剔除
        :param kwargs:
        :return:
        """

        comp_factor = self.processSyn(factData=self.dataSet['factClean'],
                                      factWeight=self.dataSet['factWeight'],
                                      **kwargs)
        return comp_factor


if __name__ == '__main__':
    pass
