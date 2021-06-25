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
from typing import List, Union, Dict, Any, Tuple

from DataAPI.LoadData import LoadData
from DataAPI.DataInput.GetData import CSV

from EvaluationIndicitor.Indicator import Indicator

from utility.FactorUtility import MethodSets
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


# 单因子有效性测试-相对因子值分析
class FactorValidityCheckRelative(MethodSets):
    """
    对于单因子的有效性检验，我们从以下几个维度进行考量：
    1.因子暴露稳定性
    截面自相关系数(一般用spearman秩相关)

    2.单因子与下期收益率回归：
        1)因子T值序列绝对值平均值--因子有效性是否显著；
        2)因子T值序列绝对值大于2的占比--因子是否稳定(防止部分异常T值干扰均值)；
        3)因子T值序列均值绝对值除以T值序列的标准差--因子有效性；
        4)因子T值序列的T检验--因子方向性
        5)因子收益率序列平均值--因子方向是否稳定；
        6)因子收益率序列大于0占比

    3.因子IC值：
        1)因子IC值序列的均值大小--判断因子方向是否一致；
        2)因子IC值序列的标准差--因子稳定性；
        3)因子IR比率--因子有效性；
        4)因子IC值序列大于零的占比--判断因子的方向

    4.分层回测检验单调性-打分法：
        将个股按照因子值等数量划分成N组，每组组内按照基准权重进行加权
        对于行业中性化处理，则需要按照行业分组，组内按照因子值分为N组，行业内的各组别按照基准行业权重加权，行业内的各组按照市值加权

        得到N组净值曲线

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
    groupNum = 10
    groupMethod = 'equalNum'

    retName = 'retOpen'

    parent_path = os.path.abspath(os.path.dirname(os.getcwd()))

    industryMapping = {6103: "化工",
                       6102: "采掘",
                       6116: "公用事业",
                       6105: "有色金属",
                       6118: "房地产",
                       6114: "轻工制造",
                       6127: "机械设备",
                       6117: "交通运输",
                       6112: "食品饮料",
                       6101: "农林牧渔",
                       6108: "电子",
                       6109: "交运设备",
                       6124: "建筑材料",
                       6104: "钢铁",
                       6106: "建筑建材",
                       6126: "电气设备",
                       6111: "家用电器",
                       6113: "纺织服装",
                       6115: "医药生物",
                       6120: "商业贸易",
                       6130: "计算机",
                       6134: "非银金融",
                       6123: "综合",
                       6132: "通信",
                       6131: "传媒",
                       6110: "信息设备",
                       6128: "国防军工",
                       6121: "休闲服务",
                       6107: "机械设备",
                       6129: "汽车",
                       6125: "建筑装饰",
                       6122: "信息服务",
                       6119: "金融服务",
                       6133: "银行", }

    def __init__(self):
        super(FactorValidityCheckRelative, self).__init__()

        self.CSV = CSV()

        self.api = LoadData()  # 数据接口

        self.ind = Indicator()  # 评价指标的计算

        self.dataSet = {}  # 原始数据集

        self.Res = defaultdict(dict)  # 因子检验结果

        self.td = self.CSV.trade_date_csv()  # 交易日序列

    # Set params
    def set_params(self, **kwargs):
        super().set_params(**kwargs)

    # Import data from external
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
        self.dataSet['expos'] = pd.merge(FP.reindex(SP.index),
                                         LP,
                                         left_index=True,
                                         right_index=True,
                                         how='left')

        # 3.Synthetic benchmark return
        self.dataSet['BMRet'] = self.api.getBMData(
            benchMarkName='benchmarkZD',
            data=self.dataSet['expos'],
            retName=self.retName,
            wName=SN.STOCK_WEIGHT.value)

        # 4.Filter Up Down Limit Stock
        self.dataSet['ExpCleanNeu'] = self.dataSet['expos'].copy(deep=True)
        self.dataSet['ExpClean'] = self.dataSet['expos'].copy(deep=True)

        # 5.worker factor: Neu factor and Raw factor
        self.processSeq(self.dataSet['ExpClean'], methodN=['RO', 'Sta'], dataName=self.factName)
        self.processSeq(self.dataSet['ExpCleanNeu'], methodN=['RO', 'Neu', 'Sta'], dataName=self.factName)

        # 6.save data
        # self.dataSet['ExpClean'][self.factName] = self.dataSet['factClean'][self.factName]

        # 7.drop Nan by fact value
        self.dataSet["ExpClean"].dropna(subset=[self.factName], inplace=True)
        self.dataSet["ExpCleanNeu"].dropna(subset=[self.factName], inplace=True)

        # 8.Clearing useless data
        for dataName in ['stockPool', 'labelPool', 'factDirty', 'expos']:
            self.dataSet.pop(dataName)

    # Factor validity test
    @timer
    def effectiveness(self,
                      plot: bool = True,
                      save: bool = True):

        # 清空之前的记录
        self.Res.clear()

        dataCleanNeu = self.dataSet['ExpCleanNeu'].copy(deep=True)
        dataClean = self.dataSet['ExpClean'].copy(deep=True)

        # check
        self.fact_stability(dataCleanNeu[self.factName])  # 因子暴露的稳定性
        self.fact_ret(data=dataCleanNeu)  # 因子收益
        self.fact_IC(data=dataCleanNeu)  # 因子与收益相关性
        self.fact_layered(data=dataCleanNeu, flag='neu')  # 因子分层-中性化
        self.fact_layered(data=dataClean, flag='non-neu')  # 因子分层-非中性化

        # plot
        if plot:
            self.plot_res(save=save)
        # if save:
        #     self.res_to_csv()

    # Stability
    def fact_stability(self, data: pd.Series):
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

    # regression: next N day stock ret to factor
    def fact_ret(self,
                 data: pd.DataFrame,
                 **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        data['hpRet'] = self.holding_ret(data[self.retName])
        df_data = data[['hpRet', SN.INDUSTRY_FLAG.value, self.factName, PVN.LIQ_MV.value]]
        df_data = df_data.rename(columns={"hpRet": KN.RETURN.value}).dropna(how='any')

        # Analytic regression result：T Value and Factor Return
        res_reg = df_data.groupby(KN.TRADE_DATE.value).apply(self.fact_ret_reg).dropna(how='all')

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

    # IC
    def fact_IC(self,
                data: pd.DataFrame,
                **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        data['hpRet'] = self.holding_ret(data[self.retName])
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

    # Group
    def fact_layered(self,
                     data: pd.DataFrame,
                     flag: str,
                     **kwargs):
        df_data = data[[self.retName, self.factName, SN.INDUSTRY_FLAG.value, SN.STOCK_WEIGHT.value, PVN.Up_Down.value]]
        df_data = df_data.rename(columns={self.retName: KN.RETURN.value})

        if self.groupMethod == 'equalNum':
            # Grouping
            df_data[SN.GROUP.value] = df_data[self.factName].groupby(KN.TRADE_DATE.value).apply(
                lambda x: pd.cut(x.rank(), bins=self.groupNum, labels=False) + 1)
        elif self.groupMethod == 'equalValue':
            # Grouping
            bins = [df_data[self.factName].quantile(i / 10) for i in range(11)]
            df_data[SN.GROUP.value] = df_data[self.factName].groupby(KN.TRADE_DATE.value).apply(
                lambda x: pd.cut(x, bins=bins, labels=False) + 1)

        # Average group return
        groupRet = self.group_ret(df_data[[KN.RETURN.value, SN.GROUP.value, SN.STOCK_WEIGHT.value]],
                                  weight_name=SN.STOCK_WEIGHT.value)
        ################################################################################################################
        # NAV: Continuous and uninterrupted, unfilled Nav
        groupRet = pd.merge(groupRet,
                            self.dataSet['BMRet'].data,
                            left_index=True,
                            right_index=True,
                            how='left').dropna()
        nav = groupRet.add(1).cumprod()
        ex_nav = nav.div(nav[self.dataSet['BMRet'].data_name], axis=0).drop(columns=self.dataSet['BMRet'].data_name)

        # 计算指标
        self.layered_ind(ex_nav, nav, flag)

        self.Res[f'Layered_{flag}'].update({"dataMerge": df_data, "exNav": ex_nav})

    """
    指标的计算和画图：这两个function请不要动他谢谢! =_=
    """

    # cal ind
    def layered_ind(self,
                    ex_nav: pd.DataFrame,
                    nav: pd.DataFrame,
                    flag: str):
        # 变量集合
        Var = []
        # ret days
        ex_ret = ex_nav.pct_change(fill_method=None).dropna()
        ret = nav.pct_change(fill_method=None).dropna()
        # Date notes
        lastDate = ex_nav.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        mapping = {"All": ":",
                   "1Y": "date1Y:",
                   "3Y": "date3Y:"
                   }

        # 基础指标的计算:年化收益率，年化波动率，年化夏普比，最大回撤
        indMethod = [self.ind.return_a, self.ind.std_a, self.ind.shape_a, self.ind.max_retreat]
        for v_ in ['All', '1Y', '3Y']:  # , '1T3Y', '3YBef'
            retSub = eval(f"ret.loc[{mapping[v_]}]")
            ex_navSub = eval(f"ex_nav.loc[{mapping[v_]}]")
            ex_retSub = eval(f"ex_ret.dropna().loc[{mapping[v_]}]")
            ind = ex_navSub.agg(indMethod)

            # 曲率
            Cur = ex_navSub.resample("M").last().rolling(3).apply(lambda y: self.PJ_Cur([1, 2, 3], y.to_list()))
            ind.loc['Cur'] = Cur.mean()
            # 捕获率
            BMRet = eval(f"retSub[self.dataSet['BMRet'].data_name].loc[{mapping[v_]}]")
            retUp = (1 + retSub[BMRet > 0]).prod() ** (244 / retSub.shape[0]) - 1
            retDown = (1 + retSub[BMRet < 0]).prod() ** (244 / retSub.shape[0]) - 1
            CapUp = retUp.div(retUp[self.dataSet['BMRet'].data_name], axis=0)
            CapDown = retDown.div(retDown[self.dataSet['BMRet'].data_name], axis=0)

            CapRatio = pd.concat([CapUp, CapDown, CapUp / CapDown], axis=1)
            CapRatio.columns = ['CapUp', 'CapDown', 'Cap']
            CapRatio = CapRatio.T.drop(columns=BMRet.name)
            ind = ind.append(CapRatio)
            # 衰减
            ind.loc['decay'] = ex_navSub.resample("M").agg(self.ind.acc_return).diff(1).mean()
            # 单调性
            Mon = ex_navSub.resample(f"{self.hp}D").last().corrwith(pd.Series([i for i in range(1, self.groupNum + 1)],
                                                                              index=ex_navSub.columns),
                                                                    axis=1,
                                                                    method='spearman').mean()
            # 单因素方差分析
            FTest, F_PValue = stats.f_oneway(*ex_retSub.dropna().values.T)

            # 数据合并
            ind.loc['Mon'] = Mon
            ind.loc['FTest'] = FTest
            ind.loc['F_PValue'] = F_PValue
            ind.index.name = 'Ind'
            ind['Period'] = v_
            ind = ind.set_index('Period', append=True).swaplevel(0, 1, axis=0)
            Var.append(ind)

        res = pd.concat(Var)
        self.Res[f"Layered_{flag}"]["Indicators"] = res

    # Plot
    def plot_res(self, **kwargs):
        """
        Don't edit it
        Parameters
        ----------
        kwargs :

        Returns
        -------

        """

        factExp = self.dataSet['ExpClean'][self.factName]
        exNav = self.Res['Layered_non-neu']['exNav']
        exRet = exNav.pct_change(fill_method=None).dropna()

        exNavNeu = self.Res['Layered_neu']['exNav']
        exRetNeu = exNavNeu.pct_change(fill_method=None).dropna()

        Ind = self.Res['Layered_non-neu']['Indicators']
        IndNeu = self.Res['Layered_neu']['Indicators']
        lastDate = exNavNeu.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        mapping = {"All": ":",
                   "1Y": "date1Y:",
                   "3Y": "date3Y:"
                   }
        factExp_df = factExp.unstack()
        dataPlot = pd.DataFrame(
            {
                "Count": factExp_df.count(axis=1),
                "25%": factExp_df.quantile(0.25, axis=1),
                "50%": factExp_df.quantile(0.5, axis=1),
                "75%": factExp_df.quantile(0.75, axis=1),
            })

        fig = plt.figure(figsize=(20, 16))

        ax1 = fig.add_subplot(4, 4, 1)
        factExp.plot.hist(bins=30, ax=ax1)

        ax2 = fig.add_subplot(4, 4, 2)
        ax2.xaxis.label.set_visible(False)
        ax2.set_ylabel('Degree of Missing')
        dataPlot['Count'].plot(label=False, ax=ax2)

        ax3 = fig.add_subplot(4, 4, 3)
        ax3.set_ylabel('Quantile')
        ax3.xaxis.label.set_visible(False)
        dataPlot[['25%', '50%', '75%']].plot(ax=ax3)

        ax4 = fig.add_subplot(4, 4, 4)
        ax4.set_ylabel('Stability')
        ax4.xaxis.label.set_visible(False)
        self.Res['Stability'].plot(legend=False, ax=ax4)

        #  ex-nav non neu
        pos = 4
        for p, sign in mapping.items():
            # 未中性化
            exNav = eval(f"(exRet.loc[{sign}] + 1).cumprod()")
            text = Ind.loc[p].loc[['Mon', 'F_PValue']]['G1'].round(4)
            text.index.name = ''

            ax = fig.add_subplot(4, 3, pos)
            exNav.plot(legend=False, ax=ax)
            plt.text(exNav.index[0],
                     exNav.iloc[-1].min(),
                     text.to_string(),
                     color="r",
                     fontsize=15,
                     weight='bold')
            ax.set_ylabel(f'EX_NAV_{p}')
            ax.xaxis.label.set_visible(False)

            # 中性化
            exNavNeu = eval(f"(exRetNeu.loc[{sign}] + 1).cumprod()")
            textNeu = IndNeu.loc[p].loc[['Mon', 'F_PValue']]['G1'].round(4)
            textNeu.index.name = ''

            ax = fig.add_subplot(4, 3, pos + 3)
            exNavNeu.plot(legend=False, ax=ax)
            plt.text(exNavNeu.index[0],
                     exNavNeu.iloc[-1].min(),
                     textNeu.to_string(),
                     color="r",
                     fontsize=15,
                     weight='bold')
            ax.set_ylabel(f'EX_NAV_NEU_{p}')
            ax.xaxis.label.set_visible(False)
            pos += 1
        plt.legend(bbox_to_anchor=(1.2, 0), loc=4, ncol=1)

        # 捕获率
        x = np.arange(len(IndNeu.columns))
        width = 0.6
        ax5 = fig.add_subplot(4, 4, 13)
        flag = 1
        for p in mapping.keys():
            ax5.bar(x + (flag - 2) * (width / 3), IndNeu.loc[(p, 'CapUp')], 0.2)
            ax5.bar(x + (flag - 2) * (width / 3), - IndNeu.loc[(p, 'CapDown')], 0.2)
            ax5.scatter(x + (flag - 2) * (width / 3), IndNeu.loc[(p, 'Cap')], marker='*', zorder=2)
            flag += 1
        ax5.set_ylabel('Cap')
        ax5.set_xticks(x)
        ax5.set_xticklabels(IndNeu.columns)

        IndNeu = IndNeu.reset_index('Period')
        # 收益率衰减度
        decay = IndNeu.loc['decay'].set_index('Period').stack().reset_index()
        decay.columns = ['Period', 'Group', 'decay']
        ax6 = fig.add_subplot(4, 4, 14)
        sns.barplot(x='Group', y='decay', hue='Period', data=decay, ax=ax6)
        ax6.legend().set_visible(False)
        ax6.xaxis.label.set_visible(False)

        # 年化收益
        return_a = IndNeu.loc['return_a'].set_index('Period').stack().reset_index()
        return_a.columns = ['Period', 'Group', 'return_a']
        ax7 = fig.add_subplot(4, 4, 15)
        sns.barplot(x='Group', y='return_a', hue='Period', data=return_a, ax=ax7)
        ax7.legend().set_visible(False)
        ax7.xaxis.label.set_visible(False)

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

        title = f"{self.factName}-{self.hp}days"

        plt.suptitle(title, fontsize=20)
        plt.subplots_adjust(left=0.05, right=0.95,
                            bottom=0.05, top=0.95,
                            wspace=0.18, hspace=0.18)
        fig.tight_layout(pad=0, w_pad=0, h_pad=0)

        if kwargs['save']:
            plt.savefig(os.path.join(FPN.Fact_testRes.value,
                                     f"{self.factName}_nav-{self.hp}days.png"),
                        dpi=100,
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
        for i in range(self.hp):
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

    def fact_ret_reg(self, data_: pd.DataFrame) -> pd.Series(float):
        """
        返回回归结果
        流通市值平方根倒数加权去除异方差问题
        """
        data_sub = data_.sort_index()
        weight = 1 / data_sub[PVN.LIQ_MV.value]
        d_ = data_sub.loc[:, data_sub.columns != SN.STOCK_WEIGHT.value]
        X = pd.get_dummies(d_.loc[:, d_.columns != KN.RETURN.value], columns=[SN.INDUSTRY_FLAG.value])
        Y = d_[KN.RETURN.value]
        reg = sm.WLS(Y, X, weights=weight).fit()
        if np.isnan(reg.rsquared_adj):
            res = pd.Series(index=['retT', 'fact_ret'])
        else:
            res = pd.Series([reg.tvalues[self.factName], reg.params[self.factName]], index=['retT', 'fact_ret'])
        return res

    def holding_ret(self, ret: pd.Series) -> pd.Series:
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

    def res_to_csv(self):
        path1 = os.path.join(FPN.Fact_testRes.value, f'FactRet{os.sep}{self.factName}.csv')
        path2 = os.path.join(FPN.Fact_testRes.value, f'FactInd{os.sep}Indicators.csv')
        path3 = os.path.join(FPN.Fact_testRes.value, f'FactInd{os.sep}{self.factName}_group.csv')

        factInd1 = pd.concat([self.Res['reg']['res_reg'],
                              self.Res['IC']['IC_rank']], axis=1).reset_index()

        factInd2 = pd.concat([self.Res['reg']['Indicators'],
                              self.Res['IC']['Indicators'],
                              pd.Series(self.Res['Stability'].mean(), index=['StabilityMean'])])

        factInd3 = self.Res['Layered_neu']['Indicators']

        factInd2 = factInd2.to_frame(self.factName).T
        header = False if os.path.exists(path2) else True

        factInd1.to_csv(path1, index=False)
        factInd2.to_csv(path2, mode='a', header=header)
        factInd3.to_csv(path3)

    # Series additional written
    @staticmethod
    def to_csv(path: str, file_name: str, data_: pd.Series):
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
    def PJ_Cur(x, y) -> float:
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
class FactorCollinearity(MethodSets):
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
                 最大化复合IC/IC_IR加权，
                 最小因子波动法
                 主成分分析
                 聚类
    2.大类间相关性强的因子考虑采用取舍，剔除相关性强的因子

    因子合成效果初检验：
    因子之间的相关性走势图

    注：
    1.对于符号相反的因子采用复合方式合成新因子时，需要调整因子的符号使因子的符号保持一致
    2.采用历史收益率或IC对因子进行加权时，默认情况下认为所有交易日都存在，可考虑对因子和权重进行日期重塑，避免数据错位
    3.在进行因子标准化处理时默认采用多个因子取交集的方式，剔除掉因子值缺失的部分
    """

    factCodeMapping = {}
    codeFactMapping = {}

    def __init__(self):
        super(FactorCollinearity, self).__init__()
        self.rp = 20  # 滚动窗口
        self.hp = 5  # 持有周期

        self.synName = 'Syn'  # 合成因子名称
        self.dataSet: Dict[str, Any] = {
            "factDirty": None,
            "factClean": None,
        }

        self.Res: Dict[str, Any] = {}  # 结果

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
                       save: bool = True):
        # 相关性相关指标计算
        corRes = self.processSingle(self.dataSet['factClean'], 'Cor')

        # 更新数据：相关性检验结果
        self.Res.update(corRes)

        if plot:
            self.plot_cor(corRes, save)

        # 数据存储
        if save:
            pass

    #  因子合成
    def factor_Synthetic(self,
                         plot: bool = True,
                         save: bool = True,
                         **kwargs):
        """
        因子复合需要考虑不同因子在不同的截面数据缺失情况，对于当期存在缺失因子时，复合因子取有效因子进行加权，而不是剔除
        考虑持仓周期，当期因子合成所需权重为hp + 1日前信息: 因为IC权重的计算需要用到未来数据
        :param plot:
        :param save:
        :param kwargs:
        :return:
        """

        dataFact = self.dataSet['factClean'].copy()
        # history weight--label
        dataWeight = self.dataSet['factWeight'].shift(self.hp + 1)
        dateList = dataWeight.index.drop_duplicates().sort_values()

        Syn = {}  # TODO check
        for winEnd in range(self.rp + self.hp, len(dateList)):  # Star from first effective data
            factSub = dataFact.loc[dateList[winEnd]]
            weightSub = dataWeight.loc[dateList[winEnd - self.rp + 1: winEnd + 1]]

            comp_factor = self.processSingle(data=factSub,
                                             weight=weightSub,
                                             methodN='Syn',
                                             **kwargs)
            Syn[dateList[winEnd]] = comp_factor
        self.Res['factSyn'] = pd.DataFrame(Syn).T.stack()
        self.Res['factSyn'].name = self.synName
        self.Res['factSyn'].index.names = ['date', 'code']

        if plot:
            self.plot_Syn(save)

        return self.Res['factSyn']

    # 与原因子相关性走势图
    def plot_Syn(self, save: bool = True):
        """
        合成因子与子因子相关性走势图
        """
        dataRaw = self.dataSet['factClean'].copy()
        SynFactor = self.Res['factSyn'].copy()
        factorS = pd.merge(dataRaw, SynFactor, left_index=True, right_index=True, how='left').dropna()
        df_cor = factorS.groupby(KN.TRADE_DATE.value).corr(method='spearman')

        df_cor.index.names = ['date', 'factor']
        df_cor_N = df_cor[self.synName].reset_index()
        df_cor_N = df_cor_N[df_cor_N['factor'] != self.synName]
        df_cor_N['date'] = pd.to_datetime(df_cor_N['date'])

        g = sns.FacetGrid(df_cor_N, row="factor", height=2, aspect=6)
        g.map_dataframe(sns.lineplot, x="date", y=self.synName)
        g.set_axis_labels("date", "cor_spearman")
        plt.suptitle(self.synName)
        g.tight_layout()

        if save:
            plt.savefig(os.path.join(FPN.Fact_corrRes.value, f"{self.synName}_pairCor_spearman.png"),
                        dpi=100,
                        bbox_inches='tight')
        plt.show()

    # Plot Cor
    def plot_cor(self,
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
        title = "_".join([self.methodProcess['Cor']['method'], self.methodProcess['Cor']['p'].get('corName', '4')])
        plt.suptitle(title, fontsize=18)
        if save:
            plt.savefig(os.path.join(FPN.Fact_corrRes.value, f"{title}.png"),
                        dpi=500,
                        bbox_inches='tight')
        plt.show()

    def save_factor(self):
        pass


if __name__ == '__main__':
    pass
