import pandas as pd
import numpy as np
import warnings
import traceback
import time
import os
import json
import statsmodels.api as sm
from scipy import stats
import collections
from typing import Iterable
from functools import reduce
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import datetime as dt
from typing import List, Any, Union

from DataBase import database_manager
from Object import (
    FactorInfo,
    GroupData,
    FactorData,
    FactorRetData,
    send_email
)
from Data.GetData import SQL

from FactorCalculation import FactorPool
from LabelPool.Labelpool import LabelPool
from StockPool.StockPool import StockPool

from FactorProcess.FactorProcess import FactorProcess, Multicollinearity
from FactorCalculation.FactorBase import FactorBase
from EvaluationIndicitor.Indicator import Indicator

from constant import (
    timer,
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN,
    FactorCategoryName as FCN
)

warnings.filterwarnings(action='ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示问题-设置字体为黑体
plt.rcParams['font.serif'] = ['simhei']
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

sns.set(font_scale=1.1)
sns.set_style("darkgrid", {"font.sans-serif": ['simhei', 'Droid Sans Fallback']})

"""
缺失值处理，需要提前定义好子样本
"""


# 数据传输转移到这
class LoadData(object):
    factor_pool_path = 'A:\\数据\\FactorPool'

    def __init__(self):
        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池

    def factor_to_csv(self, factor: pd.Series, folder_name: str):
        file_name = self.Factor.factor[factor.name].__self__.__name__
        path_ = os.path.join(os.path.join(self.factor_pool_path, folder_name), file_name + '.csv')
        # 追加写入
        factor.to_csv(path_, mode='a')


# 单因子有效性测试
class FactorValidityCheck(object):
    """
    对于单因子的有效性检验，我们从以下几个维度进行考量：
    1.单因子与下期收益率回归：
        1)因子T值序列绝对值平均值--因子有效性是否显著；
        2)因子T值序列绝对值大于2的占比--因子是否稳定；
        3)因子T值序列均值绝对值除以T值序列的标准差--因子有效性；
        4)因子收益率序列平均值--因子方向是否一致；
        5)因子收益率序列平均值零假设T检验--因子收益率是否显著不为零
    2.因子IC值：
        1)因子IC值序列的均值大小--判断因子方向是否一致；
        2)因子IC值序列绝对值均值大小--因子有效性；
        3)因子IC值序列的标准差--因子稳定性；
        4)因子IR比率--因子有效性；
        5)因子IC值累积曲线--随时间变化效果是否稳定
        6)因子IC值序列大于零的占比--判断因子效果的一致性
    3.分层回测检验单调性-打分法：
        按照基准权重合成各组净值曲线

        普通检验指标
        因子方向：最优组减去最劣组
        时间区间划分：整个回溯区间，近一年，近三年到近一年，三年前
        1)年化收益率；
        2)年化波动率；
        3)夏普比率；
        4)最大回撤；
        5)胜率

        方差分析分月度和年度
        单调性检验：差分，IC
        稳定性指标：相关性，IR
        Top组指标
        复杂度指标

    """

    industry_list = ["CI005001.WI",
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

    fact_name = None

    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self, hp: int = 1, stock_pool: str = 'StockPool_all_market', label_pool: str = 'LabelPool'):

        self.db = database_manager
        self.Q = SQL()

        self.sp_name = stock_pool
        self.lp_name = label_pool
        self.hp = hp  # 因子调仓周期

        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池

        self.factProc = FactorProcess()  # 因子预处理

        self.ind = Indicator()  # 评价指标的计算

        self.dataSet = collections.defaultdict(dict)  # 输入数据

        self.Res = collections.defaultdict(dict)  # 因子检验结果

        self.factor_mapping = self._factor_mapping()

        self.neu = 'non_neu'  # 中性化

        self.td = self.Q.trade_date_csv()  # 交易日序列

    # factor Chinese-English mapping
    def _factor_mapping(self, file_name: str = 'factor_name.json'):
        try:
            file_path = os.path.join(self.parent_path, file_name)
            infile = open(file_path, 'r', encoding='utf-8')
            res = json.load(infile)
        except Exception as e:
            print(f"read json file failed, error\n{traceback.format_exc()}")
            res = {}
        return res

    # load stock pool and label pool
    @timer
    def load_pool_data(self):
        """
        load stock pool and label pool data
        """
        # Load stock pool
        if self.sp_name == '':
            print(f"{dt.datetime.now().strftime('%X')}: Can not load stock pool!")
        else:
            print(f"{dt.datetime.now().strftime('%X')}: Loading stock pool!")
            stock_pool_method = self.Stock.__getattribute__(self.sp_name)
            effect_stock = stock_pool_method()
            self.dataSet['stockPool'] = effect_stock

        # Load label pool
        if self.lp_name == '':
            print(f"{dt.datetime.now().strftime('%X')}: Can not load label pool!")
        else:
            print(f"{dt.datetime.now().strftime('%X')}: Loading label pool!")
            label_pool_method = self.Label.__getattribute__(self.lp_name)
            stock_label = label_pool_method()
            self.dataSet['labelPool'] = stock_label

    # load factor
    @timer
    def load_factor(self,
                    fact_name: str,
                    **kwargs
                    ):
        """
        优先直接获取数据--否则数据库调取--最后实时计算
        :param fact_name:
        :param kwargs:
        :return:
        """
        if kwargs.get('factor_value', None) is None:
            if kwargs['cal']:
                try:
                    factRawData = self.Factor.factor[fact_name + '_data_raw'](**kwargs['factor_params'])
                    self.dataSet["factRawData"] = factRawData
                except Exception as e:
                    print(f"{dt.datetime.now().strftime('%X')}: "
                          f"Unable to load raw data that to calculate factor!\n{traceback.format_exc()}")
                    factor_class = FactorInfo()
                else:
                    factor_class = self.Factor.factor[fact_name](
                        data=self.dataSet["factRawData"].copy(),
                        **kwargs['factor_params'])
            else:
                factor_data_ = self.db.query_factor_data(factor_name=fact_name, db_name=kwargs['db_name'])

                print(f"{dt.datetime.now().strftime('%X')}: Get factor data from MySQL!")
                factor_data_ = factor_data_.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

                factor_class = FactorInfo()
                factor_class.data = factor_data_[fact_name]
                factor_class.factor_name = fact_name
        else:
            print(f"{dt.datetime.now().strftime('%X')}: Get factor data from input!")
            kwargs['factor_value'] = kwargs['factor_value'].set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            factor_class = FactorInfo()
            factor_class.data = kwargs['factor_value'][fact_name]
            factor_class.factor_name = fact_name

        self.fact_name = factor_class.factor_name
        self.dataSet['factDirty'][self.fact_name] = factor_class

    def process_factor(self,
                       data: pd.Series,
                       outliers: str,
                       neutralization: str,
                       standardization: str):
        """
        :param data:
        :param outliers: 异常值处理
        :param neutralization: 中性化处理
        :param standardization: 标准化处理

        :return:
        """
        if data is None:
            fact_value = self.dataSet['factDirty'][self.fact_name].data.copy(deep=True)  # 获取因子数据
            fact_value = fact_value[self.fact_name] if isinstance(fact_value, pd.DataFrame) else fact_value
        else:
            fact_value = data.copy(deep=True)

        if fact_value is None:
            print("factor data is None!")
            return

        # pre-processing factors
        if outliers + neutralization + standardization == '':
            self.dataSet["factClean"][self.fact_name] = fact_value
            self.neu = 'non_neu'
        else:
            try:
                if outliers != '':
                    print(f"{dt.datetime.now().strftime('%X')}: processing outlier")
                    fact_value = self.factProc.remove_outliers(fact_value, outliers)
                if neutralization != '':
                    print(f"{dt.datetime.now().strftime('%X')}: neutralization")
                    if 'mv' in neutralization:
                        self.factProc.raw_data['mv'] = self.dataSet['ExpRaw'][PVN.LIQ_MV.value]
                    if 'industry' in neutralization:
                        self.factProc.raw_data['industry'] = self.dataSet['ExpRaw'][SN.INDUSTRY_FLAG.value]

                    fact_value = self.factProc.neutralization(fact_value, neutralization)
                    self.neu = 'neu'
                if standardization != '':
                    print(f"{dt.datetime.now().strftime('%X')}: standardization")
                    if 'mv' in standardization:
                        self.factProc.raw_data['mv'] = self.dataSet['ExpRaw'][PVN.LIQ_MV.value]
                    fact_value = self.factProc.standardization(fact_value, standardization)
                self.dataSet["factClean"][self.fact_name] = fact_value

            except Exception as e:
                print(f"{dt.datetime.now().strftime('%X')}: "
                      f"pre-processing factors error!\n{traceback.format_exc()}")

    # Data Integration
    @timer
    def integration(self,
                    outliers: str,
                    neu: str,
                    stand: str,
                    ):
        """
        1.输入标签可能存在空值，最后结果保存时进行了以因子暴露值为基准的去空处理，
        在后续研究分析中因子暴露值可以保证不为空，但标签依然会存在空值；
        2.基准的合成才用最原始的股票池进行合成；
        3.涨跌停仅在开仓时需要处理，开仓时涨跌停不进行建仓处理，但该股在分析时依然占有一定的权重

        :param outliers: 异常值处理
        :param neu: 中心化处理
        :param stand: 标准化处理
        :return:
        """
        # 1.Get data
        SP, LP = self.dataSet.get("stockPool", None), self.dataSet.get('labelPool', None)
        FP = self.dataSet['factDirty'].get(self.fact_name, type(self.fact_name, (), {"data": None})).data

        # 2.Merge factor value and label pool on factor value that after stock pool filter
        self.dataSet['ExpRaw'] = pd.merge(FP[SP.index], LP, left_index=True, right_index=True, how='left')

        # 3.Synthetic benchmark return
        self.dataSet['BMRet'] = self.Label.bm_labels(
            data=self.dataSet['ExpRaw'][['retOpen', SN.STOCK_WEIGHT.value]],
            ret_name='retOpen',
            weight_name=SN.STOCK_WEIGHT.value,
            bm_name='benchmarkZD')

        # 4.Filter Up Down Limit Stock
        # self.dataSet['Exposure'] = self.dataSet['exposureRaw'][self.dataSet['exposureRaw'][PVN.Up_Down.value]]
        self.dataSet['ExpClean'] = self.dataSet['ExpRaw']

        # 5.process factor
        self.process_factor(data=self.dataSet['ExpClean'][self.fact_name],
                            outliers=outliers,
                            neutralization=neu,
                            standardization=stand)

        # 6.save data
        self.dataSet['ExpClean'][self.fact_name] = self.dataSet['factClean'][self.fact_name]

        # 7.drop Nan by fact value
        self.dataSet['ExpClean'] = self.dataSet['ExpClean'].dropna(subset=[self.fact_name])

    # Factor validity test
    @timer
    def effectiveness(self,
                      ret_name: str = PVN.OPEN.value,
                      group_num: int = 5,
                      plot: bool = True,
                      save: bool = True):
        """
        """
        data_clean = self.dataSet['ExpClean']
        fact_exposure = data_clean[self.fact_name]
        stock_return = data_clean[KN.RETURN.value + ret_name.capitalize()]
        stock_return.name = KN.RETURN.value
        industry_exposure = data_clean[SN.INDUSTRY_FLAG.value]
        stock_w = data_clean[SN.STOCK_WEIGHT.value]
        price_limit = data_clean[PVN.Up_Down.value]

        # 检验
        try:
            # self.factor_ret(fact_exp=fact_exposure,
            #                 stock_ret=stock_return,
            #                 ind_exp=industry_exposure,
            #                 stock_w=stock_w)
            #
            # self.IC_IR(fact_exp=fact_exposure,
            #            stock_ret=stock_return,
            #            stock_w=stock_w)

            self.monotonicity(fact_exp=fact_exposure,
                              stock_ret=stock_return,
                              ind_exp=industry_exposure,
                              stock_w=stock_w,
                              group_num=group_num,
                              up_down=price_limit)
        except Exception as e:
            print(f"{traceback.format_exc()}")
        else:
            # plot
            if plot:
                self.plotRes(save=save)
            # indicate and plot
            # reg_res = self.Res[self.fact_name].get('reg', None)
            # IC_res = self.Res[self.fact_name].get('IC', None)
            # group_res = self.Res[self.fact_name].get('Group', None)
            #
            # if reg_res is None and IC_res is None and group_res is None:
            #     return
            #
            # if save:
            #     reg_ind = reg_res['indices']
            #     IC_ind = IC_res['indices']
            #     group_ind = group_res['indices']
            #     res = pd.concat([reg_ind, IC_ind, group_ind])
            # if plot and save:
            #     p1 = reg_res['path_ret']
            #     p2 = IC_res['path_IC']
            #     p3 = group_res['nav_year']
            #     p4 = group_res['ex_nav_year']
            #     p5 = group_res['ind']
            #
            # if eff1 is not None and eff2 is not None and eff3 is not None and save:
            #     # eff1.name = eff1.name + f'_{hp}days'
            #     # eff2.name = eff2.name + f'_{hp}days'
            #     # eff3.name = eff3.name + f'_{hp}days'
            #
            #     if self.neu == 'neu':
            #         self.to_csv(FPN.factor_test_res.value, 'Correlation_neu', pd.concat([eff1, eff2, eff3['Judge']]))
            #         self.to_csv(FPN.factor_test_res.value, 'Group_neu', eff3['ind'])
            #     else:
            #         self.to_csv(FPN.factor_test_res.value, 'Correlation', pd.concat([eff1, eff2, eff3['Judge']]))
            #         self.to_csv(FPN.factor_test_res.value, 'Group', eff3['ind'])

    # 单因子与下期收益率回归
    def factor_ret(self,
                   fact_exp: pd.Series,
                   stock_ret: pd.Series,
                   ind_exp: pd.DataFrame,
                   stock_w: pd.Series,
                   **kwargs):
        """

        :param fact_exp: 因子暴露
        :param stock_ret: 个股收益标签
        :param ind_exp:行业暴露
        :param stock_w:个股权重
        :param kwargs:
        :return:
        """

        # Calculate stock returns for different holding periods and generate return label
        ret_label = self._holding_ret(stock_ret)

        df_data = pd.concat([ret_label, ind_exp, fact_exp, stock_w], axis=1, join='inner')

        # Analytic regression result：T Value and Factor Return
        res_reg = df_data.groupby(KN.TRADE_DATE.value).apply(self._reg_fact_ret, 150)
        res_reg = res_reg.dropna(how='all')
        if res_reg.empty:
            print(f"{self.fact_name}因子每期有效样本量不足150，无法检验！")
            return

        # get Trade date
        res_reg = res_reg.reindex(self.td[(self.td[KN.TRADE_DATE.value] >= res_reg.index[0]) &
                                          (self.td[KN.TRADE_DATE.value] <= res_reg.index[-1])][KN.TRADE_DATE.value])

        # Calculate Indicators
        T_mean = res_reg['T'].mean()
        T_abs_mean = abs(res_reg['T']).mean()
        T_abs_up_2 = res_reg['T'][abs(res_reg['T']) > 2].count() / res_reg.dropna().shape[0]
        T_stable = abs(res_reg['T'].mean()) / res_reg['T'].std()

        fact_ret_mean = res_reg['fact_ret'].mean()
        ret_ttest = stats.ttest_1samp(res_reg['fact_ret'].dropna(), 0)

        # 最近一年表现
        T_year = res_reg['T'][-244:]
        T_abs_mean_year = abs(T_year).mean()
        T_abs_up_2_year = T_year[abs(T_year) > 2].count() / T_year.dropna().shape[0]
        T_stable_year = abs(T_year.mean()) / T_year.std()

        indicators = pd.Series([T_abs_mean, T_abs_up_2, T_mean, T_stable,
                                T_abs_mean_year, T_abs_up_2_year, T_stable_year,
                                fact_ret_mean, ret_ttest[0]],
                               index=['T_abs_mean', 'T_abs_up_2', 'T_mean', 'T_stable',
                                      'T_abs_mean_year', 'T_abs_up_2_year', 'T_stable_year',
                                      'fact_ret', 'fact_ret_t'],
                               name=self.fact_name)

        # 因子收益路径依赖处理
        fact_ret_path = self.corr_path(res_reg['fact_ret'])

        # plot
        # self.plot_return(fact_ret=fact_ret_path)

        # save data to dict
        self.Res[self.fact_name]['reg'] = {"res_reg": res_reg,
                                           "indices": indicators,
                                           "path_ret": fact_ret_path}

        # return indicators

    # 因子IC值
    def IC_IR(self,
              fact_exp: pd.Series,
              stock_ret: pd.Series,
              stock_w: pd.Series,
              **kwargs):

        # Calculate stock returns for different holding periods and generate return label
        ret_label = self._holding_ret(stock_ret)

        df_data = pd.concat([ret_label, fact_exp, stock_w], axis=1, join='inner').dropna()

        # TODO 再测试一次
        IC_rank = df_data.groupby(KN.TRADE_DATE.value).apply(
            lambda x: self.weight_cor(x[[self.fact_name, KN.RETURN.value]].rank(), x[SN.STOCK_WEIGHT.value]))

        IC_rank = IC_rank.reindex(self.td[(self.td[KN.TRADE_DATE.value] >= IC_rank.index[0]) &
                                          (self.td[KN.TRADE_DATE.value] <= IC_rank.index[-1])][KN.TRADE_DATE.value])

        IC_mean = IC_rank.mean()
        IC_std = IC_rank.std()

        IR = IC_mean / IC_std * pow(244 / self.hp, 0.5)
        IC_up_0 = len(IC_rank[IC_rank > 0]) / IC_rank.dropna().shape[0]
        # IC_cum = IC_rank.fillna(0).cumsum() / self.hp

        IC_year = IC_rank[-244:]
        IC_mean_year = IC_year.mean()
        IC_mean_std = IC_year.std()
        IR_year = IC_mean_year / IC_mean_std * pow(244 / self.hp, 0.5)

        indicators = pd.Series([IC_mean, IC_std, IR, IC_up_0,
                                IC_mean_year, IC_mean_std, IR_year],
                               index=['IC_mean', 'IC_std', 'IR', 'IC_up_0',
                                      'IC_mean_year', 'IC_mean_std', 'IR_year'],
                               name=self.fact_name)
        # Path dependence
        IC_path = self.corr_path(IC_rank)

        # save data to dict
        self.Res[self.fact_name]['IC'] = {"IC_rank": IC_rank,
                                          "indices": indicators,
                                          "path_IC": IC_path}

        # # plot
        # self.plot_IC(IC=IC_path)

        # return indicators

    # 分层回测检验
    def monotonicity(self,
                     fact_exp: pd.Series,
                     stock_ret: pd.Series,
                     ind_exp: pd.DataFrame,
                     stock_w: pd.Series,
                     group_num: int = 5,
                     **kwargs):
        """
        # :param benchmark:
        :param fact_exp:
        :param stock_ret:
        :param ind_exp:
        :param stock_w:
        :param group_num: 分组数量
        :return:
        """
        labelList = [labelValue for labelValue in kwargs.values()]

        df_data = pd.concat([stock_ret, fact_exp, ind_exp, stock_w] + labelList, axis=1, join='inner')

        # Grouping
        df_data[SN.GROUP.value] = df_data[self.fact_name].groupby(KN.TRADE_DATE.value).apply(
            lambda x: pd.cut(x.rank(), bins=group_num, labels=False) + 1)

        # Average group return
        groupRet = self.group_ret(df_data, weight_name=stock_w.name)
        ################################################################################################################
        # NAV: Continuous and uninterrupted
        groupRet = pd.merge(groupRet, self.dataSet['BMRet'], left_index=True, right_index=True, how='left').dropna()
        nav = groupRet.add(1).cumprod(axis=0)
        ex_nav = nav.div(nav['BM'], axis=0).drop(columns='BM')
        nav.to_csv(f"C:\\Users\\Administrator\\Desktop\\Test\\{self.fact_name}.csv")
        # 计算指标
        self.groupIndexCal(ex_nav, freq='D')

        self.dataSet['Group'] = {"dataMerge": df_data,
                                 "exNav": ex_nav}
        # # 最近一年超额收益单调性，多头超额收益,多头超额收益信息比和最强组各行业表现(超额收益最大组)  TODO
        # g_ret_year = ex_nav.pct_change().iloc[-244:, :]
        # g_nav_year = (g_ret_year + 1).prod()
        #
        # Monotony = np.corrcoef((g_nav_year - 1).rank(), [i for i in range(1, group_num + 1)])[0][1]
        #
        # long_g_name = g_nav_year.idxmax()
        # long_ret = g_nav_year[long_g_name] - 1
        # long_g = g_ret_year[long_g_name]
        # long_IR_year = long_g.mean() / long_g.std()
        # ################################################################################################################

        # return self.fact_test_result[self.fact_name]['Group']

    @timer
    def factor_to_csv(self):

        factor = self.dataSet['factDirty'][self.fact_name]
        file_path = os.path.join(FPN.FactorDataSet.value, factor.factor_category)

        if not os.path.exists(file_path):
            os.makedirs(file_path)

        data_path = os.path.join(file_path, factor.factor_name + '.csv')
        factor.data.to_csv(data_path, header=True)

    """画图"""
    # Plot  TODO 目前只画了分层结果
    @timer
    def plotRes(self, **kwargs):

        factExp = self.dataSet['Group']['dataMerge'][self.fact_name]

        exNav = self.dataSet['Group']['exNav']
        exRet = exNav.pct_change().dropna()

        lastDate = exNav.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        dataPlot = pd.DataFrame(
            {
                "Count": factExp.groupby('date').count(),
                "25%": factExp.groupby('date').apply(lambda x: x.quantile(0.25)),
                "50%": factExp.groupby('date').apply(lambda x: x.quantile(0.5)),
                "75%": factExp.groupby('date').apply(lambda x: x.quantile(0.75)),
            })

        fig = plt.figure(figsize=(20, 10))

        ax1 = fig.add_subplot(2, 3, 1)
        factExp.plot.hist(bins=30, ax=ax1)

        ax2 = fig.add_subplot(2, 3, 2)
        ax2.xaxis.label.set_visible(False)
        dataPlot['Count'].plot(rot=30, label=False, ax=ax2)

        ax3 = fig.add_subplot(2, 3, 3)
        dataPlot[['25%', '50%', '75%']].plot(rot=30, legend=False, ax=ax3)
        ax3.xaxis.label.set_visible(False)

        ax4 = fig.add_subplot(2, 3, 4)
        exNav.plot(rot=30, legend=False, ax=ax4)
        ax4.set_ylabel('EX_NAV')
        ax4.xaxis.label.set_visible(False)

        ax5 = fig.add_subplot(2, 3, 5)
        (exRet.loc[date3Y:] + 1).cumprod().plot(rot=30, legend=False, ax=ax5)
        ax5.xaxis.label.set_visible(False)

        ax6 = fig.add_subplot(2, 3, 6)
        (exRet.loc[date1Y:] + 1).cumprod().plot(rot=30, legend=False, ax=ax6)
        ax6.xaxis.label.set_visible(False)
        plt.legend(bbox_to_anchor=(1.2, 0), loc=4, ncol=1)

        plt.suptitle(f"{self.fact_name}-{self.hp}days")

        plt.tight_layout(h_pad=0, w_pad=0, pad=0.5)
        if kwargs['save']:
            plt.savefig(os.path.join(FPN.factor_test_res.value,
                                     f"{self.fact_name}_nav-{self.hp}days.png"),
                        dpi=500,
                        bbox_inches='tight')

        plt.show()

    # cal ind  TODO 先用近三年替代
    def groupIndexCal(self, nav: pd.DataFrame, freq: str = "D"):
        """
        最近
        """
        # ret days
        ret = nav.pct_change()
        # Date notes
        lastDate = nav.index[-1]
        date1Y = lastDate - dt.timedelta(days=365)
        date3Y = lastDate - dt.timedelta(days=365 * 3)

        mapping = {"All": ":",
                   "1Y": "date1Y:",
                   "3Y": "date3Y:",
                   "1T3Y": "date3Y:date1Y",
                   "3YBef": ":date3Y"}

        # 基础指标的计算
        indMethod = [self.ind.return_a, self.ind.std_a, self.ind.shape_a, self.ind.max_retreat]

        for v_ in ['All', '1Y', '3Y', '1T3Y', '3YBef']:
            locals()['nav' + i] = nav.loc[locals()[mapping[i]]]
            locals()['ret' + i] = ret.dropna().loc[locals()[mapping[i]]]
            locals()['ind' + v_] = locals()['nav' + v_].agg(indMethod)
            # 改为IC
            locals()['FD' + v_] = np.sign(int(locals()['nav' + v_].iloc[-1].idxmax().split('_')[-1]) -
                                          int(locals()['nav' + v_].iloc[-1].idxmin().split('_')[-1]))
            locals()['FTest' + v_], locals()['P' + v_] = stats.f_oneway(*locals()['ret' + v_].dropna().values.T)

        # merge
        ind3Y.loc['FD3Y'] = FDAll
        ind3Y.loc['FTest3Y'] = FTest3Y
        ind3Y.loc['P3Y'] = P3Y
        self.Res[self.fact_name]['Group'] = ind3Y.unstack()

    # Series additional written
    def to_csv(self, path: str, file_name: str, data_: pd.Series):
        data_path_ = os.path.join(path, file_name + '.csv')
        data_df = data_.to_frame().T

        header = False if os.path.exists(data_path_) else True

        data_df.to_csv(data_path_, mode='a', header=header)

    def _reg_fact_ret(self, data_: pd.DataFrame, num: int = 150) -> object or None:
        """
        返回回归类
        """
        data_sub = data_.sort_index().dropna()
        if data_sub.shape[0] < num:
            res = pd.Series(index=['T', 'fact_ret'])
        else:
            # 剔除极端值
            data_sub[KN.RETURN.value] = self.fact_clean.mad(data_sub[KN.RETURN.value])

            weight = data_sub[SN.STOCK_WEIGHT.value]
            d_ = data_sub.loc[:, data_sub.columns != SN.STOCK_WEIGHT.value]
            X = pd.get_dummies(d_.loc[:, d_.columns != KN.RETURN.value],
                               columns=[SN.INDUSTRY_FLAG.value])
            Y = d_[KN.RETURN.value]
            reg = sm.WLS(Y, X, weights=weight).fit(cov_type='HC1')

            if np.isnan(reg.rsquared_adj):
                res = pd.Series(index=['T', 'fact_ret'])
            else:
                res = pd.Series([reg.tvalues[self.fact_name], reg.params[self.fact_name]],
                                index=['T', 'fact_ret'])
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

    """多路径取平均"""

    # 考虑路径依赖，多路径取平均
    def group_ret(self,
                  data: pd.DataFrame,
                  weight_name: str = SN.CSI_300_INDUSTRY_WEIGHT.value) -> pd.DataFrame:
        """
        :param data:
        :param weight_name:
        :return:
        """
        group_ = data[SN.GROUP.value].unstack().sort_index()
        group_ = group_.reindex(
            self.td[(self.td[KN.TRADE_DATE.value] >= group_.index[0]) &
                    (self.td[KN.TRADE_DATE.value] <= group_.index[-1])][KN.TRADE_DATE.value])

        # The average in the group and weighting of out-of-group BenchMark, consider return period
        res_cont_, res = [], {}
        for i in range(0, self.hp):
            group_copy = group_.copy(deep=True)
            data_ = data.copy(deep=True)

            array1 = np.arange(0, group_copy.shape[0], 1)
            array2 = np.arange(i, group_copy.shape[0], self.hp)
            row_ = list(set(array1).difference(array2))

            # 非调仓期填为空值 TODO 调仓期涨跌停收益率标记为空
            group_copy.iloc[row_] = group_copy.iloc[row_].replace(range(int(max(data_[SN.GROUP.value].dropna())) + 1),
                                                                  np.nan)

            group_copy = group_copy.ffill(limit=self.hp - 1) if self.hp != 1 else group_copy

            # 替换原组别并进行收益率的计算
            data_[SN.GROUP.value] = group_copy.stack()
            data_ = data_.dropna(subset=[KN.RETURN.value, weight_name])  # TODO
            # TODO 改
            group_ret = data_.groupby([KN.TRADE_DATE.value, SN.GROUP.value]).apply(
                lambda x: np.average(x[KN.RETURN.value], weights=x[weight_name]))
            res[i] = group_ret
        # 取平均
        res_df = pd.DataFrame(res).mean(axis=1).unstack()
        res_df.columns = [f'G_{int(col_)}' for col_ in res_df.columns]  # rename
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
        var_weight_A = np.cov(data_array[0], aweights=weight)
        var_weight_B = np.cov(data_array[-1], aweights=weight)
        # calculate the weighted correlation
        corr_weight = cov_weight / pow((var_weight_A * var_weight_B), 0.5)

        return corr_weight[0][1]


# 多因子相关性分析
class FactorCollinearity(object):
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

    parent_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

    def __init__(self):
        self.db = database_manager
        self.Q = SQL()

        self.fp = FactorProcess()  # 因子预处理
        self.Multi = Multicollinearity()  # 多因子处理

        self.factors_raw = None  # 原始因子集

        self.factor_D = {}  # 因子符号集
        self.factor_direction()

        self.td = self.Q.trade_date_csv()

    # factor direction mapping  TODO 后续改成时间序列，方便不同时期的因子合成
    def factor_direction(self, file_name: str = 'factor_direction.json'):
        try:
            file_path = os.path.join(self.parent_path, file_name)
            infile = open(file_path, 'r', encoding='utf-8')
            self.factor_D = json.load(infile)
        except Exception as e:
            print(f"read json file failed, error\n{traceback.format_exc()}")
            self.factor_D = {}

    # 获取因子数据
    def get_data(self,
                 folder_name: str = '',
                 factor_names: dict = None,
                 factors_df: pd.DataFrame = None):

        """
        数据来源：
        1.外界输入；
        2.路径下读取csv
        :param factor_names:
        :param folder_name:
        :param factors_df:
        :return:
        """
        if factors_df is None:
            try:
                factors_path = os.path.join(FPN.FactorRawData.value, folder_name)
                if factor_names:
                    factor_name_list = list(map(lambda x: x + '.csv', factor_names))
                else:
                    factor_name_list = os.listdir(factors_path)
            except FileNotFoundError:
                print(f"Path error, no folder name {folder_name} in {FPN.factor_ef.value}!")
            else:
                factor_container = []
                # 目前只考虑csv文件格式
                for factor_name in factor_name_list:
                    if factor_name[-3:] != 'csv':
                        continue
                    data_path = os.path.join(factors_path, factor_name)
                    print(f"Read factor data:{factor_name[:-4]}")
                    factor_data = pd.read_csv(data_path, index_col=[KN.TRADE_DATE.value, KN.STOCK_ID.value])
                    factor_container.append(factor_data[factor_name[:-4]])

                if not factor_container:
                    print(f"No factor data in folder {folder_name}!")
                else:
                    self.factors_raw = pd.concat(factor_container, axis=1)
        else:
            self.factors_raw = factors_df.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

    # 相关性检验

    def correctionTest(self):
        COR = self.Multi.correlation(self.factors_raw)

        print('S')
        pass

    #  因子合成
    def factor_synthetic(self,
                         method: str = 'Equal',
                         factor_D: dict = None,
                         stand_method: str = 'z_score',
                         ret_type: str = 'Pearson',
                         **kwargs):
        """
        因子复合需要考虑不同因子在不同的截面数据缺失情况，对于当期存在缺失因子时，复合因子取有效因子进行加权，而不是剔除
        :param method:
        :param factor_D:
        :param stand_method:
        :param ret_type:
        :param kwargs:
        :return:
        """
        # 更新因子符号
        if factor_D is not None:
            self.factor_D.update(factor_D)

        # 全量处理，滚动处理后续再补
        if method != 'Equal':
            if kwargs.get('fact_ret', None) is None:
                factor_name_tuple = tuple(self.factors_raw.columns)
                fact_ret = self.factor_ret_from_sql(factor_name_tuple, hp=kwargs['hp'], ret_type=ret_type)
            else:
                factor_name_tuple = tuple(kwargs['fact_ret'].columns)
                fact_ret = kwargs['fact_ret']

            if len(fact_ret['factor_name'].drop_duplicates()) < len(factor_name_tuple):
                print(f"因子{ret_type}收益数据缺失，无法进行计算")
                return

            kwargs['fact_ret'] = fact_ret.pivot_table(values='factor_return',
                                                      index=KN.TRADE_DATE.value,
                                                      columns='factor_name')
            # 交易日修正
            # IC = IC.reindex(self.td[(self.td['date'] >= IC.index[0]) & (self.td['date'] <= IC.index[-1])]['date'])
            # td = self.Q.query(self.Q.trade_date_SQL(date_sta=kwargs['fact_ret'].index[0].replace('-', ''),
            #                                         date_end=kwargs['fact_ret'].index[-1].replace('-', '')))

            kwargs['fact_ret'] = kwargs['fact_ret'].reindex(self.td[
                                                                (self.td['date'] >= kwargs['fact_ret'].index[0]) &
                                                                (self.td['date'] <= kwargs['fact_ret'].index[-1])][
                                                                'date'])

        factor_copy = self.factors_raw.copy(deep=True)
        # 因子符号修改
        for fact_ in factor_copy.columns:
            if self.factor_D[fact_] == '-':
                factor_copy[fact_] = - factor_copy[fact_]
            elif self.factor_D[fact_] == '+':
                pass
            else:
                print(f"{fact_}因子符号有误！")
                return

        # 对因子进行标准化处理

        factor_copy = factor_copy.apply(self.fp.standardization, args=(stand_method,))

        comp_factor = self.Multi.composite(factor=factor_copy,
                                           method=method,
                                           **kwargs)

        return comp_factor

    def factor_ret_from_sql(self,
                            factor_name: tuple,
                            sta_date: str = '2013-01-01',
                            end_date: str = '2020-04-01',
                            ret_type: str = 'Pearson',
                            hp: int = 1):

        fact_ret_sql = self.db.query_factor_ret_data(factor_name=factor_name,
                                                     sta_date=sta_date,
                                                     end_date=end_date,
                                                     ret_type=ret_type,
                                                     hp=hp)
        return fact_ret_sql


if __name__ == '__main__':
    W = FactorCollinearity()
