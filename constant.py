# -*-coding:utf-8-*-
# @Time:   2020/9/1 16:56
# @Author: FC
# @Email:  18817289038@163.com

import os
import sys
import time
import pandas as pd
import datetime as dt
from Object import DataInfo
from functools import wraps
from enum import Enum, unique
from typing import Callable, Union, Any

projectPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")


class FilePathName(Enum):

    Server_inputData = 'Y:\\DataBase'  # 服务端数据
    Local_inputData = 'A:\\DataBase\\SecuritySelectData\\InputData'  # 本地数据

    Fact_dataSet = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorDataSet"  # 因子库
    Fact_testRes = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorsTestResult\\"  # 因子检验结果保存
    Fact_corrRes = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorCorr\\"  # 因子相关性结果保存

    List_Date = 'A:\\DataBase\\ListDate'  # 成立日

    HFD_depth1min = r'Y:\合成数据\十档1min\因子数据'  # 高频十档分钟数据  Y:\\合成数据\\十档1min\\因子数据
    HFD_trade1min = r'B:\样本外数据_2021年\MidData1\逐笔\trade1min'  # 高频分钟数据  Y:\\合成数据\\逐笔1min
    HFD_depthVwap = r'B:\合并数据\MidData1\十档Vwap'  # 高频十档盘口数据  Y:\\合成数据\\十档Vwap
    HFD_tradeCF = r'B:\合并数据\MidData1\逐笔资金流'  # 逐笔资金流向  Y:\\合成数据\\逐笔资金流
    HFD_midData = r'B:\合并数据\MidData2'  # 高频因子中间过程数据  B:\\中间过程2
    HFD = 'A:\\DataBase\\HFD'  # 高频数据存储地址


@unique
class KeyName(Enum):
    STOCK_ID = 'code'
    TRADE_DATE = 'date'
    TRADE_TIME = 'time'
    LIST_DATE = 'listDate'
    RETURN = 'ret'


@unique
class SpecialName(Enum):
    GROUP = 'group'

    STOCK_WEIGHT = 'stockWeight'
    CSI_300 = '000300.SH'
    CSI_500 = '000905.SH'
    CSI_800 = '000906.SH'
    WI_A = 'Wind_A'

    INDUSTRY_MV = 'ind_mv'
    INDUSTRY_WEIGHT = 'ind_w'
    INDUSTRY_FLAG = 'indexCode'
    CSI_300_INDUSTRY_WEIGHT = 'csi_300_weight'
    CSI_500_INDUSTRY_WEIGHT = 'csi_500_weight'
    CSI_50_INDUSTRY_WEIGHT = 'csi_50_weight'

    CSI_300_INDUSTRY_MV = 'csi_300_mv'
    CSI_500_INDUSTRY_MV = 'csi_500_mv'
    CSI_50_INDUSTRY_MV = 'csi_50_mv'
    ANN_DATE = 'date'
    REPORT_DATE = 'report_date'


@unique
class PriceVolumeName(Enum):
    CLOSE = 'close'
    OPEN = 'open'
    HIGH = 'high'
    LOW = 'low'

    CLOSE_ADJ = 'closeAdj'
    OPEN_ADJ = 'openAdj'
    HIGH_ADJ = 'highAdj'
    LOW_ADJ = 'lowAdj'

    Up_Down = 'priceLimit'
    ISST = 'isst'
    LIST_DAYS_NUM = 'period2list'
    LIST_BOARD = 'listBoard'

    AMOUNT = 'amount'
    VOLUME = 'volume'

    ADJ_FACTOR = 'adjfactor'

    LIQ_MV = 'liqMv'
    TOTAL_MV = 'totalMv'


@unique
class BroadName(Enum):
    STIB = '科创板'


@unique
class ExchangeName(Enum):
    SSE = 'SSE'
    SZSE = 'SZSE'


@unique
class FinancialBalanceSheetName(Enum):
    Total_Asset = 'total_asset'  # 总资产
    Liq_Asset = 'liq_asset'  # 流动性资产
    ILiq_Asset = 'iliq_asset'  # 非流动性资产
    Fixed_Asset = 'fixed_asset'  # 固定资产

    Currency = 'money'  # 货币资金
    Tradable_Asset = 'tradable_asset'  # 可交易金融资产

    ST_Borrow = 'st_borrow'  # 短期借款
    ST_Bond_Payable = 'st_Bond_P'  # 短期应付债券
    ST_IL_LB_1Y = 'st_lb'  # 一年内到期的非流动负债
    LT_Borrow = 'lt_borrow'  # 长期借款

    Tax_Payable = 'tax_patable'  # 应交税费

    Total_Lia = 'total_liability'  # 总负债

    Actual_Capital = 'actual_capital'  # 总股本
    Surplus_Reserves = 'surplus_reserves'  # 盈余公积
    Undistributed_Profit = 'undistributed_profit'  # 未分配利润

    Net_Asset_Ex = 'shareholder_equity_ex'  # （不含少数股东权益）净资产
    Net_Asset_In = 'shareholder_equity_in'  # （含少数股东权益）净资产


@unique
class FinancialIncomeSheetName(Enum):
    Net_Pro_In = 'net_profit_in'  # 净利润（包含少数股东权益）
    Net_Pro_Ex = 'net_profit_ex'  # 净利润（不包含少数股东权益）
    Net_Pro_Cut = 'net_profit_cut'  # 净利润（扣除非经常性损益）

    Total_Op_Income = 'total_op_ic'  # 营业总收入
    Op_Total_Cost = 'op_total_cost'  # 营业总成本

    Op_Income = 'op_ic'  # 营业收入
    Op_Pro = 'op_pro'  # 营业利润
    Op_Cost = 'op_cost'  # 营业成本

    Tax = 'tax'  # 所得税
    Tax_Surcharges = 'tax_surcharges'  # 税金及附加


@unique
class FinancialCashFlowSheetName(Enum):
    Net_CF = 'net_cash_flow'  # 净现金流
    Op_Net_CF = 'op_net_cash_flow'  # 经营性活动产生的现金流量净额
    All_Tax = 'tax_all'  # 支付的各项税费

    Cash_From_Sales = 'cash_sales'  # 销售商品、提供劳务收到的现金

    Free_Cash_Flow = 'FCFF'  # 自由现金流


@unique
class FactorCategoryName(Enum):
    Val = 'ValuationFactor'
    Gro = 'GrowthFactors'
    Pro = 'ProfitFactor'
    Sol = 'SolvencyFactor'
    Ope = 'OperateFactor'
    EQ = 'QualityFactor'
    Size = 'SizeFactor'
    MTM = 'MomentumFactor'
    HFD = 'HighFrequencyFactor'


@unique
class FactorType(Enum):
    HF_Dis = 'HighFrequencyDistribution'
    HF_FundFlow = 'HighFrequencyFundFlow'
    HF_VP = 'HighFrequencyVolPrice'
    Tech_Beh = 'TechnicalBehavior'
    Tech_Mom = 'TechnicalMoment'
    Fund_EQ = 'FundamentalEaringQuality'
    Fund_Grow = 'FundamentalGrow'
