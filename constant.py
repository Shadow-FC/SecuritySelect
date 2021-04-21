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
    # factor_info = 'Z:\\Database\\'  # 因子信息路径

    Input_data_server = 'Y:\\DataBase'  # 服务端数据
    Input_data_local = 'A:\\DataBase\\SecuritySelectData\\InputData'  # 本地数据

    factor_pool_path = 'A:\\DataBase\\SecuritySelectData\\FactorPool\\'  # 因子池
    factor_inputData = 'A:\\DataBase\\SecuritySelectData\\FactorPool\\Factor_InputData\\'  # 因子计算所需数据
    FactorRawData = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorDataSet\\RawDataFundamental\\"  # 未经过处理的因子集
    FactorDataSet = "D:\\DataBase\\"  # 标准因子集(日频)
    # FactorDataSet = "D:\\DataBase\\NEW2"  # 标准因子集(日频)
    factor_test_res = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorsTestResult\\"  # 因子检验结果保存

    factor_ef = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorEffective\\"  # 筛选有效因子集
    factor_comp = "A:\\DataBase\\SecuritySelectData\\FactorPool\\FactorEffective\\FactorComp\\"  # 复合因子数据集

    Trade_Date = 'Y:\\DataBase'  # 交易日
    List_Date = 'A:\\DataBase\\ListDate'  # 成立日

    HFD_Stock_Depth_1min = 'Y:\\合成数据\\十档1min\\因子数据'  # 高频十档分钟数据
    HFD_Stock_M = 'Y:\\合成数据\\逐笔1min'  # 高频分钟数据
    HFD_Stock_Depth = 'Y:\\合成数据\\十档Vwap'  # 高频十档盘口数据
    HFD_Stock_CF = 'Y:\\合成数据\\逐笔资金流'  # 逐笔资金流向
    HFD_MidData = 'Y:\\合成数据\\中间过程2'  # 高频因子中间数据
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


class DBPath(Enum):
    pathMidData1 = r"D:\DataBase\MidData1"
    pathMidData2 = r"D:\DataBase\MidData2"

    TXT = os.path.join(projectPath, "FileSets")
    Record = os.path.join(projectPath, "FileSets")

    depth1min = os.path.join(pathMidData1, "depth1min")
    depthVwap = os.path.join(pathMidData1, "depthVwap")

    trade1min = os.path.join(pathMidData1, "trade1min")
    tradeCashFlow = os.path.join(pathMidData1, "tradeCashFlow")
    tradeBigOrderNum = os.path.join(pathMidData1, "tradeBigOrderNum")
    tradeBigOrderTime = os.path.join(pathMidData1, "tradeBigOrderTime")
    tradeInvF_004_ZBQX = os.path.join(pathMidData1, "tradeInvF_004_ZBQX")
    tradeInvF_005_ZBQX = os.path.join(pathMidData1, "tradeInvF_005_ZBQX")

    depth5VolSum = os.path.join(pathMidData2, "depth5VolSum")
    depth10VolSum = os.path.join(pathMidData2, "depth10VolSum")

    depthEqualIndex = os.path.join(pathMidData2, "depthEqualIndex")
    depthWeightedIndex = os.path.join(pathMidData2, "depthWeightedIndex")

    tradeRet = os.path.join(pathMidData2, "tradeRet")
    tradeVol = os.path.join(pathMidData2, "tradeVol")
    tradeClose = os.path.join(pathMidData2, "tradeClose")
    tradeAmtStd = os.path.join(pathMidData2, "tradeAmtStd")
    tradeAmtSum = os.path.join(pathMidData2, "tradeAmtSum")
    tradeTradeNum = os.path.join(pathMidData2, "tradeTradeNum")
    tradeBuyAmtSum = os.path.join(pathMidData2, "tradeBuyAmtSum")
    tradeSellAmtSum = os.path.join(pathMidData2, "tradeSellAmtSum")
    tradeSpecial1 = os.path.join(pathMidData2, "tradeSpecial1")
    tradeSpecial2 = os.path.join(pathMidData2, "tradeSpecial2")

    tradeEqualIndex = os.path.join(pathMidData2, "tradeEqualIndex")
    tradeWeightedIndex = os.path.join(pathMidData2, "tradeWeightedIndex")


@unique
class DBName(Enum):
    CSV = 'csv'
    TXT = 'txt'
    JSON = 'json'
    PKL = 'pkl'


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
class StrategyName(Enum):
    pass


# def timer(func):
#     def wrapper(*args, **kwargs):
#         func_name = func.__name__
#
#         sta = time.time()
#
#         res = func(*args, **kwargs)
#
#         rang_time = round((time.time() - sta) / 60, 4)
#
#         print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: It takes\033[0m "
#               f"\033[1;33m{rang_time:<6}Min\033[0m "
#               f"\033[1;31mto run func\033[0m "
#               f"\033[1;33m\'{func_name}\'\033[0m")
#         return res
#
#     return wrapper
#
#
# # 因子计算封装类
# class Process(object):
#     def __init__(self,
#                  funcType: str = ""):
#         self.funcType = funcType
#
#     def __call__(self, func):
#         def inner(*args, **kwargs):
#             func_name = func.__name__
#             data = kwargs['data'].set_index([KeyName.TRADE_DATE.value, KeyName.STOCK_ID.value])
#             kwargs['data'] = data.sort_index()
#
#             res = func(*args, **kwargs, name=func_name)
#             F = DataInfo(data=res['data'],
#                          data_name=res['name'],
#                          data_type=self.funcType,
#                          data_category=func.__str__().split(" ")[1].split('.')[0])
#             return F
#
#         return inner


if __name__ == '__main__':
    print('s')
