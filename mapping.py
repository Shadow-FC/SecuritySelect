# -*-coding:utf-8-*-
# @Time:   2021/4/14 17:10
# @Author: FC
# @Email:  18817289038@163.com

import os
from constant import (
    DBPath as DBP,
    DBName as DBN
)

# CPU = os.environ['NUMBER_OF_PROCESSORS']
CPU = 2

projectPath = os.path.abspath(os.path.dirname(os.getcwd()) + os.path.sep + ".")

# 函数输出相关信息
saveMapping = {
    "TxT": {"Path": DBP.TXT.value, "DBName": DBN.TXT.value},
    "Record": {"Path": DBP.Record.value, "DBName": DBN.JSON.value},

    # 中间过程1
    "depth1min": {"Path": DBP.depth1min.value, "DBName": DBN.PKL.value},
    "depthVwap": {"Path": DBP.depthVwap.value, "DBName": DBN.PKL.value},

    "trade1min": {"Path": DBP.trade1min.value, "DBName": DBN.PKL.value},
    "tradeCashFlow": {"Path": DBP.tradeCashFlow.value, "DBName": DBN.PKL.value},
    "tradeBigOrderNum": {"Path": DBP.tradeBigOrderNum.value, "DBName": DBN.PKL.value},
    "tradeBigOrderTime": {"Path": DBP.tradeBigOrderTime.value, "DBName": DBN.PKL.value},
    "tradeInvF_004_ZBQX": {"Path": DBP.tradeInvF_004_ZBQX.value, "DBName": DBN.PKL.value},
    "tradeInvF_005_ZBQX": {"Path": DBP.tradeInvF_005_ZBQX.value, "DBName": DBN.PKL.value},

    # 中间过程2
    "depth5VolSum": {"Path": DBP.depth5VolSum.value, "DBName": DBN.PKL.value},
    "depth10VolSum": {"Path": DBP.depth10VolSum.value, "DBName": DBN.PKL.value},

    "depthEqualIndex": {"Path": DBP.depthEqualIndex.value, "DBName": DBN.PKL.value},
    "depthWeightedIndex": {"Path": DBP.depthWeightedIndex.value, "DBName": DBN.PKL.value},

    "tradeRet": {"Path": DBP.tradeRet.value, "DBName": DBN.PKL.value},
    "tradeVol": {"Path": DBP.tradeVol.value, "DBName": DBN.PKL.value},
    "tradeTradeNum": {"Path": DBP.tradeTradeNum.value, "DBName": DBN.PKL.value},
    "tradeAmtSum": {"Path": DBP.tradeAmtSum.value, "DBName": DBN.PKL.value},
    "tradeBuyAmtSum": {"Path": DBP.tradeBuyAmtSum.value, "DBName": DBN.PKL.value},
    "tradeSellAmtSum": {"Path": DBP.tradeSellAmtSum.value, "DBName": DBN.PKL.value},
    "tradeAmtStd": {"Path": DBP.tradeAmtStd.value, "DBName": DBN.PKL.value},
    "tradeClose": {"Path": DBP.tradeClose.value, "DBName": DBN.PKL.value},
    "tradeSpecial1": {"Path": DBP.tradeSpecial1.value, "DBName": DBN.PKL.value},
    "tradeSpecial2": {"Path": DBP.tradeSpecial2.value, "DBName": DBN.PKL.value},

    "tradeEqualIndex": {"Path": DBP.tradeEqualIndex.value, "DBName": DBN.PKL.value},
    "tradeWeightedIndex": {"Path": DBP.tradeWeightedIndex.value, "DBName": DBN.PKL.value},

}

# 函数读取相关信息
readMapping = {
    "StockInfo": r"Y:\DataBase\AStockData.csv",

    "SyntheticDepthMid1": r'Y:\十档',
    "SyntheticTradeMid1": r"Y:\逐笔全息",

    "SyntheticDepthMid2": r'Y:\合成数据\十档1min\因子数据',
    "SyntheticTradeMid2": r'Y:\合成数据\逐笔1min',

    "SyntheticDepthIndex2": r'Y:\合成数据\十档1min\因子数据',
    "SyntheticTradeIndex2": r'Y:\合成数据\逐笔1min',

    "weight500": r'Z:\冯晨\高频指数合成\A_500Weight.pkl',
}


close_price = {
    '0m': '09:30:00',
    '30m': '10:00:00',
    '60m': '10:30:00',
    '90m': '11:00:00',
    '120m': '11:30:00',
    '150m': '13:30:00',
    '180m': '14:00:00',
    '210m': '14:30:00',
    '240m': '15:00:00'
}

time_AM = {"call": "09:30:00",
           "5min": "09:35:00",
           "10min": "09:40:00",
           "15min": "09:45:00",
           "30min": "10:00:00",
           "60min": "10:30:00",
           "all": "11:30:00"}

time_PM = {"5min": "14:55:00",
           "10min": "14:50:00",
           "15min": "14:45:00",
           "30min": "14:30:00",
           "60min": "14:00:00",
           "all": "13:00:00"}

time_std = {"all": ["09:30:00", "15:00:00"],
            "open": ["09:30:00", "10:00:00"],
            "between": ["10:00:00", "14:30:00"],
            "close": ["14:30:00", "15:00:00"]}

range_T = lambda x: (x['time'] >= '09:30:00') & (x['time'] < '15:00:00')
