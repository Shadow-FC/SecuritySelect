# -*-coding:utf-8-*-
# @Time:   2021/2/7 13:04
# @Author: FC
# @Email:  18817289038@163.com

import os
import pandas as pd
from functools import reduce
from typing import Dict, List

from DataAPI.StockAPI.StockPool import StockPool
from DataAPI.LabelAPI.Labelpool import LabelPool
from Portfolio.MultiFactorPortfolio import PortfolioModel

from constant import (
    KeyName as KN,
    SpecialName as SN,
    FilePathName as FPN,
    PriceVolumeName as PVN
)


def get_factors(factorDict: Dict[str, List[str]]) -> pd.DataFrame:
    """
    因子数据外连接，内连接会损失很多数据
    Args:
        factorDict ():

    Returns:

    """
    factor_data = []
    for fold, factor_list in factorDict.items():
        pathSub = os.path.join(FPN.Fact_dataSet.value, fold)
        for factor_name in factor_list:
            factor_path = os.path.join(pathSub, factor_name + '.csv')
            factor = pd.read_csv(factor_path)
            factor_data.append(factor)
    if len(factor_data) > 1:
        factors = reduce(lambda x, y: pd.merge(x, y,
                                               on=[KN.TRADE_DATE.value, KN.STOCK_ID.value],
                                               how='outer'), factor_data)
    else:
        factors = factor_data[0]
    factors = factors.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])
    return factors


if __name__ == '__main__':
    effectFactors = {"HighFrequencyDistributionFactor": ['Distribution001_1min_1days', 'Distribution004_1min_1days']}

    LP = LabelPool()
    SP = StockPool()

    stockPool = SP.StockPoolZD()
    label = LP.strategyLabel()

    fac_exp = get_factors(effectFactors)

    fac_exp_N = fac_exp.reindex(stockPool.index)
    label_N = label.reindex(stockPool.index).sort_index()

    parameters = {"fac_exp": fac_exp_N.dropna(),  # 因子暴露
                  "stock_ret": label[KN.RETURN.value],  # 个股收益
                  "ind_exp": label_N[SN.INDUSTRY_FLAG.value],  # 行业暴露
                  "mv": label_N[PVN.LIQ_MV.value],  # 个股市值

                  "ind_mv": label_N[SN.INDUSTRY_MV.value],  # 指数行业市值
                  "ind_weight": label_N[SN.INDUSTRY_WEIGHT.value],
                  "hp": 20
                  }

    B = PortfolioModel(**parameters)
    B.main()
