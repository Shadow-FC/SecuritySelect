# -*-coding:utf-8-*-
# @Time:   2021/4/1 16:22
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
from typing import Dict, Any
from Object import (
    DataInfo
)
from utility.utility import (
    timer
)

from DataAPI.StockAPI.StockPool import StockPool
from DataAPI.LabelAPI.Labelpool import LabelPool
from DataAPI.FactorAPI.FactorPool import FactorPool
from DataAPI.BenchmarkAPI.Benchmark import Benchmark


# 数据传输转移到这
class LoadData(object):

    def __init__(self):
        self.Factor = FactorPool()  # 因子池
        self.Label = LabelPool()  # 标签池
        self.Stock = StockPool()  # 股票池
        self.BM = Benchmark()  # 业绩基准

    @timer
    def getFactorData(self,
                      fact_name: str,
                      fact_params: Dict[str, Any],
                      fact_value: pd.DataFrame = None
                      ) -> DataInfo:
        """
        Args:
            fact_name (): 因子名称
            fact_params (): 因子参数
            fact_value (): 因子值
        Returns:
        """
        res = self.Factor.load_factor(fact_name, fact_params, fact_value)
        return res

    @timer
    def getStockPoolData(self, stockPoolName: str) -> DataInfo:
        res = self.Stock.__getattribute__(stockPoolName)()
        return res

    @timer
    def getLabelPoolData(self, labelPoolName: str) -> DataInfo:
        res = self.Label.__getattribute__(labelPoolName)()
        return res

    @timer
    def getBMData(self, benchMarkName: str, **kwargs) -> DataInfo:
        res = self.BM.__getattribute__(benchMarkName)(**kwargs)
        return res


if __name__ == '__main__':
    A = LoadData()
    A.getFactorData(1, 2, "")
