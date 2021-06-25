# -*-coding:utf-8-*-
# @Time:   2021/4/6 14:53
# @Author: FC
# @Email:  18817289038@163.com

import os
import sys
import numpy as np
import pandas as pd
import pickle5 as pickle

from Object import (
    DataInfo
)

from utility.utility import (
    timer
)
from constant import (
    KeyName as KN,
    PriceVolumeName as PVN,
    FilePathName as FPN,
    SpecialName as SN
)


class BenchmarkMethod(object):

    def benchmark_ZD(self,
                     data: pd.DataFrame,
                     retName: str,
                     wName: str) -> pd.Series(float):
        """
        基准的合成为原始样本
        Args:
            data ():
            retName (): 收益率名称
            wName (): 权重名称
        Returns:

        """
        # Nan values will carry some weight so that the benchmark return is biased
        bm_raw_data = data[[retName, wName]].dropna()
        bm_ret = bm_raw_data.groupby(KN.TRADE_DATE.value).apply(lambda x: np.average(x[retName], weights=x[wName]))
        return bm_ret


class Benchmark(object):
    def __init__(self):
        self.meth = BenchmarkMethod()
        self.local_path = FPN.Local_inputData.value

    def benchmarkZD(self,
                    data: pd.DataFrame,
                    retName: str,
                    wName: str) -> DataInfo:
        func_name = sys._getframe().f_code.co_name
        result_path = os.path.join(self.local_path, func_name + '.pkl')

        if os.path.exists(result_path):
            with open(result_path, 'rb') as f:
                bm_ret = pickle.load(f)
        else:
            # Nan values will carry some weight so that the benchmark return is biased
            bm_ret = self.meth.benchmark_ZD(data, retName, wName)
            bm_ret.name = func_name
            bm_ret.to_pickle(result_path)

        dataClass = DataInfo(data=bm_ret,
                             data_category=self.__class__.__name__,
                             data_name=func_name)
        return dataClass
