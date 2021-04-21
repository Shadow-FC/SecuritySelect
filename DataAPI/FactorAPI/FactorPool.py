# -*-coding:utf-8-*-
# @Time:   2021/4/1 15:18
# @Author: FC
# @Email:  18817289038@163.com

import os
import inspect
import importlib
import pandas as pd
import datetime as dt
from typing import Any, Dict

from Object import (
    DataInfo
)
from constant import (
    KeyName as KN
)


class FactorPool(object):
    def __init__(self):
        self.factor, self.method = self.load_factor_function()

    def load_factor_function(self):
        """
        Load strategy class from source code.
        """
        factor_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'FactorCalculate')
        Factor_class = self.load_factor_class_from_folder(factor_folder)
        Factor_function = self.load_factor_function_from_class(Factor_class)
        return Factor_function

    # get factor value
    def load_factor(self,
                    fact_name: str,
                    fact_params: Dict[str, Any],
                    fact_value: pd.DataFrame = None,
                    **kwargs
                    ) -> DataInfo:
        """
        优先直接获取数据--否则实时计算
        """
        if fact_value is None:
            print(f"{dt.datetime.now().strftime('%X')}: Starting calculate the factors!")
            try:
                factRawData = self.factor[fact_name + '_data_raw'](**fact_params)
            except Exception as e:
                print(f"{dt.datetime.now().strftime('%X')}: Unable to load raw data that to calculate factor!-{e}")
                factClass = DataInfo()
            else:
                factClass = self.factor[fact_name](data=factRawData, **fact_params)
        else:
            print(f"{dt.datetime.now().strftime('%X')}: Get factor data from input!")
            fact_value = fact_value.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

            factClass = DataInfo()
            factClass.data = fact_value[fact_name]
            factClass.data_name = fact_name

        return factClass

    # 导入因子类
    @staticmethod
    def load_factor_class_from_folder(path: str):
        """
        Load strategy class from certain folder.
        """
        Factor_class = {}
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                # 剔除自己本身
                if filename.startswith('__'):
                    continue
                class_name = filename[:-3]
                module = importlib.import_module("DataAPI.FactorAPI.FactorCalculate." + class_name)
                for class_name in dir(module):
                    value = getattr(module, class_name)
                    if isinstance(value, type):
                        Factor_class[value.__name__] = value
            return Factor_class

    # 导入因子属性
    @staticmethod
    def load_factor_function_from_class(Factor_class: dict):
        """
        Load strategy class from module file.
        """
        Factor_function, Method_function = {}, {}
        for factor_class in Factor_class.values():
            for func_name in dir(factor_class):
                if func_name.startswith('__'):
                    continue
                method_ = getattr(factor_class, func_name)
                if inspect.ismethod(method_):
                    Factor_function[func_name] = method_
                elif inspect.isfunction(method_):
                    Method_function[func_name] = method_
        return Factor_function, Method_function
