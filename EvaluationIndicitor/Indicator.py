# -*-coding:utf-8-*-
# @Time:   2020/9/4 8:42
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
import scipy as stats
import numpy as np
import datetime as dt
from sklearn import linear_model
import warnings

warnings.filterwarnings('ignore')


class Indicator(object):

    """
    时序上收益类指标的计算用到的是对数收益率，该收益率在截面上不具备可加性
    """
    cycle = {"D": 252,
             "W": 52,
             "M": 12,
             "Y": 1}

    # 累计收益率
    def acc_return(self, nav: pd.Series) -> float:
        if nav.empty:
            ret = np.nan
        else:
            ret = nav[-1] / nav[0] - 1
        return ret

    # 年化累计收益率
    def return_a(self, nav: pd.Series, freq: str = 'D'):

        sta, end = nav.index[0], nav.index[-1]

        period = (end - sta).days

        if period == 0:
            return 0
        else:
            ret_a = (self.acc_return(nav) + 1) ** (self.cycle[freq] / period) - 1
            return ret_a

    def odds(self, nav: pd.Series, bm: pd.Series) -> float:

        return sum(nav > bm) / len(nav)

    def std_a(self, nav: pd.Series, freq: str = 'D') -> float:
        std_a = nav.std() * (self.cycle[freq] ** .5)
        return std_a

    def max_retreat(self, nav: pd.Series):
        # 回撤结束时间点
        i = (nav.cummax() - nav).idxmax()
        # 回撤开始的时间点
        j = (nav[:i]).idxmax()
        x = (float(nav[i]) / nav[j]) - 1
        return x

    def shape_a(self, nav: pd.Series, freq: str = "D") -> float:
        shape_a = self.return_a(nav, freq=freq) / self.std_a(nav, freq="D")
        return shape_a
