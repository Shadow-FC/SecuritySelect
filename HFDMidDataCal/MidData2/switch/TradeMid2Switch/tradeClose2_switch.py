# -*-coding:utf-8-*-
# @Time:   2021/4/30 10:04
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
from typing import List


def tradeClose2_switch(data: List[pd.DataFrame]) -> pd.DataFrame:
    dataNew = pd.concat(data)
    return dataNew
