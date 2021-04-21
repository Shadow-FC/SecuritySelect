# -*-coding:utf-8-*-
# @Time:   2021/4/15 14:02
# @Author: FC
# @Email:  18817289038@163.com

import pandas as pd
from typing import List


def depthEqualIndex_switch(data: List[pd.DataFrame]) -> pd.DataFrame:
    dataNew = pd.concat(data)
    return dataNew
