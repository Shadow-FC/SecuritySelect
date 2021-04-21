import pandas as pd
from typing import List


# 格式转换
def tradeCashFlow_switch(data: List[pd.Series]) -> pd.DataFrame:
    res = pd.DataFrame(data)
    return res
