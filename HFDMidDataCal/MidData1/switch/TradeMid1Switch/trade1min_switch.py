import pandas as pd
from typing import List


# 格式转换
def trade1min_switch(data: List[str]) -> pd.DataFrame:
    res = pd.concat(data)
    return res
