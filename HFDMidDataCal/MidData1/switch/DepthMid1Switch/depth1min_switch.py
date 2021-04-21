import pandas as pd
from typing import List, Union


# 格式转换
def depth1min_switch(data: List[pd.DataFrame]) -> pd.DataFrame:
    res = pd.concat(data)
    return res
