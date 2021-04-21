import pandas as pd
from typing import List


def tradeAmtStd_switch(data: List[pd.DataFrame]) -> pd.DataFrame:
    dataNew = pd.concat(data)
    return dataNew
