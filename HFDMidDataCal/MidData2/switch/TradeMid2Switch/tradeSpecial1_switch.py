import pandas as pd
from typing import List


def tradeSpecial1_switch(data: List[pd.DataFrame]) -> pd.DataFrame:
    dataNew = pd.concat(data)
    return dataNew
