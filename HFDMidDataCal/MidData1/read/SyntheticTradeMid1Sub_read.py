import os
import pandas as pd
import pickle5 as pickle


def SyntheticTradeMid1Sub_read(filePath: str,
                               *args,
                               **kwargs) -> pd.DataFrame:
    with open(filePath, "rb") as f:
        dataInput = pickle.load(f)
    return dataInput
