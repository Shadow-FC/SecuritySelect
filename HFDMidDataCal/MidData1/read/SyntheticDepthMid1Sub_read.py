import os
import pandas as pd
import pickle5 as pickle


def SyntheticDepthMid1Sub_read(filePath: str,
                               *args,
                               **kwargs) -> pd.DataFrame:
    with open(filePath, "rb") as fh:
        dataInput = pickle.load(fh)
    return dataInput
