import os
import pandas as pd


def SyntheticTradeMid2_read(path: str,
                            fileName: str,
                            *args,
                            **kwargs) -> pd.DataFrame:
    dataInput = pd.read_csv(os.path.join(path, fileName))
    return dataInput
