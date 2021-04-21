import os
import pandas as pd
import pickle5 as pickle


def SyntheticTradeMid1_read(path: str,
                            fileName: str,
                            *args,
                            **kwargs) -> pd.DataFrame:
    with open(os.path.join(path, fileName), "rb") as fh:
        dataInput = pickle.load(fh)
    # dataInput = pd.read_csv(os.path.join(path, fileName), encoding='GBK')
    return dataInput
