import os
import pandas as pd


def SyntheticDepthIndex2_read(path: str,
                              fileName: str,
                              **kwargs) -> pd.DataFrame:
    dataInput = pd.read_csv(os.path.join(path, fileName))
    return dataInput
