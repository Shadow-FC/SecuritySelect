# -*-coding:utf-8-*-
# @Time:   2020/11/11 15:03
# @Author: FC
# @Email:  18817289038@163.com

import os
import json
import pandas as pd

from constant import (
    FilePathName as FPN
)


def read_excel(path: str = '',
               file_name: str = 'FactorDocument.xlsx'):

    print(f'file path: {path}, file name: {file_name}')

    file_path = os.path.join(path, file_name)
    factor_info_df = pd.read_excel(file_path, sheet_name='Factor Database')

    return factor_info_df


def save_json(json_name: str = 'factor_info.json'):
    try:
        info_df = read_excel()
    except Exception as e:
        print(f"file path error:{e}, update json file failed!")
    else:
        # switch dataframe to dict
        data_dict = info_df.set_index('Factor_ID').fillna("").to_dict('index')
        json_path = os.path.join(os.getcwd(), json_name)
        with open(json_path, mode="w+", encoding="UTF-8") as f:
            json.dump(data_dict,
                      f,
                      indent=4,
                      ensure_ascii=False)


if __name__ == '__main__':
    save_json()
