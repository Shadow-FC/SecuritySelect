import os
import json
import datetime as dt
from typing import List, Any


def saveData(DBName: str,
             position: str,
             data: Any,
             fileName: str = None,
             **kwargs):
    """
    :param DBName:
    :param position: 数据存储位置
    :param data: 数据
    :param fileName:
    :return:
    """

    # 存储csv
    if DBName == 'csv':
        # print(f"{dt.datetime.now()} save data to csv：{position}")
        # 生成路径
        if not os.path.exists(position):
            os.makedirs(position)

        data.to_csv(os.path.join(position, fileName + '.csv'), index=False)
    elif DBName == 'pkl':
        # print(f"{dt.datetime.now()} save data to pkl：{position}")
        # 生成路径
        if not os.path.exists(position):
            os.makedirs(position)

        data.to_pickle(os.path.join(position, fileName + '.pkl'))

    elif DBName == 'json':
        # print(f"{dt.datetime.now()} save data to json：{position}")
        # 生成路径
        if not os.path.exists(position):
            os.makedirs(position)

        jsonPath = os.path.join(position, fileName + '.json')

        dataOld = readJson(position, fileName)
        dataOld = dataOld + data if isinstance(data, list) else dataOld + [data]
        dataNew = list(set(dataOld))

        with open(jsonPath, mode="w+") as f:
            json.dump(dataNew,
                      f,
                      indent=4,
                      ensure_ascii=False)

    elif DBName == 'txt':
        if not os.path.exists(position):
            os.makedirs(position)

        with open(os.path.join(position, fileName + '.txt'), mode="a", encoding='UTF-8') as f:
            f.writelines(data + "\n")


# 读取json文件
def readJson(position: str, name: str) -> List[str]:
    filePath = os.path.join(position, name + '.json')
    if os.path.exists(filePath):
        with open(filePath, mode='r') as f:
            data = json.load(f)
    else:
        data = []
    return data
