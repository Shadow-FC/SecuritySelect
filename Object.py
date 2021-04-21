# -*-coding:utf-8-*-
# @Time:   2020/9/11 10:17
# @Author: FC
# @Email:  18817289038@163.com

import yagmail
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

from typing import Union, Any


@dataclass
class GroupData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    stock_id: str = ''
    industry: str = ''
    date: datetime = None
    datetime_update: datetime = None
    group: int = None

    stock_return: float = None
    holding_period: int = None
    factor_name: str = None
    factor_name_chinese: str = None
    factor_value: float = None
    factor_type: str = None  # 因子类型


@dataclass
class FactorRetData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    date: datetime = None
    datetime_update: datetime = None

    factor_return: float = None
    holding_period: int = None
    factor_T: float = None
    factor_name: str = None
    factor_name_chinese: str = None
    ret_type: str = None  # 因子收益类型


@dataclass
class FactorData(object):
    """
    Candlestick bar data of a certain trading period.
    """

    stock_id: str = ''
    date_report: datetime = None  # 报告期
    date: datetime = None  # 公布期(数据实际获得的日期)
    datetime_update: datetime = None

    factor_name: str = None
    factor_name_chinese: str = None
    factor_category: str = None
    factor_value: float = None
    factor_type: str = None  # 因子类型


# 因子数据的存储
@dataclass
class DataInfo(object):
    """
    为方便数据的流转和统一定义数据集
    数据集说明
    1.原始数据必须要有存放在data_raw中；
    2.对于加工后的数据或者后续计算所需数据则放入data属性中；
    3.数据大类data_category用来识别该数据属于什么类型的数据：例如股票池，标签池，因子池等；
    4.数据名称data_name用来给数据命名：例如PB，StockPoolZD；
    5.数据子类data_type用来对数据进行子类标识：例如BF, HFD

    """

    data_raw: Union[pd.DataFrame, pd.Series] = None  # 因子[双索引[股票ID， 交易日],...]
    data: Union[pd.DataFrame, pd.Series] = None  # 因子[双索引[股票ID， 交易日]：因子值]

    data_category: str = None
    data_name: str = None
    data_type: str = None  # 因子类型


# 发送邮件
def send_email(email, theme, contents):
    """

    :param email:
                {"person_name": {"user": "email_address",
                                 "password": "password",
                                 "host": "smtp.qq.com"}}
    :param theme: email theme
    :param contents: email contents
    :return:
    """

    for person in email.keys():
        user = email[person]['user']
        password = email[person]['password']
        host = email[person]['host']
        try:
            yag = yagmail.SMTP(user=user,
                               password=password,
                               host=host)

            yag.send([user], theme, contents)
        except:
            # Alternate mailbox
            yag = yagmail.SMTP(user="18817289038@163.com", password="excejuxyyuthbiaa",
                               host="smtp.qq.com")
            yag.send([user], theme, contents)
