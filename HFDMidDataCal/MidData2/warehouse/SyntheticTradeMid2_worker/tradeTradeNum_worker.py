# -*-coding:utf-8-*-
# @Time:   2021/3/24 15:57
# @Author: FC
# @Email:  18817289038@163.com

import numpy as np
import pandas as pd
from mapping import range_T


def tradeTradeNum_worker(data: pd.DataFrame,
                         date: str,
                         **kwargs) -> pd.DataFrame:

    def cal(data_: pd.DataFrame) -> pd.Series:
        d_sub = data_[range_T]
        d_sub['ret'] = d_sub['close'].pct_change()

        tradeNumDiff = d_sub['tradenum'].diff(1)

        tradeNumData = {
            "tradeNumRetUpSum_0": d_sub[d_sub['ret'] > 0]['tradenum'].sum(),  # 收益率大于0的笔数和
            "tradeNumRetDownSum_0": d_sub[d_sub['ret'] < 0]['tradenum'].sum(),  # 收益率小于0的笔数和
            "tradeNumRetEqualSum_0": d_sub[np.isnan(d_sub['ret']) | (d_sub['ret'] == 0)]['tradenum'].sum(),
            # 收益率等于0的笔数和(包含收益率为空的数据)

            "tradeNumDiffMean": tradeNumDiff.mean(),  # 成交笔数差分均值
            "tradeNumDiffStd": tradeNumDiff.std(),  # 成交笔数差分标准差

            "tradeNumDiffAbsMean": abs(tradeNumDiff).mean(),  # 成交笔数差分绝对值均值
            "tradeNumDiffAbsStd": abs(tradeNumDiff).std(),  # 成交笔数差分绝对值标准差

            "date": date  # 日期
        }

        return pd.Series(tradeNumData)
    data = data.iloc[:50000]
    res = data.groupby('code').apply(cal)
    return res


if __name__ == '__main__':
    print('s')
