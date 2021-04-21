import numpy as np
import pandas as pd
import time
from typing import Union, Dict, Any

from DataAPI.FactorAPI.FactorBase import FactorBase

from utility.utility import (
    Process
)
from constant import (
    KeyName as KN,
    SpecialName as SN,
    PriceVolumeName as PVN,
    FinancialBalanceSheetName as FBSN,
)


class TechnicalBehaviorFactor(FactorBase):

    def __init__(self):
        super(TechnicalBehaviorFactor, self).__init__()

    @classmethod
    @Process("BF")
    def TK_equal(cls,
                 data: pd.DataFrame,
                 n: int = 20,
                 **kwargs) -> Dict[str, Any]:
        """
        TK价值因子
        :return:
        """

        fact_name = kwargs['name'] + f'_{n}days'
        data = cls().reindex(data)

        data[KN.RETURN.value] = data.groupby(KN.STOCK_ID.value,
                                             group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
        data['v'] = cls.valueFunc(data[KN.RETURN.value])

        data[fact_name] = data['v'].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: cls().TKAlgo(x, n))

        return {"data": data[fact_name], "name": fact_name}

    @classmethod
    @Process("BF")
    def TK_BP_abs(cls,
                  data: pd.DataFrame,
                  n: int = 20,
                  **kwargs) -> Dict[str, Any]:
        """
        TK价值因子:采用BP替代股价
        :return:
        """
        fact_name = kwargs['name'] + f'_{n}days'

        data['BP'] = abs(data[FBSN.Net_Asset_Ex.value]) / data[PVN.TOTAL_MV.value]

        data = cls().reindex(data)

        data[KN.RETURN.value] = data.groupby(KN.STOCK_ID.value, group_keys=False)["BP"].pct_change(fill_method=None)
        data['v'] = cls.valueFunc(data[KN.RETURN.value])

        data[fact_name] = data['v'].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: cls().TKAlgo(x, n))

        return {"data": data[fact_name], "name": fact_name}

    @classmethod
    @Process("BF")
    def TK_PB_abs(cls,
                  data: pd.DataFrame,
                  n: int = 20,
                  **kwargs) -> Dict[str, Any]:
        """
        TK价值因子:采用BP替代股价
        :return:
        """
        fact_name = kwargs['name'] + f'_{n}days'

        data['PB'] = data[PVN.TOTAL_MV.value] / abs(data[FBSN.Net_Asset_Ex.value])

        data = cls().reindex(data)

        data[KN.RETURN.value] = data.groupby(KN.STOCK_ID.value, group_keys=False)["PB"].pct_change(fill_method=None)
        data['v'] = cls.valueFunc(data[KN.RETURN.value])

        data[fact_name] = data['v'].groupby(KN.STOCK_ID.value, group_keys=False).apply(lambda x: cls().TKAlgo(x, n))

        return {"data": data[fact_name], "name": fact_name}

    @classmethod
    @Process("BF")
    def CGO_Price(cls,
                  data: pd.DataFrame,
                  n: int = 20,
                  **kwargs) -> Dict[str, Any]:
        """
        CGO因子(未实现盈利值)
        :return:
        """

        fact_name = kwargs['name'] + f'_{n}days'
        data = cls().reindex(data)

        data['TO'] = data[PVN.AMOUNT.value] / data[PVN.LIQ_MV.value]
        # 换手率超过1，认为当天股票全换手，未换手率为0
        data.loc[data['TO'] > 1, 'TO'] = 1

        data['RP'] = data.groupby(KN.STOCK_ID.value,
                                  group_keys=False).apply(lambda x: cls().RPAlgo(x, PVN.CLOSE.value, n))
        data[fact_name] = (data[PVN.CLOSE.value] - data['RP']) / data[PVN.CLOSE.value]

        return {"data": data[fact_name], "name": fact_name}

    @classmethod
    @Process("BF")
    def CGO_Ret(cls,
                data: pd.DataFrame,
                n: int = 20,
                **kwargs) -> Dict[str, Any]:
        """
        CGO因子(未实现盈利值)
        :return:
        """

        fact_name = kwargs['name'] + f'_{n}days'
        data = cls().reindex(data)

        data['TO'] = data[PVN.AMOUNT.value] / data[PVN.LIQ_MV.value]
        # 换手率超过1，认为当天股票全换手，未换手率为0
        data.loc[data['TO'] > 1, 'TO'] = 1
        data[KN.RETURN.value] = data.groupby(KN.STOCK_ID.value,
                                             group_keys=False)[PVN.CLOSE.value].pct_change(fill_method=None)
        data['RP'] = data.groupby(KN.STOCK_ID.value,
                                  group_keys=False).apply(lambda x: cls().RPAlgo(x, KN.RETURN.value, n))
        data[fact_name] = data[KN.RETURN.value] - data['RP']

        return {"data": data[fact_name], "name": fact_name}

    @classmethod
    def TK_equal_data_raw(cls, **kwargs) -> pd.DataFrame:
        price_data = cls()._csv_data(data_name=[PVN.CLOSE_ADJ.value],
                                     file_name='AStockData')
        price_data = price_data.rename(columns={PVN.CLOSE_ADJ.value: PVN.CLOSE.value})
        return price_data

    @classmethod
    def TK_PB_abs_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20201231,
                           f_type: str = '408001000',
                           **kwargs) -> pd.DataFrame:
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)
        financial_data = financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value])
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        # switch freq
        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value],
                                     file_name='AStockData')
        price_data[PVN.TOTAL_MV.value] = price_data[PVN.TOTAL_MV.value] * 10000
        price_data = price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        # merge data
        res = pd.concat([financial_data[FBSN.Net_Asset_Ex.value], price_data], axis=1)
        res = res.dropna(subset=[FBSN.Net_Asset_Ex.value, PVN.TOTAL_MV.value])
        res = res.reset_index()
        return res

    @classmethod
    def TK_BP_abs_data_raw(cls,
                           sta: int = 20130101,
                           end: int = 20201231,
                           f_type: str = '408001000',
                           **kwargs) -> pd.DataFrame:
        sql_keys = {"BST": {"TOT_SHRHLDR_EQY_EXCL_MIN_INT": f"\"{FBSN.Net_Asset_Ex.value}\""}
                    }

        sql_ = cls().Q.finance_SQL(sql_keys, sta, end, f_type)
        financial_data = cls().Q.query(sql_)

        # TTM
        financial_ttm = cls()._switch_ttm(financial_data, FBSN.Net_Asset_Ex.value)
        financial_data = financial_data.set_index([SN.REPORT_DATE.value, KN.STOCK_ID.value])
        financial_data[FBSN.Net_Asset_Ex.value] = financial_ttm

        # switch freq
        financial_data = cls()._switch_freq(data_=financial_data, name=FBSN.Net_Asset_Ex.value, limit=120)

        price_data = cls()._csv_data(data_name=[PVN.TOTAL_MV.value],
                                     file_name='AStockData')
        price_data[PVN.TOTAL_MV.value] = price_data[PVN.TOTAL_MV.value] * 10000
        price_data = price_data.set_index([KN.TRADE_DATE.value, KN.STOCK_ID.value])

        # merge data
        res = pd.concat([financial_data[FBSN.Net_Asset_Ex.value], price_data], axis=1)
        res = res.dropna(subset=[FBSN.Net_Asset_Ex.value, PVN.TOTAL_MV.value])
        res = res.reset_index()
        return res

    @classmethod
    def CGO_Price_data_raw(cls, **kwargs) -> pd.DataFrame:
        price_data = cls()._csv_data(data_name=[PVN.CLOSE_ADJ.value, PVN.AMOUNT.value, PVN.LIQ_MV.value],
                                     file_name='AStockData')
        price_data = price_data.rename(columns={PVN.CLOSE_ADJ.value: PVN.CLOSE.value})
        price_data[PVN.LIQ_MV.value] = price_data[PVN.LIQ_MV.value] * 10000
        price_data[PVN.AMOUNT.value] = price_data[PVN.AMOUNT.value] * 10000
        return price_data

    @classmethod
    def CGO_Ret_data_raw(cls, **kwargs) -> pd.DataFrame:
        return cls.CGO_Price_data_raw(**kwargs)

    def TKAlgo(self, data: pd.Series, n: int) -> pd.Series:
        sta = time.time()
        if data.shape[0] >= n:
            df_new = self.switchForm(data, n)

            # 生成排名
            df_rank = df_new.rank(axis=1)
            # 有效样本数
            df_max = df_rank.max(axis=1)
            # 根据排名生成累积概率
            prob = df_rank.div(df_max, axis=0)
            prob[df_new >= 0] = (- prob[df_new >= 0]).add((df_max + 1) / df_max, axis=0)
            probSub = prob.sub(1 / df_max, axis=0)

            # 生成权重
            prob[df_new >= 0] = self.weightFunc(prob[df_new >= 0], 0.61)
            probSub[df_new >= 0] = self.weightFunc(probSub[df_new >= 0], 0.61)

            prob[df_new < 0] = self.weightFunc(prob[df_new < 0], 0.69)
            probSub[df_new < 0] = self.weightFunc(probSub[df_new < 0], 0.69)

            weight = prob - probSub

            # 加权
            tk = (weight * df_new).sum(axis=1)
            # 样本量不足80%剔除
            tk[df_max < round(n * 0.8)] = np.nan
            print(f"{data.index[0][1]}: {time.time() - sta}")
            return tk

    def RPAlgo(self, data: pd.Series, colName: str, n: int) -> pd.Series:
        """
        参考点的计算：未流通股流通率加权
        """
        if data.shape[0] >= n:
            TO_new = self.switchForm(data['TO'], n)
            close_new = self.switchForm(data[colName], n)

            no_TO = (1 - TO_new.shift(1, axis=1)).cumprod(axis=1)

            # 生成权重
            weight = TO_new.mul(no_TO)
            weight = weight.div(weight.sum(axis=1), axis=0)

            RP = weight.mul(close_new).sum(axis=1)

            RP[TO_new.count(axis=1) < round((n - 1) * 0.8)] = np.nan
            return RP

    def piFuncEqual(self, data: pd.DataFrame, name: str) -> Union[float, None]:
        dataNew = pd.DataFrame(data, columns=[name]).dropna()
        gain_TK, loss_TK = 0, 0
        if not dataNew.empty:
            sta = time.time()
            dataNew['p'] = 1 / dataNew.shape[0]

            # gain and loss split
            gain = dataNew[dataNew[name] >= 0].sort_values(by=name, ascending=False)
            loss = dataNew[dataNew[name] < 0].sort_values(by=name, ascending=True)

            if not gain.empty:
                gain['cumProb'] = gain['p'].cumsum()
                # probability-weight mapping
                gain['w'] = self.weightFunc(gain['cumProb'], 0.61)
                # gain['w'] = gain['cumProb'].map(lambda x: self.weightFunc(x, 0.61))
                gain['pi'] = gain['w'] - gain['w'].shift(1).fillna(0)
                gain_TK = gain[name] @ gain['pi']

            if not loss.empty:
                loss['cumProb'] = loss['p'].cumsum()
                loss['w'] = self.weightFunc(loss['cumProb'], 0.69)
                # loss['w'] = loss['cumProb'].map(lambda x: self.weightFunc(x, 0.69))
                loss['pi'] = loss['w'] - loss['w'].shift(1).fillna(0)
                loss_TK = loss[name] @ loss['pi']

            res = gain_TK + loss_TK
            print(f"IN:{time.time() - sta}")
            return 0

    @staticmethod
    def valueFunc(data: pd.Series,
                  alpha: float = 0.88,
                  lamb: float = 2.25) -> pd.Series:
        """
        价值函数：
            以0为界限的向x轴两端逐渐衰减的函数
            大于0部分函数初始斜率小于0.5，小于0部分函数初始斜率大于0.5
        Args:
            data ():
            alpha ():衰减系数
            lamb (): 风险厌恶系数
        Returns:

        """
        res = data.mask(data >= 0, lambda x: x ** alpha)
        res = res.mask(res < 0, lambda x: - lamb * (- x) ** alpha)
        return res

    @staticmethod
    def weightFunc(x: Union[pd.Series, pd.DataFrame],
                   alpha: float) -> Union[pd.Series, pd.DataFrame]:
        """
        权重函数
        Args:
            x ():
            alpha (): 衰减系数
        Returns:
        """
        up = pow(x, alpha)
        down = pow((pow(x, alpha) + pow(1 - x, alpha)), (1 / alpha))
        res = up / down
        return res


if __name__ == '__main__':
    # df_stock = pd.read_csv("D:\\Quant\\SecuritySelect\\DataInput\\AStockData.csv")
    #
    # # DataInput cleaning:Restoration stock price [open, high, low, close]
    # price_columns = ['open', 'close', 'high', 'low']
    # df_stock[price_columns] = df_stock[price_columns].multiply(df_stock['adjfactor'], axis=0)
    # A = MomentFactor(df_stock[price_columns])

    pass
