# -*-coding:utf-8-*-
# @Time:   2021/1/20 18:34
# @Author: FC
# @Email:  18817289038@163.com

from Analysis.FactorAnalysis.FactorAnalysis import *
import os


def Statistic():
    pass


def test(name: str, value_):
    factor_p = {"fact_name": name,
                "factor_params": {"switch": False},
                'db': 'TEC',
                'factor_value': value_,
                'cal': False}
    factor_process = {"outliers": '',  # mad
                      "neu": '',  # mv+industry
                      "stand": '',  # mv
                      }

    print(f"\033[1;31m{dt.datetime.now().strftime('%X')}: {f_name}\033[0m")
    Single_factor_test(params=factor_p,
                       process=factor_process,
                       hp=hp,
                       save=save)


def filter_(data):
    if len(data) < 5:
        data[~np.isnan(data)] = np.nan
    else:
        data = data
    return data


if __name__ == '__main__':
    # path_out = r'A:\Work\Working\8.文献阅读\DataInput'
    # LP = pd.read_csv(r"Y:\DataBase\AStockData.csv", usecols=['date', 'code', 'closeAdj', 'listBoard', 'totalMv', 'indexCode'])
    # SP = pd.read_csv(r"Y:\DataBase\StockAPI.csv")
    # LP['totalMv'] = LP['totalMv'] * 10000
    # LP = LP.set_index(['date', 'code'])
    # SP = SP.set_index(['date', 'code'])
    # df = pd.merge(LP, SP, on=['date', 'code'], how='right')
    # df['ret'] = df['closeAdj'].groupby('code').pct_change()
    # df.to_csv(os.path.join(path_out, 'data.csv'))

    v, t, p = {}, {}, {}
    for factor_file in ['Distribution008_1min_1days.csv', 'Distribution010_1min_1days.csv',
                        'Distribution015_1min_1days.csv',
                        'FundFlow003_1days.csv', 'FundFlow004_1days.csv', 'FundFlow006_0.2q_1days.csv',
                        'FundFlow012_1days.csv',
                        'FundFlow026_1days.csv', 'FundFlow027_1days.csv', 'FundFlow034_10min_C_1days.csv',
                        'FundFlow039_20days.csv',
                        'FundFlow040_20days.csv', 'VolPrice008_0.2q_1days.csv', 'VolPrice009_1days.csv',
                        'VolPrice013_1min_1days.csv',
                        'VolPrice017_1days.csv']:
        try:
            factor = factor_file[:-4]
            factor = factor.replace('_1days', '_5days')
            fact = pd.read_csv(os.path.join(r'C:\Users\Administrator\Desktop\Test', f"{factor}.csv"),
                               usecols=[f"G_{i}" for i in range(1, 11)] + ['date'])
            fact = fact.set_index('date')
            fact_ret = fact.pct_change()
            if fact['G_1'].iloc[-1] > fact['G_10'].iloc[-1]:
                portfolio = (fact_ret['G_1'] - fact_ret['G_10']).dropna()
            else:
                portfolio = (fact_ret['G_10'] - fact_ret['G_1']).dropna()
            portfolio = portfolio.to_frame('Net')
            portfolio['week'] = portfolio.index.map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d").weekday() + 1)
            X = pd.get_dummies(portfolio['week'], drop_first=True)
            Y = portfolio['Net']
            X = sm.add_constant(X)
            reg = sm.OLS(Y, X, hasconst=True).fit()
            v[factor] = reg.params
            t[factor] = reg.tvalues
            p[factor] = reg.pvalues
            # with open(f'C:\\Users\\Administrator\\Desktop\\Test\\{factor}.txt', 'w+') as f:
            #     f.write(str(reg.summary()))
        except Exception as e:
            print(e)

    V = pd.DataFrame(v).T
    T = pd.DataFrame(t)
    P = pd.DataFrame(p).T
    V.to_csv(f'C:\\Users\\Administrator\\Desktop\\Test\\Params.csv')
    P.to_csv(f'C:\\Users\\Administrator\\Desktop\\Test\\PValue.csv')
    print('s')

    # df = pd.read_csv(r'A:\Work\Working\8.文献阅读\DataInput\data.csv')

    # # 分年，各周收益率分布
    # df_ = df[df['stockPool_000300']]
    # fig = plt.figure(figsize=(20, 15))
    # k = 1
    # for i in [2012 + i_ for i_ in range(0, 9)]:
    #     for j in [1, 2, 3, 4, 5]:
    #         df_sub = df_[(df_['Year'] == i) & (df_['Week_day'] == j)]
    #         ax = fig.add_subplot(9, 5, k)
    #         df_sub['ret'].plot.hist(
    #             bins=np.sort([-i / 100 for i in range(1, 11)] + [0] + [i / 100 for i in range(1, 11)]), ax=ax)
    #         if (k - 1) % 5 == 0:
    #             ax.set_ylabel(i)
    #         else:
    #             ax.yaxis.label.set_visible(False)
    #
    #         if i == 2012:
    #             plt.title(f"周{j}")
    #         k += 1
    # plt.suptitle("000300")
    # plt.show()
    #
    # # 周收益率走势
    # week_ret = df.groupby(['Year_Week', 'Week_day'])['ret'].mean().unstack().fillna(0)
    # week_ret.index = week_ret.index.map(lambda x: x.split('_')[0] + '_' + f"{x.split('_')[-1]:0>2}")
    # week_ret = week_ret.sort_index()
    # week_nav = (week_ret + 1).cumprod()
    # week_ret['year'] = week_ret.index.map(lambda x: x.split('_')[0])
    # week_nav['year'] = week_nav.index.map(lambda x: x.split('_')[0])
    #
    # m, n = {}, {}
    # for k in [2012 + i_ for i_ in range(0, 9)]:
    #     data_sub = week_nav[week_nav['year'] == str(k)]
    #     F, P = pd.DataFrame(), pd.DataFrame()
    #     for i in [1, 2, 3, 4, 5]:
    #         for j in [1, 2, 3, 4, 5]:
    #             F.loc[i, j], P.loc[i, j] = stats.f_oneway(data_sub[i], data_sub[j])
    #     m[k] = F
    #     n[k] = P
    #
    # fig = plt.figure(figsize=(18, 18))
    # k = 1
    # for i in n.keys():
    #     ax = fig.add_subplot(3, 3, k)
    #     sns.heatmap(n[i], annot=True, cmap="YlGnBu")
    #     plt.title(i)
    #     k += 1
    # plt.suptitle("P_nav")
    # plt.show()
    # print('s')
