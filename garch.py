import pandas as pd
import numpy as np
from scipy import optimize
import random
from scipy.stats import norm
import xlwings as xl
import matplotlib.pyplot as plt
#import seaborn as sns
from pymongo import MongoClient


class GARCH(object):
    def __init__(self, gamma, alpha):
        super(GARCH, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = 1 - alpha

    def step(self, log_return, previous_var):
        return self.gamma + self.alpha*log_return**2 + self.beta*previous_var


class StochasticProcess(object):
    def __init__(self, mu):
        super(StochasticProcess, self).__init__()
        self.mu = mu
        self.f0 = 1

    def step(self, sigma):
        stochastic = norm.ppf(random.random())
        f1 = self.f0*np.exp(self.mu + stochastic*sigma/np.sqrt(252))
        self.f0 = f1
        return f1


def data_call(start_date, end_date):
    connection = MongoClient()
    db = connection.ccm
    data = list(db.monte_carlo.find({}, {'_id': 0, 'Future': 1, 'AtM': 1, 'Date': 1}).sort('Date'))
    df_columns = ['Date', 'Future', 'AtM', 'log_returns', 'AtM_Var', 'Norm_AtM_Vol',
                  'stochastic_factor', 'simulated_var', 'simulated_AtMVol', 'simulated_futures',
                  'simulated_log_returns', 'error', 'abs_error', 'vol_error', 'abs_vol_error']

    df = pd.DataFrame(data, columns=df_columns)
    df.set_index('Date', inplace=True)
    df = df.loc[start_date:end_date]

    df.log_returns = np.log(df.Future / df.Future.shift())
    df.AtM_Var = df.AtM * df.AtM
    df.Norm_AtM_Vol = df.AtM / 16
    # norm_AtMVol = df.Norm_AtM_Vol.mean()
    log_mean = df.log_returns.mean()
    df.stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol
    return df


# def mean_stochastic_factor(df, df1):
#     log_mean = df.log_returns.mean()
#     # norm_AtMVol = df.Norm_AtM_Vol.mean()
#     df.stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol
#     df1.stochastic_factor = (df1.log_returns - log_mean) / df1.Norm_AtM_Vol
#
#     return df, df1


def calc(guess, df, log_mean, stochastic_factors, indicator): #a = indicator for observation(1) or backtesting(0)
    assert len(df) > 1
    gamma = guess[0]
    alpha = guess[1]

    var = df.AtM_Var[1]
    atm_vol = df.AtM[1]
    future = df.Future[1]
    log_return = df.log_returns[1]

    simulated_var = [var, ]
    simulated_future = [future, ]
    simulated_log_return = [log_return, ]
    simulated_atm_vol = [atm_vol, ]

    garch = GARCH(gamma, alpha)

    for stochastic in stochastic_factors:
        var = garch.step(log_return, var)
        atm_vol = np.sqrt(var)
        if np.isnan(atm_vol):
            return np.inf
        log_return = log_mean + stochastic * atm_vol/16
        future = future*np.exp(log_return)
        simulated_future.append(future)
        simulated_var.append(var)
        simulated_atm_vol.append(atm_vol)
        simulated_log_return.append(log_return)

    if indicator != 'forecasting':
        error = np.array([i - j for i, j in zip(df.log_returns[1:], simulated_log_return)])
        vol_error = np.array([i - j for i, j in zip(df.AtM[1:], simulated_atm_vol)])
        abs_error = abs(error)
        abs_vol_error = abs(vol_error)

        df.simulated_futures[1:] = simulated_future
        df.simulated_AtMVol[1:] = simulated_atm_vol
        df.simulated_log_returns[1:] = simulated_log_return
        df.simulated_var[1:] = simulated_var

        df.error[1:] = error
        df.abs_error[1:] = abs_error
        df.vol_error[1:] = vol_error
        df.abs_vol_error[1:] = abs_vol_error

        mean_abs_error = df.abs_error.mean()
        return mean_abs_error

    else:
        df_columns = ['Futures', 'AtM', 'Var', 'log_returns']
        df_fore = pd.DataFrame(np.zeros(shape = (100,4)),columns=df_columns)

        df_fore.Futures[0:] = simulated_future
        df_fore.AtM[0:] = simulated_atm_vol
        df_fore.Var[0:] = simulated_var
        df_fore.log_returns[0:] = simulated_log_return

        plt.figure('Futures')
        plt.plot(simulated_future)
        plt.figure('AtM')
        plt.plot(simulated_atm_vol)
        plt.figure('Var')
        plt.plot(simulated_var)
        plt.figure('Log Returns')
        plt.plot(simulated_log_return)


        return df_fore


def simulated_stochastic(length):
    factor = [random.random() for i in range(length-1)]
    factor_inv = np.array([norm.ppf(i) for i in factor])
    return factor_inv


if __name__ == '__main__':
    df_list = ['2009-06-15', '2014-12-31', '2016-12-13']
    df = data_call(df_list[0], df_list[1])

    df1 = data_call(df_list[1], df_list[2])

    df2 = data_call('2016-12-10', '2016-12-13')

    log_mean = df.log_returns.mean()
    # gamma = 0.0003343
    # alpha = 0.0072841

    # df1 = data_call(df_list[1], df_list[0])
    #
    # df, df1 = mean_stochastic_factor(df, df1)

    # df1 = real_stochastic(df1)

    initial_guess = np.array([0, 0])

    cons = [{'type': 'ineq', 'fun': lambda x: x - 0},
            {'type': 'ineq', 'fun': lambda x: x - 0}]

    result = optimize.minimize(calc, initial_guess, args=(df.copy(True), log_mean, df.stochastic_factor[2:], 'observation'),
                               constraints=cons)

    calc(result.x, df, log_mean, df.stochastic_factor[2:], 'observation')
    print(result.x)
    # df.plot(y='error')
    # df1, back_test = calc(result, df1, 0)

    stochastic = simulated_stochastic(len(df1) - 1)

    calc(result.x, df1, log_mean, stochastic, 'backtesting')

    stochastic_forecasting = simulated_stochastic(100)

    df_fore = calc(result.x, df2, log_mean, stochastic_forecasting, 'forecasting')


    file_name = ('futures.xlxs')
    with open(file_name, 'w') as f:
        book = xl.Book(file_name)
        book.activate()
        sheet = book.sheets[0]
        sheet.range('A1').value = df_fore
        book.save()
        book.close()

    plt.show()
