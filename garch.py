import pandas as pd
import numpy as np
from scipy import optimize
import random
from scipy.stats import norm
#import matplotlib as plt
#import seaborn as sns
from pymongo import MongoClient

class GARCH(object):
    def __init__(self, gamma, alpha):
        self.alpha = alpha
        self.gamma = gamma
        self.beta = 1 - alpha

    def step(self, log_return, previous_var):
        return self.gamma + self.alpha*log_return**2 + self.beta*previous_var

#class MCS(object):


def data_call(a,b):
    connection = MongoClient()
    db = connection.ccm
    data = list(db.monte_carlo.find({}, {'_id': 0, 'Future': 1, 'AtM': 1, 'Date': 1}).sort('Date'))
    df_columns = ['Date', 'Future', 'AtM', 'log_returns', 'AtM_Var', 'Norm_AtM_Vol',
                  'stochastic_factor', 'simulated_var', 'simulated_AtMVol', 'simulated_futures',
                  'simulated_log_returns', 'error', 'abs_error', 'vol_error', 'abs_vol_error']

    df = pd.DataFrame(data, columns=df_columns)
    df.set_index('Date', inplace=True)
    #df = df.loc['2009-06-15':'2014-12-31']
    df = df.loc[a:b]

    df.log_returns = np.log(df.Future / df.Future.shift())
    df.AtM_Var = df.AtM * df.AtM
    df.Norm_AtM_Vol = df.AtM / 16
#    log_mean = df.log_returns.mean()
    # norm_AtMVol = df.Norm_AtM_Vol.mean()
#    df.stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol

    return df


def mean_stochastic_factor(df, df1):
    log_mean = df.log_returns.mean()
    # norm_AtMVol = df.Norm_AtM_Vol.mean()
    df.stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol
    df1.stochastic_factor = (df1.log_returns - log_mean) / df1.Norm_AtM_Vol

    return df, df1


def calc(guess, df, a): #a = indicator for observation(1) or backtesting(0)

    gamma = guess[0]
    alpha = guess[1]
    log_mean = df.log_returns.mean()

    var = df.AtM_Var[a]
    atm_vol = df.AtM[a]
    future = df.Future[a]
    log_return = df.log_returns[a]

    simulated_var = [var, ]
    simulated_future = [future, ]
    simulated_log_return = [log_return, ]
    simulated_atm_vol = [atm_vol, ]

    garch = GARCH(gamma, alpha)

    for stochastic in df.stochastic_factor[a+1:]:
        var = garch.step(log_return, var)
        atm_vol = np.sqrt(var)
        log_return = log_mean + stochastic * atm_vol/16
        future = future*np.exp(log_return)
        simulated_future.append(future)
        simulated_var.append(var)
        simulated_atm_vol.append(atm_vol)
        simulated_log_return.append(log_return)

    error = np.array([i - j for i, j in zip(df.log_returns[a:], simulated_log_return)])
    vol_error = np.array([i - j for i, j in zip(df.AtM[a:], simulated_atm_vol)])
    abs_error = abs(error)
    abs_vol_error = abs(vol_error)

    df.simulated_futures[a:] = simulated_future
    df.simulated_AtMVol[a:] = simulated_atm_vol
    df.simulated_log_returns[a:] = simulated_log_return
    df.simulated_var[a:] = simulated_var

    df.error[a:] = error
    df.abs_error[a:] = abs_error
    df.vol_error[a:] = vol_error
    df.abs_vol_error[a:] = abs_vol_error

    abs_mean = df.abs_error.mean()
    df.plot(y = 'error')
    #df.plot(x=df.Date, y='error')#, y='vol_error')
    return abs_mean, df


def real_stochastic(df1):

    factor = [random.random() for i in range(1, len(df1.index))]
    factor_inv = np.array([norm.ppf(factor[i]) for i in range(0, len(factor))])
    df1.stimulated_stochastic_factor[1:] = factor_inv

    return df1

def main():
    df_list = ['2009-06-15', '2014-12-31', '2016-12-13']
    df = data_call(df_list[0], df_list[1])
    # gamma = 0.0003343
    # alpha = 0.0072841
    df1 = data_call(df_list[1], df_list[0])

    df, df1 = mean_stochastic_factor(df, df1)

    #df1 = real_stochastic(df1)

    initial_guess = np.array([0, 0])

    cons = [{'type': 'ineq', 'fun': lambda x: x - 0},
            {'type': 'ineq', 'fun': lambda x: x - 0}]

    result, df = optimize.minimize(calc, initial_guess, args=(df, 1, ), constraints=cons)

    #df.plot(y='error')
    df1, back_test = calc(result, df1, 0)

    return result, back_test


if __name__ == '__main__':
    sols, backsols = main()
    print(sols, backsols)
