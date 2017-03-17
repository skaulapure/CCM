from scipy.stats import norm
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


# Pricing engine module that calculates Option prices and Greeks with black scholes
# The Black Scholes Formula
"""
# CallPutFlag - This is set to 'c' for call option, anything else for put
# S - Stock price
# K - Strike price
# T - Time to maturity
# r - Riskfree interest rate
# d - Dividend yield
# v - Volatility
"""


def BlackScholes(CallPutFlag, S, K, T, r, d, v):
    d1 = (math.log(float(S) / K) + ((r - d) + v * v / 2.) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    if CallPutFlag == 'c':
        return S * math.exp(-d * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-d * T) * norm.cdf(-d1)


def simulated_stochastic(length):
    factor = [random.random() for i in range(length-1)]
    factor_inv = np.array([norm.ppf(i) for i in factor])
    return factor_inv


def cal(var, future, log_return, atm_vol, log_mean):

    gamma = 0.00031575
    alpha = 0.00690033
    beta = 1 - alpha

    simulated_var = [var, ]
    simulated_future = [future, ]
    simulated_log_return = [log_return, ]
    simulated_atm_vol = [atm_vol, ]

    t1 = 100

    stoc_factors = simulated_stochastic(t1)

    for stochastic in stoc_factors:
        var = gamma + alpha * log_return ** 2 + beta * var

        atm_vol = np.sqrt(var)
        if np.isnan(atm_vol):
            return np.inf
        log_return = log_mean + stochastic * atm_vol / 16
        future = future * np.exp(log_return)
        simulated_future.append(future)
        simulated_var.append(var)
        simulated_atm_vol.append(atm_vol)
        simulated_log_return.append(log_return)

    return future, atm_vol


def plot(s, v, c, p, straddle):
    plt.figure('Futures')
    plt.hist(s, facecolor='green')
    plt.figure('Vol')
    plt.hist(v)
    plt.figure('Call')
    plt.hist(c)
    plt.figure('Put')
    plt.hist(p)
    plt.figure('Straddle')
    plt.hist(straddle)

    return


def simulation_lists(simulations, var, future, log_return, atm_vol, log_mean, K, r, d):
    c = []
    p = []
    fut = []
    vol = []
    straddle = []

    for i in range(0, simulations):
        S, v = cal(var, future, log_return, atm_vol, log_mean)
        fut.append(S)
        vol.append(v)
        c.append(BlackScholes('c', S, K, 100/252.0, r, d, v))
        p.append(BlackScholes('p', S, K, 100/252.0, r, d, v))
        straddle.append(c[i] + p[i])

    df_columns = ['Futures', 'Vol', 'Call', 'Put', 'Straddle']
    df_fore = pd.DataFrame(np.zeros(shape=(simulations, 5)), columns=df_columns)

    df_fore.Futures[0:] = fut
    df_fore.Vol[0:] = vol
    df_fore.Call[0:] = c
    df_fore.Put[0:] = p
    df_fore.Straddle[0:] = straddle

    df_fore.to_csv('futures_options.csv')
    plot(fut, vol, c, p, straddle)

    return


def ext():
    file_name = 'futures1.xlsx'
    df_fore = pd.read_excel(file_name)

    var = df_fore.Var[0]  # Volatility
    log_return = df_fore.log_returns[0]
    atm_vol = df_fore.AtM[0]
    future = df_fore.Futures[0]  # Stock price
    K = df_fore.Futures[0]  # Strike price
    print(K)
    log_mean = df_fore.log_returns.mean()
    return var, future, log_return, atm_vol, log_mean, K


def main():
    r = 0.0075  # Riskfree interest rate
    d = 0.00  # Dividend yield
    simulations = 100
    var, future, log_return, atm_vol, log_mean, K = ext()
    simulation_lists(simulations, var, future, log_return, atm_vol, log_mean, K, r, d)
    plt.show()

    return

if __name__ == "__main__":
    main()
