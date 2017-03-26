from scipy.stats import norm
#import matplotlib.mlab as mlab
from decimal import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import profile


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
    #print(T)
    d1 = (math.log(float(S) / K) + ((r - d) + v * v / 2.) * float(T)) / (v * math.sqrt(float(T)))
    d2 = d1 - v * math.sqrt(T)
    if CallPutFlag == 'c':
        return S * math.exp(-d * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-d * T) * norm.cdf(-d1)


def simulated_stochastic(length):
    factor = [random.random() for i in range(length-1)]
    factor_inv = np.array([norm.ppf(i) for i in factor])
    return factor_inv


def cal(var, future, log_return, log_mean, fut, vol, i, c, p, K, r, d, straddle): #, c_in, p_in):

    gamma = 0.00031575
    alpha = 0.00690033
    beta = 1 - alpha

    t1 = 100

    stoc_factors = simulated_stochastic(t1)

#    for stochastic in stoc_factors:
    for j in range(1, len(stoc_factors)+1):

        var = gamma + alpha * log_return ** 2 + beta * var

        atm_vol = np.sqrt(var)

        if np.isnan(atm_vol):
            return np.inf

        log_return = log_mean + stoc_factors[j-1] * atm_vol / 16

        future = future * np.exp(log_return)

        fut[i, j] = future
        vol[i, j] = atm_vol
        c[i, j] = BlackScholes('c', fut[i, j], K, (100 - j)/252.0, r, d, vol[i,j])
        p[i, j] = BlackScholes('p', fut[i, j], K, (100 - j)/ 252.0, r, d, vol[i, j])
        straddle[i, j] = c[i,j] + p[i, j]

    return fut, vol, c, p, straddle


def plot(fut, vol, c, p, straddle):
    plt.figure('Futures')
    for i in fut:
        plt.plot(i)

    plt.figure('Volatility')
    for i in vol:
        plt.plot(i)

    plt.figure('Call')
    for i in c:
        plt.plot(i)

    plt.figure('Put')
    for i in p:
        plt.plot(i)

    plt.figure('Straddle')
    for i in straddle:
        plt.plot(i)
    return


def simulation_lists(simulations, r, d, var, future, log_return, atm_vol, log_mean, K): #, c_in, p_in):
    duration = 100
    c = np.zeros(shape=(simulations, duration))
    p = np.zeros(shape=(simulations, duration))
    fut = np.zeros(shape=(simulations, duration))
    vol = np.zeros(shape=(simulations, duration))
    straddle = np.zeros(shape=(simulations, duration))

    for i in range(0, simulations):

        fut[i, 0] = future
        vol[i, 0] = atm_vol
        c[i, 0] = BlackScholes('c', fut[i, 0], K, 100/252.0, r, d, vol[i,0])
        p[i, 0] = BlackScholes('p', fut[i, 0], K, 100 / 252.0, r, d, vol[i, 0])
        straddle[i, 0] = c[i, 0] + p[i, 0]
        fut, vol, c, p, straddle = cal(var, fut[i,0], log_return, log_mean, fut, vol, i, c, p, K, r, d, straddle) #, c_in, p_in)

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
    log_mean = df_fore.log_returns.mean()
    return var, future, log_return, atm_vol, log_mean, K


def main():
    r = 0.0075  # Riskfree interest rate
    d = 0.00  # Dividend yield
    simulations = 100
  #  c_in = input('Enter call value: ')
 #   p_in = input('Enter put value: ')
    var, future, log_return, atm_vol, log_mean, K = ext()
    simulation_lists(simulations, r, d, var, future, log_return, atm_vol, log_mean, K) #, c_in, p_in)
    print('done')
    plt.show()

    return

if __name__ == "__main__":
    main()
