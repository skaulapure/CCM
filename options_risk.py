from scipy.stats import norm
#import matplotlib.mlab as mlab
from decimal import *
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import profile
import seaborn as sns
from pymongo import MongoClient
from scipy import optimize


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

class GARCH(object):
    def __init__(self, gamma, alpha):
        super(GARCH, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = 1 - alpha

    def step(self, log_return, previous_var):
        return self.gamma + self.alpha*log_return**2 + self.beta*previous_var


class SimulationEngine(object):
    def __init__(self, iteration):
        super(SimulationEngine, self).__init__()
        self.iteration = iteration
        self.simulation_result = []
        self.data = {}
        self.log_mean = None
        self.gamma = 0
        self.alpha = 0
        self.beta = None
        self.stochastic_factors = []

    def data_call(self, start_date, end_date):
        # Call data do the calculation
        connection = MongoClient()
        db = connection.ccm
        data = list(db.monte_carlo.find({}, {'_id': 0, 'Future': 1, 'AtM': 1, 'Date': 1}).sort('Date'))
        df_columns = ['Date', 'Future', 'AtM', 'log_returns', 'AtM_Var']
        df = pd.DataFrame(data, columns=df_columns)
        df.set_index('Date', inplace=True)
        df = df.loc[start_date:end_date]
        df.log_returns = np.log(df.Future / df.Future.shift())
        df.AtM_Var = df.AtM * df.AtM
        df.Norm_AtM_Vol = df.AtM / 16
        self.log_mean = log_mean = df.log_returns.mean()
        df.stochastic_factor = stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol
        self.data = df
        self.stochastic_factors = stochastic_factor

        #self.data = {}
        #self.log_mean = None

    def optimization(self):
        initial_guess = np.array([0, 0])

        cons = [{'type': 'ineq', 'fun': lambda x: x - 0},
                {'type': 'ineq', 'fun': lambda x: x - 0}]

        result = optimize.minimize(self.error_finding, initial_guess,
                                   args=(self.data['log_returns']), constraints=cons)
        self.gamma, self.alpha = result.x
        self.beta = 1 - self.alpha
        return

    @staticmethod
    def calc(guess, log_mean, stochastic_factors, future, atm_vol, log_return, var):
        gamma = guess[0]
        alpha = guess[1]

        result = \
            {
                'Date': [],
                'future': [future],
                'atm_vol': [atm_vol],
                'call': [],
                'put': [],
                'straddle': []
            }

        simulated_log_return = [log_return, ]

        garch = GARCH(gamma, alpha)

        for stochastic in stochastic_factors:
            var = garch.step(log_return, var)
            atm_vol = np.sqrt(var)
            if np.isnan(atm_vol):
                return np.inf
            log_return = log_mean + stochastic * atm_vol / 16
            future = future * np.exp(log_return)

            simulated_log_return.append(log_return)

            result['future'].append(future)
            result['atm_vol'].append(atm_vol)
            #result['call'].append(call)
            #result['put'].append(put)
            #result['straddle'].append(call + put)

        return result, simulated_log_return

    def error_finding(self, guess, log_returns):
        r, simulated_returns = self.calc(guess, self.log_mean, self.stochastic_factors,
                                         self.data['Future'][1], self.data['AtM'][1], log_returns, self.data['AtM_Var'][1])
        error = np.array([i - j for i, j in zip(log_returns[1:], simulated_returns)])
        abs_error = abs(error)
        mean_abs_error = np.mean(abs_error)
        return mean_abs_error

    def fin(self):
        for i in range(self.iteration):
            stoc_factors = [norm.ppf(random.random()) for i in range(100 - 1)]
            guess = [self.gamma, self.alpha]
            r, s = self.calc(guess, self.log_mean, stoc_factors, self.data['Future'][-1],
                             self.data['AtM'][-1], self.data['log_return'][-1], self.data['AtM_Var'][-1])
            self.simulation_result.append(r)

    @staticmethod
    def calcualte_risk(instruments, simulation):
        output = \
            {
                'Future': [],
                'C@1000': [],
                'P@900': [],
                'Straddle': [],
                'Total': [],
            }

        for i, fut, vol in enumerate(zip(simulation['future'], simulation['atm_vol'])):
            position_value = 0
            position_delta = 0
            for instrument in instruments:
                qty = instrument['Qty']
                if instrument['Type'] == 'F':
                    # Do something
                    position_value += fut * qty
                    position_delta += instrument['Qty']
                    output['Future'].append(fut * qty)

                if instrument['Type'] == 'C':
                    if not instrument.get('VolMulitplier'):
                        instrument['VolMultiplier'] = instrument['Vol'] / vol
                    if instrument['DtE'] - i < 0:
                        pass
                    call_value = BlackScholes('C', fut, instrument['Strike'], (instrument['DtE'] - i) / 252.0,
                                              0.0075, 0, vol * instrument['VolMultiplier'])
                    position_value += call_value * qty
                    output['C@1000'].append(call_value * qty)

                if instrument['Type'] == 'P':
                    if not instrument.get('VolMulitplier'):
                        instrument['VolMultiplier'] = instrument['Vol'] / vol
                    if instrument['DtE'] - i < 0:
                        pass
                    put_value = BlackScholes('P', fut, instrument['Strike'], (instrument['DtE'] - i) / 252.0,
                                             0.0075, 0, vol * instrument['VolMultiplier'])
                    position_value += put_value * qty
                    output['P@900'].append(put_value * qty)

                if instrument['Type'] == 'S':
                    pass

                output['Total'].append(position_value)
        return output

    def call_risk(self):
        instruments = \
            [
                dict(Name='Future', Type='F', Qty=3),
                dict(Name='C@1000', Type='C', Strike=1000, Vol=0.4, DtE=21, Qty=1),
                dict(Name='P@900', Type='P', Strike=900, Vol=0.33, DtE=22, Qty=-2),
            ]
        self.calcualte_risk(instruments, self.simulation_result[0])

    def show_result(self):
        for i in self.simulation_result:
            plt.plot(i['future'])
            plt.plot(i['atm_vol'])
            plt.plot(i['call'])
            plt.plot(i['put'])
            plt.plot(i['straddle'])
        # plotting


def BlackScholes(CallPutFlag, S, K, T, r, d, v):
    d1 = (math.log(float(S) / K) + ((r - d) + v * v / 2.) * float(T)) / (v * math.sqrt(float(T)))
    d2 = d1 - v * math.sqrt(T)
    if CallPutFlag == 'c':
        return S * math.exp(-d * T) * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
    else:
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * math.exp(-d * T) * norm.cdf(-d1)


if __name__ == "__main__":
    mod = SimulationEngine(5)
    mod.data_call('2009-06-15', '2014-12-31')
    mod.optimization()
    plt.show()
