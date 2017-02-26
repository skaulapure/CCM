'''Initial code for the garch model. I still need to use functions and incorporate a way to solve for gamma, alpha and beta. '''

from pymongo import MongoClient
import pandas as pd
import numpy as np
import math

def sol(alpha,gamma,df):
    beta = 1 - alpha
    def calc(df):

        df.set_index('Date', inplace=True)

        df = df.loc['2009-06-15':'2014-12-31']

        df.log_returns = df.Future.pct_change()

        df.AtM_Var = df['AtM']**2

        df.Norm_AtM_Vol = df.AtM.divide(math.sqrt(252))

        log_mean = df.log_returns.mean()
        norm_AtMVol = df.Norm_AtM_Vol.mean()

        df.stocastic_factor = (df.log_returns - log_mean).divide(df.Norm_AtM_Vol)

        df.simulated_var[1] = df.AtM_Var[1]

        df.simulated_futures[1] = df.Future[1]

        df.simulated_log_returns[1] = df.log_returns[1]

        df.simulated_AtMVol[1] = df.simulated_var[1] ** .5

# values of gamma, alpha, beta taken from prototype
#gamma = 0.0003343
#alpha = 0.0072841
#beta = 0.9927159

        for i in range(2, len(df.index)):
            df.simulated_var[i] = i

            df.simulated_var[i] = gamma + alpha * (df.simulated_log_returns[i-1] - log_mean)**2 + (beta * df.simulated_var[i-1])

            df.simulated_AtMVol[i] = df.simulated_var[i] ** .5

            df.simulated_futures[i] = df.simulated_futures[i-1] * np.exp(log_mean \
                                                        + (df.stocastic_factor[i] * df.simulated_AtMVol[i] / 16))

            df.simulated_log_returns[i] = math.log(df.simulated_futures[i] / df.simulated_futures[i-1])

        df.error = df.log_returns - df.simulated_log_returns

        df.vol_error = df.AtM - df.simulated_AtMVol
        df.abs_error = df.error.abs()

        df.abs_vol_error = df.vol_error.abs()
        print (df['Abs_Error'])
        return
    return

    calc(df)
def main():
    connection = MongoClient()

    db = connection.ccm

    data = list(db.monte_carlo.find({}, {'_id': 0, 'Future': 1, 'AtM': 1, 'Date': 1}).sort('Date'))

    df = pd.DataFrame(data, columns=['Date', 'Future', 'AtM', 'log_returns', 'AtM_Var', 'Norm_AtM_Vol',
                                     'stocastic_factor', 'simulated_var', 'simulated_AtMVol', 'simulated_futures',
                                     'simulated_log_returns', 'error', 'abs_error', 'vol_error', 'abs_vol_error'])


    sol(0.0072841,0.0003343,df)
    return

if __name__ == "__main__":
    main()
