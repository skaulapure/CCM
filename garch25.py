import math
import sys
import pandas as pd
import numpy as np
from pymongo import MongoClient
from scipy import optimize


class GarchSimulation(object):
    def __init__(self):
        super(GarchSimulation, self).__init__()
        self.data = {}
        self.log_mean = None
        self.vol_std = None
        self.vol_mean = None
        self.omega = 0
        self.gamma = 0
        self.alpha_1 = 0
        self.alpha_2 = 0
        self.beta_1 = 0
        self.beta_2 = 0
        self.beta_3 = 0
        self.beta_4 = 0
        self.beta_5 = 0
        self.stochastic_factors = []
        self.vv_noise = []
        self.fore_vv_noise = []

    def data_call(self, start_date, end_date):
        # Call data do the calculation
        connection = MongoClient()
        db = connection.ccm
        data = list(db.monte_carlo.find({}, {'_id': 0, 'Future': 1, 'AtM': 1, 'Date': 1}).sort('Date'))
        df_columns = ['Date', 'Future', 'AtM', 'log_returns', 'AtM_Var', 'Norm_AtM_Vol', 'Norm_Var', 'Vol_Change',
                      'VV_Noise']
        df = pd.DataFrame(data, columns=df_columns)
        df.set_index('Date', inplace=True)
        df = df.loc[start_date:end_date]
        df.log_returns = np.log(df.Future / df.Future.shift())
        df.AtM_Var = df.AtM * df.AtM
        df.Norm_AtM_Vol = df.AtM / 16.0
        self.log_mean = log_mean = df.log_returns.mean()
        df.stochastic_factor = stochastic_factor = (df.log_returns - log_mean) / df.Norm_AtM_Vol
        df.Norm_Var = df.AtM_Var / 256.0
        df.Vol_Change = np.log(df.AtM / df.AtM.shift())
        self.vol_mean = vol_change_mean = df.Vol_Change.mean()
        self.vol_std = vol_change_std = df.Vol_Change.std()
        df.VV_Noise = vv_noise = (df.Vol_Change - vol_change_mean) / vol_change_std
        self.data = df
        self.vv_noise = vv_noise
        self.stochastic_factors = stochastic_factor
        return df

    def noise_calc(self, para):
        omega = para[0]
        gamma = para[1]
        alpha_1 = para[2]
        alpha_2 = para[3]
        betas = para[4:]

        vv_noise = []
        vv_noise[0:6] = self.vv_noise[0:6]

        stochastic_factor = self.stochastic_factors

        for i in range(6, len(stochastic_factor)):
            if stochastic_factor[i-1] > 0:
                stoch_mul = 1
            else:
                stoch_mul = 0

            v_mul = vv_noise[i-5: i]

            vv_noise.append(omega + (alpha_1 + gamma * stoch_mul) * stochastic_factor[i-1] +
                            alpha_2 * stochastic_factor[i-2] + np.dot(betas, v_mul))
        self.fore_vv_noise = vv_noise
        return vv_noise

    def forecast(self, vv_noise):
        fore_vol_change = [(i * self.vol_std + self.vol_mean) for i in vv_noise]

        vol = self.data['Norm_AtM_Vol'][1]

        fore_norm_vol = [vol, ]

        for v in fore_vol_change[2:]:
            try:
                vol *= math.exp(v)

            except OverflowError:
                print(np.sum(fore_vol_change[2:]))
                vol *= math.exp(math.log(sys.float_info.max))

            fore_norm_vol.append(vol)

        fore_norm_var = [v * v for v in fore_norm_vol]
        fore_atm = [v * 16 for v in fore_norm_vol]
        #fore_var = [var * var for var in fore_norm_var]
        fore_log_returns = [self.log_mean + nv * sf for nv,sf in zip(fore_norm_vol, self.stochastic_factors[1:])]

        future = self.data['Future'][0]
        fore_futures = [future, ]

        for i in fore_log_returns:
            try:
                future *= math.exp(i)
            except OverflowError:
                print(np.sum(fore_log_returns))
                future *= math.exp(math.log(sys.float_info.max))
            fore_futures.append(future)

        return fore_atm

    def error(self, guess):
        atm_old = self.data['AtM'][1:]
        vv_noise = self.noise_calc(guess)
        atm_new = self.forecast(vv_noise)
        abs_vol_error = [abs(o - n) for o, n in zip(atm_old, atm_new)]
        sum_abs_vol_error = np.sum(abs_vol_error)
        return sum_abs_vol_error

    def optimization(self):
        initial_guess = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

        print('starting optimization')
        result = optimize.minimize(self.error, initial_guess)
        print('optimization done')

        self.omega, self.gamma, self.alpha_1, self.alpha_2 = result.x[0:4]

        self.beta_1, self.beta_2, self.beta_3, self.beta_4, self.beta_5 = result.x[4:]

        return result.x

garch = GarchSimulation()
garch.data_call('2009-06-15','2011-06-02')
#garch.error([ 0.03851945, -0.1591309 ,  0.56942879,  0.36986185,  0.12920284,
 #             0.21863221,  0.08846381, -0.25667422, -0.87225564])
print(garch.optimization())
