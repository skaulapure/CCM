from datetime import datetime
from random import gauss
from math import exp, sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

def generate_asset_price(S,v,r,T):
    return S * exp((r - 0.5 * v**2) * T + v * sqrt(T) * gauss(0,1.0))


def simulation_array(simulations, duration, S, v, r, T):
    k = np.zeros(shape=(simulations, duration))
    s = S
    for i in range(0, simulations):
        k[i,0] = s
        for j in range(1, duration):
            S=k[i,j-1]
            asset_price = generate_asset_price(S, v, r, T)
            k[i, j] = (asset_price)
    #print (k[3,0])
    return k


def simu_plot(k):
    plt.figure(1)
    sns.kdeplot(k[-1, :])

    plt.figure(2)
    for i in k:
        plt.plot(i)
    return


def main():
    S = 57.30 # underlying price
    v = 0.20 # vol of 20%
    r = 0.0015 # rate of 0.15%c
    simulations = 100
    today = pd.datetime.today()
    end_date = today
    start_date = today - BDay(20)

    duration = (end_date - start_date).days
    T = duration/ 252.0

    k=simulation_array(simulations, duration, S, v, r, T)
    simu_plot(k)
    plt.show()
    return

if __name__ == "__main__":
    main()
