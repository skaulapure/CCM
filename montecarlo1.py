from datetime import datetime
from random import gauss
from math import exp, sqrt
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay


def generate_asset_price(s, v, r, t):
    return s * exp((r - 0.5 * v ** 2) * t + v * sqrt(t) * gauss(0, 1.0))


def simulation_array(simulations, duration, s1, v, r, t):
    k = np.zeros(shape=(simulations, duration))
    s_f = s1
    for i in range(0, simulations):
        k[i, 0] = s_f
        for j in range(1, duration):
            s = k[i, j - 1]
            k[i, j] = generate_asset_price(s, v, r, t)  # asset_price
    # print (k[3,0])
    return k


def simu_plot(k):
    plt.figure(1)
    sns.kdeplot(k[-1, :])

    plt.figure(2)
    for i in k:
        plt.plot(i)
    return


def main():
    s = 57.30  # underlying price
    v = 0.20  # vol of 20%
    r = 0.0015  # rate of 0.15%c
    simulations = 10000
    today = pd.datetime.today()
    end_date = today
    start_date = today - BDay(20)

    duration = (end_date - start_date).days
    t = duration / 252.0

    k = simulation_array(simulations, duration, s, v, r, t)
    simu_plot(k)
    plt.show()
    return


if __name__ == "__main__":
    main()
