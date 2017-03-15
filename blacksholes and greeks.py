from scipy.stats import norm
import datetime
import math

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


        # Greeks in the Blacksholes
        # Calculating the partial derivatives for a Black Scholes Option (Call)


"""
Return:
Delta: partial wrt S
Gamma: second partial wrt S
Theta: partial wrt T
Vega: partial wrt v
Rho: partial wrt r
"""


def Black_Scholes_Greeks_Call(S, K, r, v, T, d):
    d1 = (math.log(float(S) / K) + ((r - d) + v * v / 2.) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    T_sqrt = math.sqrt(T)
    Delta_Call = norm.cdf(d1)
    Gamma_Call = norm.pdf(d1) / (S * v * T_sqrt)
    Theta_Call = - (S * v * norm.pdf(d1)) / (2 * T_sqrt) - r * K * math.exp(-r * T) * norm.cdf(d2)
    Vega_Call = S * T_sqrt * norm.pdf(d1)
    Rho_Call = K * T * math.exp(-r * T) * norm.cdf(d2)
    return Delta_Call, Gamma_Call, Theta_Call, Vega_Call, Rho_Call


# Calculating the partial derivatives for a Black Scholes Option (Put)

def Black_Scholes_Greeks_Put(S, K, r, v, T, d):
    d1 = (math.log(float(S) / K) + ((r - d) + v * v / 2.) * T) / (v * math.sqrt(T))
    d2 = d1 - v * math.sqrt(T)
    T_sqrt = math.sqrt(T)
    Delta_Put = -norm.cdf(-d1)
    Gamma_Put = norm.pdf(d1) / (S * v * T_sqrt)
    Theta_Put = -(S * v * norm.pdf(d1)) / (2 * T_sqrt) + r * K * math.exp(-r * T) * norm.cdf(-d2)
    Vega_Put = S * T_sqrt * norm.pdf(d1)
    Rho_Put = -K * T * math.exp(-r * T) * norm.cdf(-d2)
    return Delta_Put, Gamma_Put, Theta_Put, Vega_Put, Rho_Put


def main():
    S = 116.43  # Stock price
    K = [120.78, 130, 111.34, 108.25]  # Strike price
    T1 = (datetime.date(2015, 8, 30) - datetime.date(2015, 6, 1)).days
    T = T1 / 252.0  # Time to maturity
    r = 0.05  # Riskfree interest rate
    d = 0.06  # Dividend yield
    v = 0.35  # Volatility

    for i in range(0,len(K)):
        print (BlackScholes('c', S, K[i], T, r, d, v))
        print (Black_Scholes_Greeks_Call(S, K[i], r, v, T, d))
        print (Black_Scholes_Greeks_Put(S, K[i], r, v, T, d))
    return

if __name__ == "__main__":
    main()
