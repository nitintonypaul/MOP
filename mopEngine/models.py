from scipy.optimize import minimize
import numpy as np
from mopEngine.blackLitterman import computeBLreturns

# Variance model
def Variance(w, SIGMA):

    # Variance Objective
    # minimize {weights.T * COVARIANCE * weights}
    def f(w):
        return w @ SIGMA @ w
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        return result.x
    else:
        raise ValueError("Variance Optimization failed")

# Maximum Diversification Portfolio Model
def MDP(w, SIGMA, sigma):

    # MDP Objecitve
    # maximize {(weights.T * assetVolatility) / sqrt(VARIANCE)}
    def f(w):
        var = w @ SIGMA @ w
        weightvol = w @ sigma

        return -1 * (weightvol / np.sqrt(var))
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        return result.x
    else:
        raise ValueError("Max Diversification Optimization failed")

# Mean-Variance model
def MVO(w, SIGMA, LAMBDA, tickers, p, q, omega, lambdaBL, TAU):

    # Computing Black Litterman Returns
    BLret = computeBLreturns(tickers, SIGMA, P=p, Q=q, OMEGA=omega, lam=lambdaBL, TAU=TAU)

    # MVO Objective
    # maximize {weights.T * returns - VARIANCE}
    def f(w):
        return (LAMBDA/2)*(w @ SIGMA @ w) - (w @ BLret)
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        return result.x
    else:
        raise ValueError("Mean-Variance Optimization failed")

# Conditional Value-at-Risk (CVaR) model
def CVaR(w, tickers, ALPHA, data):
    
    # CVaR Objective 
    # Minimize {v + 1/(1-CONFIDENCE)N * SUM{N} (max(-wR-v, 0))}
    def f(x):

        w = x[:-1]
        v = x[-1]

        N = trimmed_returns.shape[0]
        SUM = 0

        for i in range(N):
            sample = trimmed_returns[i]
            var_excess = max(-np.dot(w, sample) - v, 0)
            SUM += var_excess

        return v + (SUM / ((1 - ALPHA) * N))

    V = 1
    x0 = np.append(w, V)

    returns_list = []
    for i in tickers:
        stock_returns = data[i]["Close"].pct_change(fill_method=None).dropna().values
        returns_list.append(stock_returns)

    # Failsafe
    minLen = min(len(x) for x in returns_list)
    trimmed_returns = np.array([r[-minLen:] for r in returns_list]).T

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(None,None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        return result.x[:-1]
    else:
        raise ValueError("CVaR Optimization failed")

# Mean Conditional Value-at-Risk (MCVaR) model
def MCVaR(w, tickers, ALPHA, data):
    
    # MCVaR Objective 
    # Maximize {weights.T * returns - CVaR}
    def f(x):

        w = x[:-1]
        v = x[-1]

        N = trimmed_returns.shape[0]
        SUM = 0
        MEAN_TERM = 0

        for i in range(N):
            sample = trimmed_returns[i]
            meanT = np.dot(w, sample)
            MEAN_TERM += meanT
            var_excess = max((-1*meanT) - v, 0)
            SUM += var_excess

        cvar = v + (SUM / ((1 - ALPHA) * N))

        return cvar - (MEAN_TERM/N)

    V = 1
    x0 = np.append(w, V)

    returns_list = []
    for i in tickers:
        stock_returns = data[i]["Close"].pct_change(fill_method=None).dropna().values
        returns_list.append(stock_returns)

    # Failsafe
    minLen = min(len(x) for x in returns_list)
    trimmed_returns = np.array([r[-minLen:] for r in returns_list]).T

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(None,None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        return result.x[:-1]
    else:
        raise ValueError("Mean-CVaR Optimization failed")