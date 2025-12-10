from scipy.optimize import minimize
import numpy as np
from mopEngine.blackLitterman import computeBLreturns
import logging

# Logging system
# Taking config from root
logger = logging.getLogger(__name__)

# Variance model
def Variance(w, SIGMA):

    logger.info("VARIANCE OPTIMIZATION INITIATED")

    # Variance Objective
    # minimize {weights.T * COVARIANCE * weights}
    def f(w):
        return w @ SIGMA @ w
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("VARIANCE OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("VARIANCE OPTIMIZATION FAILED")
        raise ValueError("Variance Optimization failed")

# Maximum Diversification Portfolio Model
def MDP(w, SIGMA, sigma):

    logger.info("MDP OPTIMIZATION INITIATED")

    # MDP Objecitve
    # maximize {(weights.T * assetVolatility) / sqrt(VARIANCE)}
    def f(w):
        var = w @ SIGMA @ w
        weightvol = w @ sigma

        return -1 * (weightvol / np.sqrt(var))
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("MDP OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("MDP OPTIMIZATION FAILED")
        raise ValueError("Max Diversification Optimization failed")

# Mean-Variance model
def MVO(w, SIGMA, LAMBDA, tickers, p, q, omega, lambdaBL, TAU):

    logger.info("MVO OPTIMIZATION INITIATED")

    # Computing Black Litterman Returns
    BLret = computeBLreturns(tickers, SIGMA, P=p, Q=q, OMEGA=omega, lam=lambdaBL, TAU=TAU)

    # MVO Objective
    # maximize {weights.T * returns - VARIANCE}
    def f(w):
        return (LAMBDA/2)*(w @ SIGMA @ w) - (w @ BLret)
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("MVO OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("MVO OPTIMIZATION FAILED")
        raise ValueError("Mean-Variance Optimization failed")

# Conditional Value-at-Risk (CVaR) model
def CVaR(w, tickers, ALPHA, data):
    
    logger.info("CVAR OPTIMIZATION INITIATED")

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

    # Robustness check
    if ALPHA > 1 or ALPHA < 0:
        logger.error("INVALID CVAR CONFIDENCE")
        raise ValueError("Confidence is out of bounds (0,1)")
    
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
        logger.info("CVAR OPTIMIZATION SUCCESSFUL")
        return result.x[:-1]
    else:
        logger.error("CVAR OPTIMIZATION FAILED")
        raise ValueError("CVaR Optimization failed")

# Mean Conditional Value-at-Risk (MCVaR) model
def MCVaR(w, tickers, ALPHA, data):
    
    logger.info("MCVAR OPTIMIZATION INITIATED")

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
    
    # Robustness check
    if ALPHA > 1 or ALPHA < 0:
        logger.error("INVALID CVAR CONFIDENCE")
        raise ValueError("Confidence is out of bounds (0,1)")
    
    V = 1
    x0 = np.append(w, V)

    returns_list = []
    for i in tickers:
        stock_returns = data[i]["Close"].pct_change(fill_method=None).dropna().values
        returns_list.append(stock_returns)

    # Failsafe
    minLen = min(len(x) for x in returns_list)
    trimmed_returns = np.array([r[-minLen:] for r in returns_list]).T

    result = minimize(f, x0, method='SLSQP', bounds=[(0,1)]*len(w) + [(None), (None)], constraints= [{'type':'eq','fun': lambda x: x[:-1].sum()-1}])
    if result.success:
        logger.info("MCVAR OPTIMIZATION SUCCESSFUL")
        return result.x[:-1]
    else:
        logger.error("MCVAR OPTIMIZATION FAILED")
        raise ValueError("Mean-CVaR Optimization failed")

# Kelly Criterion Model
def Kelly(w, fr, tickers, data):
    logger.info("KELLY OPTIMIZATION INITIATED")

    # Kelly Objective 
    # Minimize E[log(1+wTr)]
    def f(w):

        N = trimmed_returns.shape[0]
        SUM = 0

        # Expectation robust against outliers
        for i in range(N):
            sample = trimmed_returns[i]
            portfolio_returns = max(np.dot(w, sample), -0.99)
            SUM += np.log(1+ fr * portfolio_returns)

        return -SUM / N

    # Robustness check
    if fr > 1 or fr <= 0:
        logger.error("INVALID KELLY FRACTION")
        raise ValueError("Fraction is out of bounds (0,1]")
    
    returns_list = []
    for i in tickers:
        returns_list.append(data[i]["Close"].pct_change(fill_method=None).dropna().values)

    # Failsafe
    minLen = min(len(x) for x in returns_list)
    trimmed_returns = np.array([r[-minLen:] for r in returns_list]).T

    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        logger.info("KELLY OPTIMIZATION SUCCESSFUL")
        return result.x
    else:
        logger.error("KELLY OPTIMIZATION FAILED")
        raise ValueError("Kelly Optimization failed")