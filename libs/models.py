from scipy.optimize import minimize
import numpy as np
from libs.blackLitterman import computeBLreturns

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
def MVO(w, SIGMA, LAMBDA, tickers, p, q, omega):

    # Computing Black Litterman Returns
    BLret = computeBLreturns(tickers, SIGMA, P=p, Q=q, OMEGA=omega)
    print(BLret)

    # MVO Objective
    # maximize {weights.T * returns - VARIANCE}
    def f(w):
        return (LAMBDA/2)*(w @ SIGMA @ w) - (w @ BLret)
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        return result.x
    else:
        raise ValueError("Mean-Variance Optimization failed")