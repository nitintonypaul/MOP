from scipy.optimize import minimize
import numpy as np

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