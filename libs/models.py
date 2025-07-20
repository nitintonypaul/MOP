from scipy.optimize import minimize

def Variance(w,cov):
    def f(w):
        return w @ cov @ w
    
    result = minimize(f, w, method='SLSQP', bounds=[(0,1)]*len(w), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    if result.success:
        return result.x
    else:
        raise ValueError("Variance Optimization failed")