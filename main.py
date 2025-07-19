import yfinance as yf
import numpy as np
from scipy.optimize import minimize

from utils import getCovariance
import models

tickers = [
    "PG",
    "TSLA",
    "GME",
    "NVDA",
    "AMD",
    "LCID"
]

model = "variance"

# Assigning initial weights
weights = np.ones(len(tickers)) / len(tickers)

# Fetching data
data = yf.download(tickers=tickers, period=f"{len(tickers)*10}d", interval="1d", group_by="ticker", auto_adjust=True)

# Obtaining covariance after Ledoit Wolf Shrinkage
covariance = getCovariance(data, tickers)

if model.lower() == "variance":
    result = minimize(models.Variance, weights, args=(covariance,), method='SLSQP', bounds=[(0,1)]*len(weights), constraints= [{'type':'eq','fun': lambda x: x.sum()-1}])
    optimized = result.x

print("Success:", result.success)
print("Message:", result.message)
print("Fun (objective value):", result.fun)
print("Resulting weights:", result.x)

print(optimized)