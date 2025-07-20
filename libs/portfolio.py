import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from tabulate import tabulate
import libs.models as models

class Portfolio:
    def __init__(self, tickers, amount):
        self.tickers = tickers
        self.data = self.Fetch()
        self.amount = amount
        self.weights = np.ones(len(self.tickers)) / len(self.tickers)
        self.covar = self.Covariance()

    def Fetch(self, period=100):
        return yf.download(tickers=self.tickers, period=f"{period}d", group_by="ticker", auto_adjust=True)

    def Covariance(self):
        returnMatrix = []
        for ticker in self.tickers:
            asset = self.data[ticker]["Close"].pct_change().dropna().values
            returnMatrix.append(asset)

        LW = LedoitWolf().fit(np.array(returnMatrix).T)
        return LW.covariance_ 

    def Volatility(self):
        volvector = []
        for ticker in self.tickers:
            asset = self.data[ticker]["Close"]
            logreturns = np.log(asset / asset.shift(1)).dropna()
            volvector.append(round(logreturns.std(), 5))
        
        return volvector
    
    def Stats(self):
        table = []
        for ticker, weight in zip(self.tickers, self.weights):
            table.append([ticker, round(weight, 3), round(weight*self.amount, 2)])
        
        return tabulate(table, headers=["STOCK", "WEIGHT", "AMOUNT"], tablefmt='plain')
    
    def Optimize(self, method="variance"):
        
        optimizers = {
            "variance":[models.Variance, [self.weights, self.covar]],
            "mdp":[models.Variance, [self.weights, self.covar, self.Volatility()]]
        }

        if method.lower() in optimizers:
            function, args = optimizers[method.lower()]
            self.weights = function(*args)
        
        else:
            raise ValueError("Invalid optimizer method")