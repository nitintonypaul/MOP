# Modules
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from tabulate import tabulate
import libs.models as models
import pickle
import os

# Portfolio Class
class Portfolio:

    # Caching essential values
    def __init__(self, tickers, amount):
        self.tickers = tickers
        self.amount = amount
        self.weights = np.ones(len(tickers)) / len(tickers)
        self.data = self.Fetch()
        self.covar = self.Covariance()

    # Saving Portfolio to binary
    @classmethod
    def Save(cls, portfolio_instance, name):
        portfolio_data = {"tickers":portfolio_instance.tickers, "weights":portfolio_instance.weights, "amount":portfolio_instance.amount}
        os.makedirs('portfolios', exist_ok=True)

        with open(f"portfolios/{name}.bin", 'wb') as file:
            pickle.dump(portfolio_data, file)
    
    # Loading portfolio from binary
    @classmethod
    def Load(cls, name):
        with open(f"portfolios/{name}.bin", 'rb') as file:
            portfolio_data = pickle.load(file)

        tickers = portfolio_data["tickers"]
        weights = portfolio_data["weights"]
        amount = portfolio_data["amount"]

        loadedPortfolio = cls(tickers, amount)
        loadedPortfolio.weights = weights

        return loadedPortfolio
    
    # Fetching values
    def Fetch(self, period=100):
        print("FETCHING DATA...")
        return yf.download(tickers=self.tickers, period=f"{period}d", group_by="ticker", auto_adjust=True)

    # Computing covariance
    def Covariance(self):
        returnMatrix = []
        for ticker in self.tickers:
            asset = self.data[ticker]["Close"].pct_change().dropna().values
            returnMatrix.append(asset)

        LW = LedoitWolf().fit(np.array(returnMatrix).T)
        return LW.covariance_ 

    # Computing volatility per asset
    def Volatility(self):
        volvector = []
        for ticker in self.tickers:
            asset = self.data[ticker]["Close"]
            logreturns = np.log(asset / asset.shift(1)).dropna()
            volvector.append(round(logreturns.std(), 5))
        
        return np.array(volvector)
    
    # Returning stats of the portfolio as a table
    def Stats(self):
        table = []
        for ticker, weight in zip(self.tickers, self.weights):
            table.append([ticker, round(weight, 3), round(weight*self.amount, 2)])
        
        return tabulate(table, headers=["STOCK", "WEIGHT", "AMOUNT"], tablefmt='plain')

    # Optimize function to optimize using a valid optimizer
    # Optimizers to date: VARIANCE, MDP
    def Optimize(self, method="variance", risk=1, time=1, p=np.array([0, 0, 0]), q=np.array([0, 0, 0]), omega=np.array([0, 0, 0]), confidence=None):
        
        # Resetting weights to prevent false convergence
        tempweights = np.ones(len(self.tickers)) / len(self.tickers)

        optimizers = {
            "variance":[models.Variance, [tempweights, self.covar*time]],
            "mdp":[models.MDP, [tempweights, self.covar*time, self.Volatility()*np.sqrt(time)]],
            "mean-variance":[models.MVO, [tempweights, self.covar*time, risk, self.tickers, p, q, omega]]
        }

        if method.lower() in optimizers:
            function, args = optimizers[method.lower()]
            self.weights = function(*args)
        
        else:
            raise ValueError("Invalid optimizer method")