import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from tabulate import tabulate
import libs.models as models
import pickle
import os

# Portfolio Object
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
    def Optimize(self, method="variance", p=None, q=None, omega=None, confidence=None):
        
        # Resetting weights to prevent false convergence
        tempweights = np.ones(len(self.tickers)) / len(self.tickers)

        optimizers = {
            "variance":[models.Variance, [tempweights, self.covar]],
            "mdp":[models.MDP, [tempweights, self.covar, self.Volatility()]]
        }

        if method.lower() in optimizers:
            function, args = optimizers[method.lower()]
            self.weights = function(*args)
        
        else:
            raise ValueError("Invalid optimizer method")