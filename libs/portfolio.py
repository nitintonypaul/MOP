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

    # PortfolioError
    # A nice touch
    class PortfolioError(Exception):
        pass

    # Caching essential values
    def __init__(self, tickers, amount):
        self.tickers = tickers
        self.amount = amount
        self.weights = np.ones(len(tickers)) / len(tickers)
        self.data = self.Fetch()
        self.covar = self.Covariance()
        self.history = self.Fetch(period=365)

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
            asset = self.data[ticker]["Close"].pct_change(fill_method=None).dropna().values
            returnMatrix.append(asset)

        min_len = min(len(r) for r in returnMatrix)
        trimmed = np.array([r[-min_len:] for r in returnMatrix]).T
        
        LW = LedoitWolf().fit(trimmed)
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
    def Optimize(
            self, 
            method="variance", 
            risk=0.2, 
            time=1, 
            p=None, 
            q=None, 
            omega=None, 
            confidence=0.9,
            lambdaBL=2.5,
            tauBL=0.025
        ):

        # If A single Black-Litterman matrix is skipped, the entire view framework is defaulted
        # Black-Litterman will be calculated without views. i.e. Market Implied Returns
        if p is None or q is None or omega is None:
            p, q, omega = np.zeros(3), np.zeros(3), np.zeros(3)
        
        # Resetting weights to prevent false convergence
        tempweights = np.ones(len(self.tickers)) / len(self.tickers)

        # Available Optimizers dictionary
        optimizers = {
            "variance":[models.Variance, [tempweights, self.covar*time]],
            "mdp":[models.MDP, [tempweights, self.covar*time, self.Volatility()*np.sqrt(time)]],
            "mean-variance":[models.MVO, [tempweights, self.covar*time, risk, self.tickers, p, q, omega, lambdaBL, tauBL]],
            "cvar":[models.CVaR, [tempweights, self.tickers, confidence, self.history]],
            "mean-cvar":[models.MCVaR, [tempweights, self.tickers, confidence, self.history]]
        }

        # Checking if optimizer is valid
        if method.lower() in optimizers:
            function, args = optimizers[method.lower()]
            self.weights = function(*args)
        
        else:
            raise self.PortfolioError(f"Invalid Optimizer method: {method}")
    