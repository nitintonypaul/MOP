# Modules
import numpy as np
import yfinance as yf
from sklearn.covariance import LedoitWolf
from tabulate import tabulate
import mopEngine.models as models
import pickle
import os
import logging

# Logging system
# Taking config from root
logger = logging.getLogger(__name__)

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
        self.history = self.Fetch()
        self.data = self.history.iloc[-100:]
        self.covar = self.Covariance()

    # Saving Portfolio to binary
    @classmethod
    def Save(cls, portfolio_instance, name):

        logger.info(f"Saving portfolio to {name}.bin")

        portfolio_data = {"tickers":portfolio_instance.tickers, "weights":portfolio_instance.weights, "amount":portfolio_instance.amount}
        os.makedirs('portfolios', exist_ok=True)

        with open(f"portfolios/{name}.bin", 'wb') as file:
            pickle.dump(portfolio_data, file)
    
    # Loading portfolio from binary
    @classmethod
    def Load(cls, name):

        logger.info(f"Loading portfolio from {name}.bin")

        with open(f"portfolios/{name}.bin", 'rb') as file:
            portfolio_data = pickle.load(file)

        tickers = portfolio_data["tickers"]
        weights = portfolio_data["weights"]
        amount = portfolio_data["amount"]

        loadedPortfolio = cls(tickers, amount)
        loadedPortfolio.weights = weights

        return loadedPortfolio
    
    # Fetching values
    def Fetch(self, period=465):

        logger.info("FETCHING DATA")

        # If data is empty or some other error
        try:
            data = yf.download(tickers=self.tickers, period=f"{period}d", group_by="ticker", auto_adjust=True)
            
            if data.empty:
                logger.warning("FETCH FAILED")
                raise self.PortfolioError("No data returned from yfinance")

            logger.info("FETCH SUCCESSFUL")
            return data
        
        except Exception as e:
            logger.exception("FETCH FAILED")
            raise self.PortfolioError(f"Failed to fetch data due to an underlying error: {e}")

    # Computing covariance
    def Covariance(self):
        
        logger.info("COMPUTING COVARIANCE")

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
    # Optimizers to date: Variance, MDP, MVO, CVaR, Mean-CVaR
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

        tickers_length = len(self.tickers)

        # If A single Black-Litterman matrix is skipped, the entire view framework is defaulted
        # Black-Litterman will be calculated without views. i.e. Market Implied Returns
        if p is None or q is None or omega is None:
            p, q, omega = np.zeros(tickers_length), np.zeros(tickers_length), np.zeros(tickers_length)
        
        # Resetting weights to prevent false convergence
        tempweights = np.ones(tickers_length) / tickers_length

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
            logger.error("INVALID OPTIMIZER")
            raise self.PortfolioError(f"Invalid Optimizer method: {method}")

    # A minimum Viable Backtest using stock history over 1 year   
    def Performance(self):
        
        logger.info("INITIATING BACKTEST (0 REBALANCES)")

        # Computing global returns of each asset from portfolio history
        globalReturns = []
        for ticker in self.tickers:
            # Data isolation from Covariance matrix data
            # i.e. first 365 values
            stock_history = self.history[ticker]["Close"].pct_change(fill_method=None).dropna().values[:365]
            globalReturns.append(stock_history)
        globalReturns = np.array(globalReturns)

        # Computing portfolio returns (weight * each element in global returns)
        portfolioReturns = []
        for returns in globalReturns.T:
            RETURN = self.weights @ returns
            portfolioReturns.append(RETURN)
        
        # Caching repeated or risky values
        portfolioReturns = np.array(portfolioReturns)
        portfolioAvg = portfolioReturns.mean()
        downsideSTD = portfolioReturns[portfolioReturns < 0].std()
        
        # Metrics to be displayed
        SHARPE = portfolioAvg / portfolioReturns.std()
        SORTINO = portfolioAvg / downsideSTD if downsideSTD > 0 else np.nan
        VOLATILITY = portfolioReturns.std() * np.sqrt(252)
        MAX_RETURNS = max(portfolioReturns)
        MIN_RETURNS = min(portfolioReturns)
        AVERAGE = portfolioAvg
        TOTAL = np.prod(1 + portfolioReturns) - 1
        HIT = len(portfolioReturns[portfolioReturns > 0]) / len(portfolioReturns)

        # Metrics and Corresponding values
        metrics = [
            'Sharpe',
            'Sortino',
            'Volatility (Annual)',
            'Highest Return (Daily)',
            'Lowest Return (Daily)',
            'Average Return (Daily)',
            'Total Return (Compounded)',
            'Win Ratio'
        ]
        values = [VOLATILITY, MAX_RETURNS, MIN_RETURNS, AVERAGE, TOTAL, HIT]
        values = [round(SHARPE, 3), round(SORTINO, 3)] + [f"{round(x*100, 3)}%" for x in values]

        logger.info("BACKTEST SUCCESSFUL")

        # Returning in table format
        return (zip(metrics, values))