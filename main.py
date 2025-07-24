from mopEngine.portfolio import Portfolio
import numpy as np
import logging
from tabulate import tabulate

# Logging configurations
# Logging to terminal
logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler()
    ]
)

#TICKERS = ["AAPL", "TSLA", "MSFT", "JPM"]
#AMOUNT = 10000

# BLACK-LITTERMAN VIEW OVERVIEW
# P matrix - Asset outperformance (n views × m assets) (n <= m)
# Q vector - View Vector (n × 1) 
# Omega - Confidence in views(n × n) diagonal matrix

# Example
# Consider a Portfolio of [AAPL, TSLA, MSFT, JPM]
# View 1: TSLA will outperform MSFT by 5%
# View 2: AAPL will have an absolute return of 8%
P = np.array([
    [0,  1, -1,  0],   # TSLA - MSFT = 5%
    [1,  0,  0,  0]    # AAPL absolute = 8%
])
Q = np.array([
    0.05,  # TSLA - MSFT = +5%
    0.08   # AAPL = +8%
])

# Higher values = less confidence
OMEGA = np.diag([0.0025, 0.0025])  # Low uncertainty = strong views

# Inputs
TICKERS = input("ENTER STOCKS (eg: 'AAPL TSLA MSFT'): ").split(" ")
AMOUNT = float(input("ENTER INVESTMENT AMOUNT: "))

# Declaring portfolio object
investments = Portfolio(tickers=TICKERS, amount=AMOUNT)

# Optimizing Portfolio with various objectives
print("\nPORTFOLIO DATA BEFORE OPTIMIZING")
print(investments.Stats())

investments.Optimize(method="cvar")
print("\nPORTFOLIO DATA AFTER CVAR")
print(investments.Stats())

investments.Optimize(method="mdp")
print("\nPORTFOLIO DATA AFTER MDP")
print(investments.Stats())

# MVB (Minimum Viable Backtest)
print("\nPERFORMANCE METRICS")
print(tabulate(list(investments.Performance()), headers=["METRIC", "VALUE"], tablefmt="plain"))