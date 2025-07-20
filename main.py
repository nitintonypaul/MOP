from libs.portfolio import Portfolio

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
AMOUNT = 10000

# Declaring portfolio object
my_portfolio = Portfolio(tickers=TICKERS, amount=AMOUNT)

# Optimizing Portfolio with variance
print("\nPORTFOLIO DATA BEFORE OPTIMIZING")
print(my_portfolio.Stats())

my_portfolio.Optimize(method="variance")

print("\nPORTFOLIO DATA AFTER OPTIMIZING")
print(my_portfolio.Stats())