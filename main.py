from libs.portfolio import Portfolio

#TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
#AMOUNT = 10000

# Inputs
TICKERS = input("ENTER STOCKS (eg: 'AAPL TSLA MSFT'): ").split(" ")
AMOUNT = float(input("ENTER INVESTMENT AMOUNT: "))

# Declaring portfolio object
investments = Portfolio(tickers=TICKERS, amount=AMOUNT)

# Optimizing Portfolio with various objectives
print("\nPORTFOLIO DATA BEFORE OPTIMIZING")
print(investments.Stats())

investments.Optimize(method="mdp")

print("\nPORTFOLIO DATA AFTER MAXIMUM DIVERSIFICATION")
print(investments.Stats())

investments.Optimize(method="variance")

print("\nPORTFOLIO DATA AFTER VARIANCE OPTIMIZATION")
print(investments.Stats())

Portfolio.Save(investments, "firstInvestment")
print("Portfolio saved to portfolio/firstInvestment.bin")