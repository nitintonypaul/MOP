# Multi-Objective Portfolio Allocation Engine

A Python-based engine designed for advanced portfolio optimization

---

## Project Objective

This project aims to develop a **multi-objective portfolio allocation engine** in Python. It provides a versatile framework for exploring and applying various portfolio optimization techniques, including those for high-performance and computationally intensive financial analysis.

---

## Core Features

* **Multiple Optimization Models:** Implements a suite of industry-standard optimization techniques:
    * **Minimum Variance:** Aims to find the portfolio with the lowest possible risk.
    * **Maximum Diversification (MDP):** Seeks to maximize the portfolio's diversification ratio.
    * **Mean-Variance Optimization (MVO):** The classic Markowitz model to balance risk and return (Black Litterman).
    * **Conditional Value at Risk (CVaR):** Focuses on minimizing losses in the worst-case scenarios (tail risk).
    * **Mean-CVaR:** Maximizes return while minimizing worst-case (tail) losses.
* **Black-Litterman Model Integration:** Fuses market-implied returns with an investor's custom views to produce more stable and intuitive allocations.
* **Robust Risk Modeling:** Uses the Ledoit-Wolf shrinkage estimator to compute a well-conditioned and stable covariance matrix.
* **Performance Backtesting:** Evaluates a "buy-and-hold" portfolio strategy against historical data, reporting key metrics like Sharpe Ratio, Sortino Ratio, and total returns.
* **Portfolio Persistence:** Save your configured portfolios to disk and load them back in later sessions for continued analysis.

---

## Disclaimer

This software is provided for **educational and research purposes only**. It is not intended for live trading or investment decisions. The author is **not liable for any financial losses or damages** incurred from the use of this software. Users should exercise their own due diligence and consult with a financial professional.

---

## Installation & Usage

To get started with the engine:

1.  **Clone the repository (or download the ZIP file):**
    ```bash
    git clone https://github.com/nitintonypaul/MOP.git
    ```
    
2.  **Navigate into the directory:**
    ```bash
    cd MOP
    ```
    
3. **Create a virtual environment (recommended):**
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```  

4. **Install required packages:**
   ```bash
   pip install -r requirements.txt
   ```
   
5.  **Run the demo:**
    ```bash
    python main.py
    ```

The core optimization engine is located in the `lib/` directory. You can integrate it into your own Python projects by copying the `lib/` folder.

---

## Quickstart: A Tutorial

This guide will walk you through the entire process of creating, optimizing, analyzing, and saving a portfolio.

### 1. Initializing a Portfolio

First, import the `Portfolio` class and create an instance. You need to provide a list of stock tickers and the total initial investment amount.

When you create a `Portfolio` object, it automatically:
-   Fetches the last year of daily stock price data from Yahoo Finance.
-   Calculates the asset covariance matrix using the Ledoit-Wolf method.
-   Initializes the portfolio with equal weights for all assets.
    
    ```py
    from portfolio import Portfolio
    import logging

    # Configure basic logging to see the engine's output
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

    # Define your assets and total investment
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    initial_amount = 100000

    # Create a portfolio instance
    p = Portfolio(tickers, initial_amount)
    print("Portfolio created successfully!")

    ```

### 2. Viewing Portfolio Statistics

Use the `.Stats()` method to see the current allocation of assets in your portfolio, including their weights and monetary values.
    
```py
    # Display the initial, equally-weighted portfolio
    print("--- Initial Portfolio Allocation ---")
    print(p.Stats())
```

**Output:**

    --- Initial Portfolio Allocation ---
    STOCK      WEIGHT    AMOUNT
    AAPL         0.25   25000.0
    MSFT         0.25   25000.0
    GOOGL        0.25   25000.0
    AMZN         0.25   25000.0

### 3. Optimizing the Portfolio

The core of the engine is the `.Optimize()` method. You can choose from several allocation strategies by specifying the `method` parameter.

#### Example: Minimum Variance Optimization

This will calculate the asset weights that result in the lowest possible portfolio volatility.

```py
    print("\nOptimizing for Minimum Variance...")
    p.Optimize(method="variance")

    # Display the new, optimized allocation
    print("--- Optimized Portfolio Allocation (Min Variance) ---")
    print(p.Stats())
```

**Output (example values):**

    Optimizing for Minimum Variance...
    --- Optimized Portfolio Allocation (Min Variance) ---
    STOCK      WEIGHT    AMOUNT
    AAPL        0.358   35800.0
    MSFT        0.112   11200.0
    GOOGL       0.481   48100.0
    AMZN        0.049    4900.0

### 4. Analyzing Performance

After optimizing, you can run a simple backtest with the `.Performance()` method. This evaluates how your new (static) weights would have performed over the past year.

The method returns a list of (metric, value) tuples, which can be easily printed or converted to a dictionary.
```py
    print("\n--- Backtest Performance Results ---")
    performance_results = p.Performance()

    # Use tabulate for a clean printout
    from tabulate import tabulate
    print(tabulate(list(performance_results), headers=["METRIC", "VALUE"]))
```

**Output (example values):**

    --- Backtest Performance Results ---
    METRIC                      VALUE
    ------------------------  ---------
    Sharpe                      1.845
    Sortino                     3.123
    Volatility (Annual)         18.551%
    Highest Return (Daily)      2.781%
    Lowest Return (Daily)       -3.411%
    Average Return (Daily)      0.142%
    Total Return (Compounded)   41.876%
    Win Ratio                   58.110%

### 5. Saving and Loading a Portfolio

You can persist your portfolio's state (tickers, weights, and amount) to a file and load it back later. This is useful for saving the results of a time-consuming optimization.

#### Saving the Portfolio
The `Portfolio.Save()` is a classmethod. You pass it the portfolio instance and a filename.

```py
    # Save our optimized portfolio to 'my_tech_portfolio.bin'
    filename = "my_tech_portfolio"
    Portfolio.Save(p, filename)
    print(f"\nPortfolio saved to {filename}.bin")
```

#### Loading the Portfolio
The `Portfolio.Load()` classmethod reconstructs a portfolio from a file. It will re-fetch fresh market data but will apply the saved weights.

```py
    # Imagine this is a new session. Let's load our saved portfolio.
    print("\nLoading portfolio from disk...")
    loaded_p = Portfolio.Load(filename)

    # Verify that the loaded portfolio has the correct, optimized weights
    print("--- Loaded Portfolio Allocation ---")
    print(loaded_p.Stats())
```

This tutorial covers the complete workflow of the Multi-Objective Portfolio Allocation Engine. Feel free to experiment with different tickers, optimization methods, and parameters.

---

## License

This project is open-sourced under the **Apache 2.0 License**.
