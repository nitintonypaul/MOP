# Multi-Objective Portfolio Allocation Engine

A Python-based engine designed for advanced portfolio optimization

---

## Project Objective

This project aims to develop a **multi-objective portfolio allocation engine** in Python. It provides a versatile framework for exploring and applying various portfolio optimization techniques, including those for high-performance and computationally intensive financial analysis.

---

## Available Optimizers

* **Variance Optimization:** Focuses on minimizing portfolio variance for a given set of returns.
* **Maximum Diversification Optimization (MDP):** Implements risk diversification strategies beyond simple variance reduction.

---

## Upcoming Enhancements

* **Advanced Models:** Integration of Black-Litterman with Mean-Variance Optimization (MVO), Conditional Value-at-Risk (CVaR/mean-CVaR), and Entropic Value-at-Risk (EVaR/mean-EVaR).
* **Robust & Tail-Risk Optimization:** Development of robust optimization techniques and specialized tail-risk models.
* **Dynamic Portfolio Management:** Features for refreshing portfolios and efficient data persistence (save/load).

---

## Disclaimer

This software is provided for **educational and research purposes only**. It is not intended for live trading or investment decisions. The author is **not liable for any financial losses or damages** incurred from the use of this software. Users should exercise their own due diligence and consult with a financial professional.

---

## Installation & Usage

To get started with the engine:

1.  **Clone the repository (or download the ZIP file):**
    ```bash
    git clone https://github.com/nitintonypaul/MOP
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

## License

This project is open-sourced under the **Apache 2.0 License**.
