import time
import yfinance as yf
import numpy as np
import logging

# Logging system
# Taking config from root
logger = logging.getLogger(__name__)

# Black Litterman return model
def computeBLreturns(tickers, COVARIANCE, P, Q, OMEGA, lam=2.5,  TAU=0.025):
    
    # Computing market cap
    data = []
    logger.info("FETCHING MARKET CAPITAL")
    for ticker in tickers:
        # Obtaining market cap
        info = yf.Ticker(ticker).info
        data.append(info.get('marketCap', 0))

        # Logging message if market caps are unable to be found
        # Takes weight as 0
        if np.isnan(data[-1]) or data[-1] == 0:
            logger.warning(f"UNABLE TO FETCH MARKET CAP FOR {ticker}")
            print(f"Unable to get market capital for {ticker}")

        # Sleeping for 1 second to prevent throttling yfinance
        time.sleep(1)
    
    # Computing market weights
    # Asset CAP / Total Portfolio CAP
    # Richer the asset, greater the market weight
    marketWeights = np.array(data) / np.array(data).sum()

    # Computing Market Implied returns
    BIG_PIE = lam * (COVARIANCE @ marketWeights)

    # Logging Market Weights
    print("MARKET WEIGHTS:", dict(zip(tickers, marketWeights)))

    # When views are not given
    if (P.any() == False and Q.any() == False and OMEGA.any() == False):
        # Market Implied returns are returned
        logger.info("BLACK-LITTERMAN SUCCESSFUL")
        return BIG_PIE
    
    else:
        # Caching reused matrices
        COV_CONF = np.linalg.inv(TAU * COVARIANCE)
        try:
            OMEGA_INV = np.linalg.inv(OMEGA)
        except np.linalg.LinAlgError:
            logger.error("NON-INVERTIBLE OMEGA")
            raise ValueError("UNCERTAINTY MATRIX (OMEGA) IS NON-INVERTIBLE")

        # Computing two mutliples of Black Litterman Model
        M1 = np.linalg.inv(COV_CONF + (P.T @ OMEGA_INV @ P)) # Posterior Covariance Matrix
        M2 = (COV_CONF) @ BIG_PIE + (P.T @ OMEGA_INV @ Q) # Adjusted return vector
        
        # Flattening to 1D array... if it isnt
        BLreturns = (M1 @ M2).flatten()
        logger.info("BLACK-LITTERMAN SUCCESSFUL")
        return BLreturns