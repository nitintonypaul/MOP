from sklearn.covariance import LedoitWolf
import numpy as np


def getCovariance(data, tickers):
    returnMatrix = []
    for i in range(len(tickers)):
        asset = data[tickers[i]]["Close"].pct_change().dropna().values
        returnMatrix.append(asset)

    LW = LedoitWolf().fit(np.array(returnMatrix).T)

    return LW.covariance_ 
