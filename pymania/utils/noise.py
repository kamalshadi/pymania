from .regression import lslinear, RegressionError
import numpy as np

eps = 0.0001

MIN_REGRESSION_POINTS = 5

class NoiseError(Exception):
    pass

def sweep_threshold(data):
    l = len(data)
    if l < 3*MIN_REGRESSION_POINTS:
        raise NoiseError('Insufficient data for noise threshold estimation')
    n0 = data[MIN_REGRESSION_POINTS,1]
    n1 = data[l-MIN_REGRESSION_POINTS,1]
    thresholds = np.linspace(n0+eps,n1-eps,100)
    eta = np.zeros((100,1))
    his = -float('inf')
    for i,t in enumerate(thresholds):
        chunk1 = data[data[:,1]>t]
        chunk2 = data[data[:,1]<=t]
        if len(chunk1)<MIN_REGRESSION_POINTS or len(chunk2)<MIN_REGRESSION_POINTS:
            eta[i] = 0
            continue
        eta[i] = r2_fraction(chunk1,chunk2)
    return (thresholds,eta)

def find_threshold(data):
    thresholds,eta = sweep_threshold(data)
    ind = np.argmax(eta)
    return thresholds[ind]



def r2_fraction(chunk1,chunk2):
    eps = 0.001
    try:
        D1 = lslinear([xx[0] for xx in chunk1],[xx[1] for xx in chunk1])
        D2 = lslinear([xx[0] for xx in chunk2],[xx[1] for xx in chunk2])
    except RegressionError:
        return 0
    if (D2['r2']<eps):
        return 0
    if (D1['r2']>100):
        return 0
    if D1['r2']<eps:
        return 0
    return np.log((D1['r2']/D2['r2']))
