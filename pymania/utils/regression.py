from scipy.optimize import curve_fit
import numpy as np

class RegressionError(Exception):
    pass

def lslinear(x,y):
    if len(x)<2:
        raise RegressionError('Cannot regress with fewer than two points.')
    D = {}
    w = sorted(zip(x,y))
    x = [xx[0] for xx in w]
    y = [xx[1] for xx in w]
    D['x'] = x
    D['y'] = y
    popt, pcov = curve_fit(func, x, y,maxfev = 10000)
    D['popt'] = popt
    D['std'] = np.sqrt(pcov[0,0])
    # a = sorted(list(set(a)))
    z = [func(xx,*popt) for xx in x]
    D['z'] = z
    residuals = [(y[q] - z[q])**2 for q in range(len(z))]
    ss_res = np.sum(residuals)
    ss_tot = np.sum((np.array(y)-np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)
    r_squared = r_squared * 100
    D['r2'] = r_squared
    D['intercept'] = popt[1]
    D['slope'] = popt[0]
    return D

def func(x, a, b):
    return a*x + b

class Regressor:
    def __init__(self,slope,intercept,r2,popt=None):
        self.slope = slope
        self.intercept = intercept
        self.r2 = r2
        self.popt = popt

    def predict(self,x):
        return self.intercept + x*self.slope

    def correct(self,p):
        return p[1] - p[0]*self.slope
