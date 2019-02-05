import numpy as np
from sklearn.linear_model import LinearRegression
from ..config import *

class RegressionError(Exception):
    pass

def lslinear(x,y):
    if len(x)<config.MIN_REGRESSION_POINTS:
        raise RegressionError('Cannot regress with fewer than required points.')
    D = {}
    w = sorted(zip(x,y))
    x = [xx[0] for xx in w]
    y = [xx[1] for xx in w]
    D['x'] = x
    D['y'] = y
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    D['r2'] = 100*reg.score(x, y)
    D['intercept'] = reg.intercept_[0]
    D['slope'] = reg.coef_[0][0]
    return D


class Regressor:
    def __init__(self,slope,intercept,r2,kind='independent',popt=None):
        self.slope = slope
        self.intercept = intercept
        self.r2 = r2
        self.popt = popt
        self.kind = kind
        self.is_good = False

    def __str__(self):
        return f'Regressor({self.slope:.3f},{self.intercept:.3f},{self.r2:.0f})'

    def __repr__(self):
        return f'Regressor({self.slope},{self.intercept},{self.r2})'

    def isNull(self):
        return self.r2==0

    def predict(self,x):
        return self.intercept + x*self.slope

    def correct(self,p):
        return p[1] - p[0]*self.slope

    def to_list(self):
        """Convert the object to a list containing slope, Intercept and R2

        :return: A list with three elements: Slope, Intercept and R2
        """
        return [self.slope, self.intercept, self.r2]

def create_null_regressor():
    tmp = np.log(1/config.NOS)
    return Regressor(0,tmp,0,'Null')
