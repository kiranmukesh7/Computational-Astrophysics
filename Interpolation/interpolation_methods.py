import numpy as np

class interp():

    def __init__(self,xin,xsmpl,fsmpl,xerr=None,yerr=None):
        self.xin = xin
        self.xsmpl = xsmpl
        self.fsmpl = fsmpl
        if(xerr != None):
            self.xerr = xerr
        if(yerr != None):
            self.yerr = yerr

    def dp_dx1(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return self.fsmpl[i]*((self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])**2) + self.fsmpl[i+1]*((self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])**2) - self.fsmpl[i+1]/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_dx2(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return -self.fsmpl[i]*((self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])**2) - self.fsmpl[i+1]*((self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])**2) + self.fsmpl[i]/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_df1(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return (self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_df2(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return (self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])

    def interp(self,typ="linear"):
        '''
        input arguments: xin, xsmpl, fsmpl, typ
        Returns the one-dimensional piecewise interpolant to a function with given discrete data points (xp, fp), evaluated at x.
        Type of interpolation to be performed is to be specified using the typ argument.
        '''
        z = np.zeros_like(self.xin)
        is_err = False
        if(self.xerr != None or self.yerr != None):
            zerr = np.zeros_like(self.xin)
            is_err = True
        N = len(self.xsmpl)
        if(typ == "lagrange"):
            for j in range(N):
                prod = self.fsmpl[j]*np.ones_like(self.xin)
                for i in range(N):
                    if(i!=j):
                        prod *= (self.xin-self.xsmpl[i])/(self.xsmpl[j]-self.xsmpl[i])
                z += prod
            return z
        elif(typ == "linear"):
            for i in range(N-1):
                idx = np.where(np.logical_and(self.xin>=self.xsmpl[i],self.xin<=self.xsmpl[i+1]))
                z[idx] = self.fsmpl[i] + (self.xin[idx] - self.xsmpl[i])*((self.fsmpl[i+1] - self.fsmpl[i])/(self.xsmpl[i+1] - self.xsmpl[i]))
                if(is_err):
                    zerr[idx] = int(self.xerr != None)*( np.abs(self.dp_dx1(idx,i))*xerr[i] + np.abs(self.dp_dx2(idx,i))*xerr[i+1] ) + int(self.yerr != None)*( np.abs(self.dp_df1(idx,i))*yerr[i] + np.abs(self.dp_df2(idx,i))*yerr[i+1] )
            if(is_err):
                return z,zerr
            else:
                return z


def interp(xin,xsmpl,fsmpl,typ="linear"):
    '''
    input arguments: xin, xsmpl, fsmpl, typ
    Returns the one-dimensional piecewise interpolant to a function with given discrete data points (xp, fp), evaluated at x.
    Type of interpolation to be performed is to be specified using the typ argument.
    '''
    z = np.zeros_like(xin)
    zerr = np.zeros_like(xin)
    N = len(xsmpl)
    if(typ == "lagrange"):
        for j in range(N):
            prod = fsmpl[j]*np.ones_like(xin)
            for i in range(N):
                if(i!=j):
                    prod *= (xin-xsmpl[i])/(xsmpl[j]-xsmpl[i])
            z += prod
        return z
    elif(typ == "linear"):
        for i in range(N-1):
            idx = np.where(np.logical_and(xin>=xsmpl[i],xin<=xsmpl[i+1]))
            z[idx] = fsmpl[i] + (xin[idx] - xsmpl[i])*((fsmpl[i+1] - fsmpl[i])/(xsmpl[i+1] - xsmpl[i]))
            zerr[idx] = np.abs()
        return z

def lagrange_float(xin,xsmpl,fsmpl): # inefficient code
    z = []
    N = len(xsmpl)
    if(type(xin) == float):
        xin = np.array([xin])
    for xi in xin:
        z.append(np.sum([fsmpl[j]*np.prod([(xin-xsmpl[i])/(xsmpl[j]-xsmpl[i]) for i in np.arange(N) if i!=j]) for j in np.arange(N)]))
    return z
