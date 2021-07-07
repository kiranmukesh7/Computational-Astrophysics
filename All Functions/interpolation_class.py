import numpy as np

class interp:

    def __init__(self,xin,xsmpl,fsmpl,xerr=None,yerr=None):
        self.xin = xin
        self.xsmpl = xsmpl
        self.fsmpl = fsmpl
        self.xerr = None
        self.yerr = None
        self.is_err = False
        if(xerr is not None):
            if(len(xerr) != len(self.xsmpl)):
                print("x and x_err arrays should have same length!")
                return 
            self.xerr = xerr
            self.is_err = True
        if(yerr is not None):
            if(len(yerr) != len(self.fsmpl)):
                print("y and y_err arrays should have same length!")
                return 
            self.yerr = yerr
            self.is_err = True

    def dp_dx1(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return self.fsmpl[i]*((self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])**2) + self.fsmpl[i+1]*((self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])**2) - self.fsmpl[i+1]/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_dx2(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return -self.fsmpl[i]*((self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])**2) - self.fsmpl[i+1]*((self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])**2) + self.fsmpl[i]/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_df1(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return (self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])
        if(typ == "lagrange"):
            return (self.xsmpl[i+1]-self.xin[idx])/(self.xsmpl[i+1]-self.xsmpl[i])

    def dp_df2(self,idx,i,typ="linear"):
        if(typ == "linear"):
            return (self.xin[idx]-self.xsmpl[i])/(self.xsmpl[i+1]-self.xsmpl[i])

    def interp(self,typ="linear",extrapol="True"):
        '''
        input arguments: xin, xsmpl, fsmpl, typ
        Returns the one-dimensional piecewise interpolant to a function with given discrete data points (xp, fp), evaluated at x.
        Type of interpolation to be performed is to be specified using the typ argument.
        '''
        z = np.zeros_like(self.xin)
        zerr = np.zeros_like(self.xin)
        N = len(self.xsmpl)
        if(typ == "lagrange"):
            for j in range(N):
                prod = self.fsmpl[j]*np.ones_like(self.xin)
                if(self.yerr is not None): 
                    err_prod = self.fsmpl[j]*np.ones_like(self.xin)
                for i in range(N):
                    if(i!=j):
                        prod *= (self.xin-self.xsmpl[i])/(self.xsmpl[j]-self.xsmpl[i])
                        if(self.yerr is not None): 
                            err_prod *= (self.xin-self.xsmpl[i])/(self.xsmpl[j]-self.xsmpl[i])
                z += prod
                if(self.yerr is not None): 
                    zerr += err_prod
            if(self.yerr is not None): 
                return z, zerr
            else:
                return z
        elif(typ == "linear"):
            for i in range(N-1):
                if(i == 0 and extrapol):
                    idx = np.where(self.xin<=self.xsmpl[i+1])        
                if(i == N-2 and extrapol):
                    idx = np.where(self.xin>=self.xsmpl[i])
                elif(i>0 and i<=N-2):
                    idx = np.where(np.logical_and(self.xin>=self.xsmpl[i],self.xin<=self.xsmpl[i+1]))
                z[idx] = self.fsmpl[i] + (self.xin[idx] - self.xsmpl[i])*((self.fsmpl[i+1] - self.fsmpl[i])/(self.xsmpl[i+1] - self.xsmpl[i]))
                if(self.xerr is not None):                    
                    zerr[idx] += ( np.abs(self.dp_dx1(idx,i))*self.xerr[i] + np.abs(self.dp_dx2(idx,i))*self.xerr[i+1] ) 
                if(self.yerr is not None):                    
                    zerr[idx] += ( np.abs(self.dp_df1(idx,i))*self.yerr[i] + np.abs(self.dp_df2(idx,i))*self.yerr[i+1] )

            if(self.is_err):
                return z,zerr
            else:
                return z

    def get_interpolator(self,xs,ys,xerr=None,yerr=None,typ="linear"):	
        if(typ == "linear"):
            return lambda x: ys[0] + (x - xs[0])*((ys[1] - ys[0])/(xs[1] - xs[0]))
        if(typ == "lagrange"):
            N = len(xs)
            return lambda x: sum([np.prod([(x-xs[i])/(xs[j]-xs[i]) for i in range(N) if i!= j]) for j in range(N)])

'''
def interp(xin,xsmpl,fsmpl,typ="linear"):
  
    input arguments: xin, xsmpl, fsmpl, typ
    Returns the one-dimensional piecewise interpolant to a function with given discrete data points (xp, fp), evaluated at x.
    Type of interpolation to be performed is to be specified using the typ argument.

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
    return z'''
