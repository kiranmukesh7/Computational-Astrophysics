import numpy as np

def stopping_criteria(x1,x2,stop_type):
	if(stop_type == "abs"):
		return np.abs(x2 - x1)
	if(stop_type == "rel"):
		return np.abs(x2/x1 - 1.0)

def bisection(f,x1,x2,max_iters=100,xtol=1e-6,ftol=1e-4,*args):
	if(f(x1,*args)*f(x2,*args)>0):
		print("Both f(x1) and f(x2) have same sign! There possibly does not exist a unique root in the given interval. ")
		return
	elif(f(x1,*args) == 0):
		return x1, 0
	elif(f(x2,*args) == 0):
		return x2, 0
	tol = stopping_criteria(x1,x2,"abs")
	x0 = (x1+x2)/2.0;
	ctr = 1
	while(stopping_criteria(x1,x2,"abs")>xtol and abs(f(x0,*args)) > ftol and ctr <max_iters):
		#if(f(x1,*args)*f(x2,*args)<0):
			x1,x2 = np.where(f(x1,*args)*f(x0,*args) < 0,(x1,x0),(x0,x2))
			ctr += 1
	tol = stopping_criteria(x1,x2,"abs")
	return x0,ctr,tol/2.0

def newton_raphson(z, f, fprime, fdoubleprime = None, max_iters=100, stop="abs", xtol=1e-6, ftol=1e-4,*arg):
    """The Newton-Raphson method."""
    err = 0
    for i in range(max_iters):
        step = f(z,*arg)/fprime(z,*arg)
        if(fdoubleprime is not None):
            err = fdoubleprime(z,*arg)/fprime(z,*arg)
            err *= (-0.5)*(step**2)
        if(stopping_criteria(z-step,z,stop) < xtol and abs(f(z,*arg)) < ftol):
            if(fdoubleprime is None):
                return z, i
            else:
                return z, i,err
        z -= step
    if(abs(step) > xtol or abs(f(z,*arg)) > ftol):
        print("Defined tolerance levels not reached! Need More iterations")
    if(fdoubleprime is None):
        return z,i
    else:
        return z,i,err
'''
# copy of original code -- before modification to include error estimate
def newton_raphson(z, f, fprime, fdoubleprime, max_iters=100, stop="abs", xtol=1e-6, ftol=1e-4,*arg):
    """The Newton-Raphson method."""
    for i in range(max_iters):
        step = f(z,*arg)/fprime(z,*arg)
        if(stopping_criteria(z-step,z,stop) < xtol and abs(f(z,*arg)) < ftol):
            return z, i
        z -= step
    if(abs(step) > xtol or abs(f(z,*arg)) > ftol):
        print("Defined tolerance levels not reached! Need More iterations")
    return z, i
'''
def plot_newton_iters(f, df_dx, n=200, extent=[-1,1,-1,1]):
    """Shows how long it takes to converge to a root using the Newton-Rahphson method."""
    m = np.zeros((n,n))
    xmin, xmax, ymin, ymax = extent
    for r, x in enumerate(np.linspace(xmin, xmax, n)):
        for s, y in enumerate(np.linspace(ymin, ymax, n)):
            z = x + y*1j
            m[r, s] = newton_raphson(z, f, df_dx)[1]
    return m

def plot_newton_basins(f, df_dx, n=200, extent=[-1,1,-1,1]):
    """Shows basin of attraction for convergence to each root using the Newton-Raphson method."""
    root_count = 0
    roots = {}

    m = np.zeros((n,n))
    xmin, xmax, ymin, ymax = extent
    for r, x in enumerate(np.linspace(xmin, xmax, n)):
        for s, y in enumerate(np.linspace(ymin, ymax, n)):
            z = x + y*1j
            root = np.round(newton_raphson(z, f, df_dx)[0], 1)
            if(not root in roots):
                roots[root] = root_count
                root_count += 1
            m[r, s] = roots[root]
    return m

def get_roots_idx(x_Bis,f,df_dx,step=0.01):
	E_arr = np.arange(x_Bis[0],x_Bis[1],step)
	idx = np.where(E_arr != 0)
	E_arr = E_arr[idx]
	f_E = f(E_arr)
	asign = np.sign(f_E)
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
	idx = np.where(signchange == 1)[0][1:]
	valid_idx = []
	for i in idx:
		if( np.abs(df_dx(E_arr[i]) - df_dx(E_arr[i-1]) ) < 0.1 ):
			valid_idx.append(i)
	idx = np.array(valid_idx)
	return E_arr, idx


'''def newton_raphson_old(f1,f2,x0,eps,max_iters,stop="abs",*arg):
	x1 = x0 - (f1(x0,*arg)/f2(x0,*arg))
	ctr = 0
	eps1 = stopping_criteria(x1,x0,stop)
	while(eps1>=eps and ctr >= max_iters):
		x0 = x1;
		x1 = x0 - (f1(x0,*arg)/f2(x0,*arg))
		eps1 = stopping_criteria(x1,x0,stop)
		ctr += 1
	return x1,ctr

def newton_raphson_regularized(f1,f2,x0,eps,feps=None,stop="abs",*arg):
	if(feps is None):
		feps = eps
	x1 = x0 - (f1(x0)/f2(x0))
	ctr = 0
	eps1 = stopping_criteria(x1,x0,stop)
	while(np.abs(x1-x0)>=eps or np.abs(f1(x1)) >= feps):
		x0 = x1;
		x1 = x0 - (f1(x0)/f2(x0))
		eps1 = stopping_criteria(x1,x0,stop)
		ctr += 1
	return x1,ctr'''


'''def bisection(f,x1,x2,xtol,*args):
	if(f(x1,*args)*f(x2,*args)>0):
		print("Both f(x1) and f(x2) have same sign! There possibly does not exist a unique root in the given interval. ")
		return
	tol = stopping_criteria(x1,x2,"abs")
	N = int(np.floor(tol/xtol) + 1)
	x0 = 0
	ctr = 0
	for i in range(N):
		if(f(x1,*args)*f(x2,*args)<0):
			x0 = (x1+x2)/2.0;
			x1,x2 = np.where(f(x1,*args)*f(x0,*args) < 0,(x1,x0),(x0,x2))
	return x0,N'''

