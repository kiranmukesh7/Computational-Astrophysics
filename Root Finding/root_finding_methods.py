import numpy as np

def stopping_criteria(x1,x2,stop_type):
	if(stop_type == "abs"):
		return np.abs(x2 - x1)
	if(stop_type == "rel"):
		return np.abs(x2/x1 - 1.0)

def bisection(f,x1,x2,eps,stop="abs",*args):
	if(f(x1,*args)*f(x2,*args)>0):
		print("Both f(x1) and f(x2) have same sign! There possibly does not exist a unique root in the given interval. ")
		return
	eps1 = stopping_criteria(x1,x2,stop)
	x0 = 0
	ctr = 0
	while(eps1 >= eps):
		if(f(x1,*args)*f(x2,*args)<0):
			x0 = (x1+x2)/2.0;
			x1,x2 = np.where(f(x1,*args)*f(x0,*args) < 0,(x1,x0),(x0,x2))
			eps1 = stopping_criteria(x1,x2,stop)
			ctr += 1
	return x0,ctr

def newton_raphson(f1,f2,x0,eps,stop="abs",*arg):
	x1 = x0 - (f1(x0,*arg)/f2(x0,*arg))
	ctr = 0
	eps1 = stopping_criteria(x1,x0,stop)
	while(eps1>=eps):
		x0 = x1;
		x1 = x0 - (f1(x0,*arg)/f2(x0,*arg))
		eps1 = stopping_criteria(x1,x0,stop)
		ctr += 1
	return x1,ctr
