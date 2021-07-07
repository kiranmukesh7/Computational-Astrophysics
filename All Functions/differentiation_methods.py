import numpy as np

def dydx(x,y=None,method="central",f=None,h=0.01,*argv):
	if(f is None):
		if(len(x)==len(y)):
			if(method=="forward"):
				zy = y[1:] - y[:-1]
				zx = x[1:] - x[:-1]
				return zy/zx, x[:-1]
			if(method=="backward"):
				zy = y[1:] - y[:-1]
				zx = x[1:] - x[:-1]
				return zy/zx, x[1:]
			if(method=="central"):
				zy = y[2:] - y[:-2]
				zx = x[2:] - x[:-2]
				return zy/zx, x[1:-1]
			else:
				print("Method = central, forward or backward")
		else:
			print("len(x) != len(y)")
			return
	else:
		if(method=="forward"):
			return (f(x + h,*argv) - f(x,*argv))/h
		if(method=="backward"):
			return (f(x,*argv) - f(x - h,*argv))/h
		if(method=="central"):
			return (f(x + h,*argv) - f(x - h,*argv))/(2*h) 
		else:
			print("Method = central, forward or backward")		

def d2ydx2(x,y=None,method="central",f=None,h=0.01,*argv):
	if(f is None):
		if(len(x)==len(y)):
			if(method=="forward"):
				zy = y[2:] - 2*y[1:-1] + y[:-2]
				zx = x[2:] - x[:-2]
				return zy/zx, x[:-2]
			if(method=="backward"):
				zy = y[:-2] - 2*y[1:-1] + y[2:]
				zx = x[2:] - x[:-2]
				return zy/zx, x[2:]
			if(method=="central"):
				zy = y[2:] + y[:-2] - 2*y[1:-1]
				zx = x[2:] - x[:-2]
				return zy/zx, x[1:-1]
			else:
				print("Method = central, forward or backward")
		else:
			print("len(x) != len(y)")
			return
	else:
		if(method=="forward"):
			return (f(x + 2*h,*argv) - 2*f(x + h,*argv) + f(x,*argv))/h**2
		if(method=="backward"):
			return (f(x - 2*h,*argv) - 2*f(x - h,*argv) + f(x,*argv))/h**2
		if(method=="central"):
			return (f(x + h,*argv) + f(x - h,*argv) - 2*f(x,*argv))/h**2
		else:
			print("Method = central, forward or backward")	
		
def richardson_interpolation(J,x,f,df,h,method="central",*argv):
	T = np.zeros((J,J,len(x)))
	k0 = np.where(method == "central",2,1)
	for j in range(J):
		T[j][0] = df(x=x,f=f,h=h,method=method,*argv)
		for k in range(1,j+1):
			T[j][k] = T[j][k-1] + (T[j][k-1] - T[j-1][k-1])/((2**k0)**(k) - 1)
			h /= 2
	return T

def richdydx(x,y=None,method="central",f=None,h=0.01,*argv):
	return (-f(x+2*h) + 8*f(x+h) - 8*f(x-h) + f(x-2*h))/(12*h)

#y = np.array([0,1,2,3,4,5])
#f = lambda x: np.exp(x)
#x = np.array([0,1,2,3,4,5])
#x = np.array([0])
#T = richardson_interpolation(J=2,x=x,f=f,df=dydx,h=0.1,method="central")
#print(T[:,:,0])
#print(dydx(x=x,f=f,h=0.1))
#print(richdydx(x=x,f=f,h=0.1))
#diff = (4*dydx(x=x,f=f,h=0.05,method="central") - dydx(x=x,f=f,h=0.1,method="central"))/3.0
#print(diff)
#print(richardson_interpolation(J=2,x=x,f=f,df=dydx,h=0.1,method="backward"))
#print(richardson_interpolation(J=2,x=x,f=f,df=dydx,h=0.1,method="central"))

