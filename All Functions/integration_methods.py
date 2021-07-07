import numpy as np

def midpoint(f,a,b,n=1000,*args): # n = number of intervals
    x = np.linspace(a,b,n+1,True)
    x = 0.5*(x[1:]+x[:-1])
    I = np.sum(f(x,*args))
    I *= np.abs(float(b-a))/(n)
    return I

def trapz(f,a,b,n=1000,*args): # n = number of intervals
    x = np.linspace(a,b,n+1,True)
    I = np.sum(f(x[1:-1],*args))
    I += 0.5*(f(x[0],*args)+f(x[-1],*args))
    I *= np.abs(float(b-a))/(n)
    return I

def simpson13(f,a,b,n=1000,*args): # each of the n intervals are again divided into 2 intervals
    if(n%2 == 0):
        x = np.linspace(a,b,n+1,True)
        I = (1./3)*(f(x[0],*args)+f(x[-1],*args))
        I += (4.0/3)*np.sum(f(x[1::2],*args))
        I += (2.0/3)*np.sum(f(x[2:-1:2],*args))
        I *= np.abs(float(b-a))/(n)
        return I
    else:
        print("Input Even number of intervals!")
        return None

def simpson38(f,a,b,n=3000,*args): # each of the n intervals are again divided into 2 intervals
    if(n%3 == 0):
        x = np.linspace(a,b,n+1,True)
        I = (f(x[0],*args)+f(x[-1],*args))
        I += 3*np.sum(f(x[1:-1:3],*args))
        I += 3*np.sum(f(x[2:-1:3],*args))
        I += 2*np.sum(f(x[3:-1:3],*args))
        I *= (3/8)*np.abs(float(b-a))/(n)
        return I
    else:
        print("Input number of intervals which is multiple of 3!")
        return None
    
def quartic_poly(f,a,b,n=400,*args): # each of the n intervals are again divided into 4 intervals
    if(n%4 == 0):
        x = np.linspace(a,b,n+1,True)
        I = (14./45)*(f(x[0],*args)+f(x[-1],*args))
        I += (64.0/45)*np.sum(f(x[1:-1:4],*args))
        I += (24.0/45)*np.sum(f(x[2:-1:4],*args))
        I += (64.0/45)*np.sum(f(x[3:-1:4],*args))
        I += (14.0/45)*np.sum(f(x[4:-1:4],*args))
        I *= np.abs(float(b-a))/(n)
        return I
    else:
        print("Input number of intervals which is multiple of 4!")
        return None

def rhomberg(J,f,a,b,*args): # richardson extrapolation for only trapz method
    h = float(b-a)
    T = np.zeros((J,J))
    for j in range(J):
      n = int(2**(j))
      T[j][0] = trapz(f,a,b,n,*args)
      for k in range(1,j+1):
        T[j][k] = T[j][k-1] + (T[j][k-1] - T[j-1][k-1])/(4**(k) - 1)
      h /= 2
    return T

def richardson_extrapolation(J,f,a,b,t=2,method="trapz",*args): # general richardson extrapolation for integration 
	h = float(b-a)
	multiplier = 1
	if(method == "trapz"):
		method = trapz
		k0 = 2
	if(method == "midpoint"):
		method = midpoint
		k0 = 2
	if(method == "simpson13"):
		method = simpson13
		k0 = 4
		multiplier = 2
	if(method == "simpson38"):
		method = simpson38
		k0 = 4
		multiplier = 3
	T = np.zeros((J,J))
	for j in range(J):
		n = multiplier*int(t**(j))
		T[j][0] = method(f,a,b,n,*args)
		for k in range(1,j+1):
			T[j][k] = T[j][k-1] + (T[j][k-1] - T[j-1][k-1])/((t**k0)**(k) - 1)
		h /= t
	return T

'''
f = lambda x: np.exp((-1)*x**2)
n = 4
a = 0
b = 1
T = richardson_extrapolation(n,f,a,b,method="midpoint")
print(T)
T = richardson_extrapolation(n,f,a,b,method="trapz")
print(T)
T = richardson_extrapolation(n,f,a,b,method="simpson13")
print(T)
T = richardson_extrapolation(n,f,a,b,method="simpson38",t=3)
print(T)
'''
