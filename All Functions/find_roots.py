import numpy as np
import root_finding_methods as rf
import sys

# Finding Wein's displacement law in terms of frequency:
#const = (4.799237345) * 10**(-11)
#f = lambda x: 3 - (x/(1-np.exp(-x))
#df_dx = lambda x: (-1./(1-np.exp(-x)))*(1+(x/(np.exp(x)-1)))
#x_Bis = [0,30]

# Finding Wein's displacement law in terms of wavelength:
const = 69.50356301
f = lambda x: 1./(1-np.exp(-1./x)) - 5*x
df_dx = lambda x: -5 + (np.exp(-1./x)/x**2)/(1-np.exp(-1.0/x))**2
x_Bis = [0,1]

xtol = 1e-6
ftol = 1e-4
max_iters = 10000
etype = "abs"
x_Bis = [0,30]

E_arr, idx = rf.get_roots_idx(x_Bis,f,df_dx)

if(len(idx) == 0):
	print("No Roots in given interval!")
	sys.exit()

print("Roots found by Bisection method: \n")

for i in idx:
	E,n_iters,dE = rf.bisection(f=f,x1=E_arr[i-1],x2=E_arr[i],xtol=xtol,ftol=ftol,max_iters=max_iters,)
#	print(E,f(E))
	print("E = {} +/- {}, f(E) = {}".format(E/const,dE/const,f(E)))

print("\n Roots found by Newton Raphson method: \n")

for i in idx:
	E,n_iters = rf.newton_raphson(f=f,fprime=df_dx,z=E_arr[i],xtol=xtol,ftol=ftol,max_iters=max_iters,stop="abs")
#	print(E,f(E))
	print(E/const,f(E))

