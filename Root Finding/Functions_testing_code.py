import numpy as np
from root_finding_methods import *

def f(x):
	return np.cos(x) - x**3

def df_dx(x):
	return -np.sin(x) - 3*x**2

eps = 1e-6
print("Root found by Bisection method: {}".format(bisection(f,0,1,eps)[0]))
print("Root found by Newton Raphson method: {}".format(newton_raphson(f,df_dx,1,eps)[0]))
