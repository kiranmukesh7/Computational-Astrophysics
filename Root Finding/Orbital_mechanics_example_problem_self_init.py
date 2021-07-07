import numpy as np
from root_finding_methods import *
from astropy import constants as const
import argparse
import sys

def f(x,t,e=0.0167,P=365.25635):
	return ((2.0*np.pi)/P)*(t) + e*np.sin(x) - x
def df_dx(x,t):
	return e*np.cos(x) - 1.0

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--e", required=False, help="eccentricity", default=0.0167)
ap.add_argument('-t','--t', nargs='+', required=True, help="Times at which the position is to be calculated", type=float)
ap.add_argument("-eps", "--eps", required=False, help="tolerance", default=1e-7)
ap.add_argument("-etype", "--etype", required=False, help="Absolute/Relative error", default="abs")
args = vars(ap.parse_args())

e = float(args["e"])
t = args["t"]
eps = float(args["eps"])
etype = str(args["etype"])
a = const.au.value
b = a*np.sqrt(1-e**2)
E_arr = np.arange(0,2*np.pi,0.01)

print("Using Newtom-Raphson Method:\n")
for i in t:
	idx = np.argmax(np.abs(df_dx(E_arr,i)))
	x_NR = E_arr[idx]
	E,n_iters = newton_raphson(f,df_dx,x_NR,eps,etype,i)
	print("On {}th day, The cartesian coordinates of Earth is ({},{}) AU".format(i,np.round(np.cos(E),3), np.round((b/a)*np.sin(E),3)))
	print("Number of iterations : {}".format(n_iters))

print("Using Bisection Method:\n")
for i in t:
	f_E = f(E_arr,i,e)
	asign = np.sign(f_E)
	signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
	idx = np.where(signchange == 1)[0][1] # selecting something in between the range of E_arr
	x_Bis = np.array([E_arr[idx-1],E_arr[idx]])
	E,n_iters = bisection(f,0.01,2*np.pi,eps,etype,i)
	print("On {}th day, The cartesian coordinates of Earth is ({},{}) AU".format(i,np.round(np.cos(E),3), np.round((b/a)*np.sin(E),3)))
	print("Number of iterations : {}".format(n_iters))

