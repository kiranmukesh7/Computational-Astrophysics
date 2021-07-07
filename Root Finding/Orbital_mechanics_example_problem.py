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
ap.add_argument('-xbis','--x_Bis', nargs='+', required=True, help="Interval", type=float)
ap.add_argument("-xnr", "--x_NR", required=True, help="Initial guess")
ap.add_argument("-show", "--show", required=False, help="Show plot or not",default="False")
args = vars(ap.parse_args())

e = float(args["e"])
t = args["t"]
eps = float(args["eps"])
etype = str(args["etype"])
a = const.au.value
b = a*np.sqrt(1-e**2)
show = args["show"]

if(show.lower() == "true"):
	import matplotlib.pyplot as plt
	E = np.arange(0,2*np.pi,0.01)
	plt.plot(E,f(E,t[2],e))
	plt.show()
	sys.exit()

x_NR = float(args["x_NR"])
x_Bis = args["x_Bis"]

print("Using Newtom-Raphson Method:\n")
for i in t:
	E,n_iters = newton_raphson(f,df_dx,x_NR,eps,etype,i)
	print("On {}th day, The cartesian coordinates of Earth is ({},{}) AU".format(i,np.round(np.cos(E),3), np.round((b/a)*np.sin(E),3)))
	print("Number of iterations : {}".format(n_iters))

print("Using Bisection Method:\n")
for i in t:
	E,n_iters = bisection(f,0.01,2*np.pi,eps,etype,i)
	print("On {}th day, The cartesian coordinates of Earth is ({},{}) AU".format(i,np.round(np.cos(E),3), np.round((b/a)*np.sin(E),3)))
	print("Number of iterations : {}".format(n_iters))
