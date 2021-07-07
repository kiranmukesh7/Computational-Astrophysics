import numpy as np
import matplotlib.pyplot as plt
import time
import interpolation_methods as ipm

f = lambda x: 1./(25.*x*x+1.)

#f = lambda x: 1./(25.*x*x+1.)

x=np.arange(-1.,1.,0.01)
xsmpl=np.append(x[::20],x[-1])
fsmpl=f(xsmpl)

N=len(xsmpl)
pxj=np.ones(N)

tic = time.time()
z_float = ipm.lagrange_float(x,xsmpl,fsmpl)
toc = time.time()
print("Time taken by looping in Lagrange interpolation method: {} ms".format(np.round((toc-tic)*1e3,3)))

tic = time.time()
z_array = ipm.interp(x,xsmpl,fsmpl,typ="lagrange")
toc = time.time()
print("Time taken by optimized implementation of Lagrange interpolation method: {} ms".format(np.round((toc-tic)*1e3,3)))

plt.plot(x,z_array,'b-')
plt.plot(x,f(x),'k-')
plt.scatter(xsmpl,fsmpl,c='r')
plt.savefig("lagrange.png")
plt.close()

tic = time.time()
z_lin = ipm.interp(x,xsmpl,fsmpl,typ="linear")
toc = time.time()
print("Time taken by optimized implementation of Linear interpolation method: {} ms".format(np.round((toc-tic)*1e3,3)))

plt.plot(x,z_lin,'b-')
plt.plot(x,f(x),'k-')
plt.scatter(xsmpl,fsmpl,c='r')
plt.savefig("linear.png")
plt.close()
