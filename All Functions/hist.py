import numpy as np
import matplotlib.pyplot as plt
np.random.seed(90099)
def custom_hist(x,n):
	width = np.around((np.amax(x)-np.amin(x))/n, decimals=1)
	x = np.sort(x)
	bins = np.linspace(np.amin(x),np.amax(x),n+1)
	hist = np.zeros(n)
	for i in range(n):
#		idx = np.where(np.logical_and(x>=np.amin(x)+i*width,x<np.amin(x)+(i+1)*width))
		if(i != n-1):
			idx = np.where(np.logical_and(x>=bins[i],x<bins[i+1]))
		else:
			idx = np.where(np.logical_and(x>=bins[i],x<=bins[i+1]))
		hist[i] = len(idx[0])
	return (hist,bins)
'''
#x = np.array([2.2,3.3,7.6,13.3,1.8,1.9,20.2,21.6,21.7,22.27])
x = np.random.rand(10)
print(np.sort(x))
custom_hist,custom_bins = custom_hist(x,5)
print(custom_hist,custom_bins)
y = plt.hist(x,5)
print(y)
'''
