import numpy as np
import itertools
from scipy import stats
import root_finding_methods as rf
import differentiation_methods as dm
import sys

def gradient_search(f,df,num_epochs=1000,a=None,lr=0.2,descent=False):
	if(a is None):
		a = np.random.rand()
	if(descent):
		for i in range(num_epochs):
			da = df(a)
			a -= lr*da		
	else:
		for i in range(num_epochs):
			da = df(a)
			a += lr*da
	return a

def grid_search(f,a,opt_type="min"): # or max # a = [[0,1,0.1],[2,3,0.4]] # grid start, stop, step
	a_opt = None
	for i in range(len(a)):
		a[i] = np.arange(a[i][0],a[i][1],a[i][2])
	if(opt_type == "min"):
		tmp = np.inf
		for element in itertools.product(*a):
			if(f(*element)<tmp):
				a_opt = element
				tmp = f(*element)
	if(opt_type == "max"):
		tmp = -np.inf
		for element in itertools.product(*a):
			if(f(*element)>tmp):
				a_opt = element
				tmp = f(*element)
	return a_opt

def rchi2(y_pred,y,yerr,dof):
    s = np.sum(((y_pred-y)/yerr)**2)
    v = (len(y)-dof)
    s /= v
    p = 1 - stats.chi2.cdf(s, v)
    return s,p

def mse(y_pred,y):
    return np.sum((y_pred-y)**2)

def curvefit_grad_search(f,a,x,y,yerr=None,metric="rchi2",num_epochs=1000,h=0.001,lr=1e-5): # a is initial seed point , a = [a1,a2,a3,...,an]
	for epoch in range(num_epochs):
		da = np.zeros(len(a))
		for i in range(len(a)):
			red_chi2 = np.array([])
			a_i = np.array([])
			for j in range(-1,2,1):
				a[i] += j*h
				y_pred = np.array([f(*x[k],*a) for k in range(len(x))])
				s,p = rchi2(y_pred,y,yerr,len(a))
				red_chi2 = np.append(red_chi2,s)
				a_i = np.append(a_i,a[i])
				a[i] -= j*h
			da[i],_ = dm.dydx(a_i,red_chi2,h=h)
		# updating the values of parameters
		a -= lr*da
	# finding error values in the parameters
	# d2chi2/dx2
	da = np.zeros(len(a))
	for i in range(len(a)):
		red_chi2 = np.array([])
		a_i = np.array([])
		for j in range(-1,2,1):
			a[i] += j*h
			y_pred = np.array([f(*x[k],*a) for k in range(len(x))])
			s,p = rchi2(y_pred,y,yerr,len(a))
			red_chi2 = np.append(red_chi2,s)
			a_i = np.append(a_i,a[i])
			a[i] -= j*h
		da[i],_ = dm.d2ydx2(a_i,red_chi2,h=h)
	da = np.sqrt(1/da)
	return a,da

def curvefit_grid_search(f,a,x,y,yerr=None,metric="rchi2"): # or max # a = [[0,1,0.1],[2,3,0.4]] # grid start, stop, step
	ret_arr = []
	a_opt = None
	a_opt_err = []
	tmp = np.inf
	a_copy = np.copy(a)
	for i in range(len(a)):
		a[i] = np.arange(a[i][0],a[i][1],a[i][2])
	if(metric == "rchi2"):
		for element in itertools.product(*a):
			y_pred = np.array([f(*x[i],*element) for i in range(len(x))])
			s,p = rchi2(y_pred,y,yerr,len(a))
			if(s<tmp):
				a_opt = element
				tmp = s
		a_opt = list(a_opt)
		for k in range(len(a_opt)):
			s = []
			for j in range(-1,2,1):
				a_opt[k] += j*a_copy[k][-1] 
				y_pred = np.array([f(*x[i],*a_opt) for i in range(len(x))])
				tmp,_ = rchi2(y_pred,y,yerr,len(a_opt))
				s.append(tmp)
				a_opt[k] -= j*a_copy[k][-1] 
			tmp = a_opt[k] - a_copy[k][-1]*(((s[2]-s[1])/(s[0]-2*s[1]+s[2]))+0.5)
			ret_arr.append(tmp)
			tmp = a_copy[k][-1]*np.sqrt(2.0/(s[0]-2*s[1]+s[2]))
			a_opt_err.append(tmp)
		return ret_arr,a_opt_err 
	if(metric == "mse"):
		for element in itertools.product(*a):
			y_pred = np.array([f(*x[i],*element) for i in range(len(x))])
			s = mse(y_pred,y)
			if(s<tmp):
				a_opt = element
				tmp = s
		return a_opt

def plot(x,y,xlabel,ylabel,title,savename,ft=15,typ="semilogy"):
  plt.figure(figsize=(8,6))
  if(typ == "scatter"):
    plt.scatter(x,y)  
  if(typ == "semilogy"):
    plt.semilogy(x,y)  
    plt.yscale('symlog')
  if(typ == "plot"):
    plt.plot(x,y)  
  if(typ == "semilogx"):
    plt.semilogx(x,y)  
    plt.xscale('symlog')
  if(typ == "loglog"):
    plt.loglog(x,y)  
    plt.yscale('symlog')
    plt.xscale('symlog')
  plt.grid()
  plt.xlabel(xlabel,fontsize=ft)
  plt.ylabel(ylabel,fontsize=ft)
  plt.title(title,fontsize=ft)
  plt.xticks(fontsize=ft)
  plt.yticks(fontsize=ft)
  plt.savefig("{}.png".format(savename), bbox_inches="tight")

def plot_hist(data,bins,hist,xlabel,title,savename,frac_w=1.25,ft=15,a = 5.5,typ="plot",norm=False): # plot or scatter
	plt.figure(figsize=(8,6))
	x1 = bins[:-1]
	x2 = np.append(x1[1::],bins[-1])
	y = np.where(norm,(hist*frac_w)/np.amax(hist),hist)
	w = (np.array(x2) - np.array(x1))/frac_w   
	plt.bar(x1, y, width=w, align='edge',alpha=0.5)
	x = np.arange(-1,1,0.01)
	if(typ == "scatter"):
		plt.scatter(data,p(data,a),c='r',label="Theoratical PDF")
	if(typ == "plot"):
		plt.plot(data,p(data,a),c='r',label="Theoratical PDF")
	plt.xlabel(xlabel,fontsize=ft)
	plt.title(title,fontsize=ft)
	plt.xticks(fontsize=ft)
	plt.yticks(fontsize=ft)
	plt.savefig("{}.png".format(savename), bbox_inches="tight")

'''f = lambda x,y,z,a1,a2,a3: a1*x**2+a2*y+a3*z
x = np.random.rand(100,3)
a = [1,0,3]
y = [f(*x[i],*a) for i in range(len(x))]
y_err = 0.1*np.random.randn(100)
y_err *= y
arr = [[0.9,1.1,0.01],[-0.1,0.1,0.01],[2.9,3.1,0.01]] 
a_fit = curvefit_grid_search(f=f,a=arr,x=x,y=y,yerr=y_err,metric="rchi2")
print(a_fit)'''
