import numpy as np

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

def grid_search(f,a,opt_type="min"): # or max # a = {a1:[0,1,0.1]} # grid start, stop, step
	a_opt = 0
	a_arr = np.arange(a[0],a[1],a[2])
	if(opt_type == "min"):
		tmp = np.inf
		for i in a_arr:
			if(f(i)<tmp):
				a_opt = i
	if(opt_type == "max"):
		tmp = -np.inf
		for i in a_arr:
			if(f(i)>tmp):
				a_opt = i
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
