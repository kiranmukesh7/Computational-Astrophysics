import numpy as np
import matplotlib.pyplot as plt
import custom_linalg as linalg
from scipy import stats

def get_best_fit_func(x,y,yerr,basis_f):
    N = len(x)
    A = np.array([[basis_f[j](x[i])/yerr[i] for j in range(len(basis_f.keys()))] for i in range(N)])
    b = np.array([[(y[i])/yerr[i] for j in range(1)] for i in range(N)])
    alpha = linalg.matmul(A.T,A)
    beta = linalg.matmul(A.T,b)
    epsilon = np.linalg.inv(alpha)
    a = linalg.matmul(epsilon,beta)
    da = np.diag(epsilon)
    f = lambda x: np.sum([a[i]*basis_f[i](x) for i in range(len(a))])
    return f,a,epsilon

def get_best_polyfit(x,y,yerr,degree):
    M = degree + 1
    N = len(x)
    A = np.array([[(x[i]**j)/yerr[i] for j in range(M)] for i in range(N)])
    b = np.array([[(y[i])/yerr[i] for j in range(1)] for i in range(N)])
    alpha = linalg.matmul(A.T,A)
    beta = linalg.matmul(A.T,b)
    epsilon = np.linalg.inv(alpha)
    a = linalg.matmul(epsilon,beta)
    da = np.diag(epsilon)
    f = lambda x: np.sum([a[i]*(x**i) for i in range(len(a))])
    return f,a,epsilon

def get_rchi2(f,x,y,yerr,dof):
    s = 0
    for i in range(len(x)):
        s += ((f(x[i])-y[i])/yerr[i])**2
    v = (len(x)-dof)
    s /= v
    p = 1 - stats.chi2.cdf(s, v)
    return s,p

def plot(x,y,yerr,yfit,yerr_fit,f=None,xlabel="x",ylabel="y",title="Curve Fit: Linear Regression",savename="linefit",ft = 17.5,typ="with_err"): # w/o error in model
    fig1 = plt.figure(1,figsize=(8,6))
    #Plot Data-model
    frame1 = fig1.add_axes((.1,.3,.8,.6))
    plt.errorbar(x=x,y=y,yerr=yerr,fmt='.b',capsize=4,label="Data",ecolor='k',elinewidth=1.3)
    if(typ == "without_err"):
        xlinspace = np.linspace(np.amin(x),np.amax(x),1000) 
        plt.plot(xlinspace,np.array([f(i) for i in xlinspace]),'r',label="Best fit Line")
    if(typ == "with_err"):
        plt.errorbar(x=x,y=yfit,yerr=yerr_fit,fmt='-r',capsize=4,label="Model",ecolor='k',elinewidth=1.3)
    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
    plt.ylabel(ylabel,fontsize=ft)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.grid()
    plt.legend(fontsize=ft)
    plt.title(title,fontsize=ft)
    #Residual plot
    difference = yfit - y
    frame2=fig1.add_axes((.1,.1,.8,.2))        
    plt.errorbar(x=x,y=difference,yerr=yerr+yerr_fit,fmt='b.',capsize=4,ecolor='k',elinewidth=1.3)
    plt.ylabel("Residue",fontsize=ft)
    plt.xlabel(xlabel,fontsize=ft)
    plt.xticks(fontsize=ft)
    plt.yticks(fontsize=ft)
    plt.grid()
    plt.savefig("{}.png".format(savename),bbox_inches="tight")
    plt.close()

def get_y_fit_err(x,f,cov,basis_f):
    x_fit = np.array( [[basis_f[j](x[i]) for i in range(len(x))] for j in range(len(basis_f.keys()))] )    
    y_fit = np.array([f(i) for i in x])
    s = linalg.matmul(cov,x_fit)
    s = np.diag(linalg.matmul(x_fit.T,s))
    y_fit_err = np.sqrt(s)
    return y_fit_err
