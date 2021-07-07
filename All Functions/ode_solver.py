import numpy as np
import matplotlib.pyplot as plt

def euler(x,x0,y0,f,n=1): # n = number of steps between x0 and x
    h = (x-x0)/n
    x,y = x0,y0
    for i in range(n):
        y = y + h*f(x,y)
        x += h
    return y
            
def modified_euler(x,x0,y0,f,n=1): # n = number of steps between x0 and x
    h = (x-x0)/n
    x,y = x0,y0
    for i in range(n):
        y_m = y + 0.5*h*f(x,y)
        x_m = x + 0.5*h
        y = y + h*f(x_m,y_m)
        x += h
    return y

def improved_euler(x,x0,y0,f,n=1): # n = number of steps between x0 and x
    h = (x-x0)/n
    x,y = x0,y0
    for i in range(n):
        y_n = f(x,y)
        y = y + 0.5*h*(y_n + f(x+h,y+(h*y_n)))
        x += h
    return y

def rk4(x,x0,y0,f,n=1,*args): # n = number of steps between x0 and x
    h = (x-x0)/n
    x,y = x0,y0
    for i in range(n):
        f0 = f(x,y,*args)
        f1 = f( x+(0.5*h), y + (0.5*h*f0) ,*args)
        f2 = f( x+(0.5*h), y + (0.5*h*f1) ,*args)
        f3 = f( x+h, y + h*f2 ,*args)
        y = y + (h/6.0)*(f0 + 2*f1 + 2*f2 + f3)
        x += h
    return y

def time_stepping_method(x,x0,y0,f,a,b,c,d,n=1): # general function
    h = (x-x0)/n
    x,y = x0,y0
    for i in range(n):
        y_n = f(x,y)
        y = y + h*(a*y_n + b*f(x+c*h,y+d*(h*y_n)))
        x += h
    return y

# 2nd order ODE
#fy = lambda x,y,z: z
#fz = lambda x,z,y: 6*y - z
#F = lambda x: 0.2*(np.exp(2*x)-np.exp(-3*x))

def euler2(x,x0,y0,z0,fy,fz,n=1,*args):
    h = (x-x0)/n
    x = [x0,x0]
    y = [y0,y0]
    z = [z0,z0]
    for i in range(n):
        x[(i+1)%2] = x[i%2]+h
        z[(i+1)%2] = euler(x[(i+1)%2],x[i%2],z[i%2],fz,1,y[i%2],*args)
        y[(i+1)%2] = euler(x[(i+1)%2],x[i%2],y[i%2],fy,1,z[i%2],*args)
    return y[(i+1)%2],z[(i+1)%2]

def modified_euler2(x,x0,y0,z0,fy,fz,n=1,*args):
    h = (x-x0)/n
    x = [x0,x0]
    y = [y0,y0]
    z = [z0,z0]
    for i in range(n):
        x[(i+1)%2] = x[i%2]+h
        z[(i+1)%2] = modified_euler(x[(i+1)%2],x[i%2],z[i%2],fz,1,y[i%2],*args)
        y[(i+1)%2] = modified_euler(x[(i+1)%2],x[i%2],y[i%2],fy,1,z[i%2],*args)
    return y[(i+1)%2],z[(i+1)%2]

def improved_euler2(x,x0,y0,z0,fy,fz,n=1,*args):
    h = (x-x0)/n
    x = [x0,x0]
    y = [y0,y0]
    z = [z0,z0]
    for i in range(n):
        x[(i+1)%2] = x[i%2]+h
        z[(i+1)%2] = improved_euler(x[(i+1)%2],x[i%2],z[i%2],fz,1,y[i%2],*args)
        y[(i+1)%2] = improved_euler(x[(i+1)%2],x[i%2],y[i%2],fy,1,z[i%2],*args)
    return y[(i+1)%2],z[(i+1)%2]

def rk42(x,x0,y0,z0,fy,fz,n=1,*args):
    h = (x-x0)/n
    x = [x0,x0]
    y = [y0,y0]
    z = [z0,z0]
    for i in range(n):
        x[(i+1)%2] = x[i%2]+h
        z[(i+1)%2] = rk4(x[(i+1)%2],x[i%2],z[i%2],fz,1,y[i%2],*args)
        y[(i+1)%2] = rk4(x[(i+1)%2],x[i%2],y[i%2],fy,1,z[i%2],*args)
    return y[(i+1)%2],z[(i+1)%2]

def time_stepping_method2(x,x0,y0,z0,fy,fz,a,b,c,d,n=1,*args):
    h = (x-x0)/n
    x = [x0,x0]
    y = [y0,y0]
    z = [z0,z0]
    for i in range(n):
        x[(i+1)%2] = x[i%2]+h
        z[(i+1)%2] = time_stepping_method(x[(i+1)%2],x[i%2],z[i%2],fz,a,b,c,d,1,y[i%2],*args)
        y[(i+1)%2] = time_stepping_method(x[(i+1)%2],x[i%2],y[i%2],fy,a,b,c,d,1,z[i%2],*args)
    return y[(i+1)%2],z[(i+1)%2]

def euler_ode2(x,y0,z0,fy,fz,method,*args):
	y = np.zeros(len(x))
	z = np.zeros(len(x))
	y[0] = y0
	z[0] = z0
	for i in range(len(x)-1):
		h = x[i+1]-x[i]
		y[i+1] = y[i] + h*fy(x[i],y[i],z[i],*args)
		z[i+1] = z[i] + h*fz(x[i],y[i+1],z[i],*args)
	return y,z

def modified_euler_ode2(x,y0,z0,fy,fz,method,*args):
	y = np.zeros(len(x))
	z = np.zeros(len(x))
	y[0] = y0
	z[0] = z0
	for i in range(len(x)-1):
		h = x[i+1]-x[i]
		y_m = y[i] + 0.5*h*fy(x[i],y[i],z[i],*args) #         y_m = y + 0.5*h*f(x,y)
		z_m = z[i] + 0.5*h*fz(x[i],y[i],z[i],*args)
		x_m = x[i] + 0.5*h # x_m = x + 0.5*h 
		y[i+1] = y[i] + h*fy(x_m,y_m,z_m,*args) # y = y + h*f(x_m,y_m)
		z[i+1] = z[i] + h*fz(x_m,y_m,z_m,*args) 
	return y,z		

def improved_euler_ode2(x,y0,z0,fy,fz,method,*args):
	y = np.zeros(len(x))
	z = np.zeros(len(x))
	y[0] = y0
	z[0] = z0
	for i in range(len(x)-1):
		h = x[i+1]-x[i]

		y_n = fy(x[i],y[i],z[i],*args) # y_n = f(x,y)
		z_n = fz(x[i],y[i],z[i],*args) 

		y[i+1] = y[i] + 0.5*h*(y_n + fy(x[i+1],y[i]+h*y_n,z[i]+h*z_n,*args))
		z[i+1] = z[i] + 0.5*h*(z_n + fz(x[i+1],y[i]+h*y_n,z[i]+h*z_n,*args))
	return y,z		

def rk4_ode2(x,y0,z0,fy,fz,method,*args): 
	y = np.zeros(len(x))
	z = np.zeros(len(x))
	y[0] = y0
	z[0] = z0
	for i in range(len(x)-1):
		h = x[i+1]-x[i]
		ky1 = fy(x[i],y[i],z[i],*args)
		kz1 = fz(x[i],y[i],z[i],*args)

		ky2 = fy(x[i]+(0.5*h),y[i]+(0.5*h*ky1),z[i]+(0.5*h*kz1),*args) # f( x+(0.5*h), y + (0.5*h*f0) ,*args)
		kz2 = fz(x[i]+(0.5*h),y[i]+(0.5*h*ky1),z[i]+(0.5*h*kz1),*args) 

		ky3 = fy(x[i]+(0.5*h),y[i]+(0.5*h*ky2),z[i]+(0.5*h*kz2),*args) # f( x+(0.5*h), y + (0.5*h*f1) ,*args)
		kz3 = fz(x[i]+(0.5*h),y[i]+(0.5*h*ky2),z[i]+(0.5*h*kz2),*args)

		ky4 = fy(x[i]+h,y[i]+h*ky3,z[i]+h*kz3,*args) # f( x+h, y + h*f2 ,*args)
		kz4 = fz(x[i]+h,y[i]+h*ky3,z[i]+h*kz3,*args)

		y[i+1] = y[i] + (h/6.0)*(ky1 + 2*ky2 + 2*ky3 + ky4)
		z[i+1] = z[i] + (h/6.0)*(kz1 + 2*kz2 + 2*kz3 + kz4)
	return y,z
	
'''
fy = lambda x,y,z: z
fz = lambda x,y,z: -y
#F = lambda x: 0.2*(np.exp(2*x)-np.exp(-3*x))

theta_0 = [0,3]
t = np.linspace(0,40,240)

y_euler,z_euler = euler_ode2(t,theta_0[0],theta_0[1],fy,fz,rk42)
y_rk4,z_rk4 = rk4_ode2(t,theta_0[0],theta_0[1],fy,fz,rk42)
y_meuler,z_meuler = modified_euler_ode2(t,theta_0[0],theta_0[1],fy,fz,rk42)
y_ieuler,z_ieuler = improved_euler_ode2(t,theta_0[0],theta_0[1],fy,fz,rk42)

plt.plot(t,y_euler,'r',label="euler")
plt.plot(t,y_rk4,'b',label="rk4")
plt.plot(t,y_meuler,'g',label="modified_euler")
plt.plot(t,y_ieuler,'m',label="improved_euler")
plt.legend()
plt.show()
'''
