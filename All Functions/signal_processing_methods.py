import numpy as np
import matplotlib.pyplot as plt
import math
import sys

def nextpow2(x):
	return 1 if x == 0 else math.ceil(math.log2(x))

def slow_dft(x):
	x = np.asarray(x, dtype=complex)
	N = x.shape[0]
	n = np.arange(N)
	k = n.reshape((N, 1))
	M = np.exp(-2j * np.pi * k * n / N)
	return np.dot(M, x)

def dit_fft(x):
	x = np.asarray(x, dtype=complex)
	N = nextpow2(len(x))
	N = 2**N
	x = np.append(x,np.zeros(N-len(x)))
	if N <= 2:
		return slow_dft(x)
	else:
		X_even = dit_fft(x[::2])
		X_odd = dit_fft(x[1::2])
		W_Nk = np.exp(-2j * np.pi * np.arange(N//2) / N) # W_N^k
		return np.concatenate([X_even + (W_Nk*X_odd), X_even - (W_Nk*X_odd)])

def slow_idft(x):
	x = np.asarray(x, dtype=complex)
	N = x.shape[0]
	n = np.arange(N)
	k = n.reshape((N, 1))
	M = np.exp(2j * np.pi * k * n / N)
	return np.dot(M, x)

def dit_ifft(x):
	N = nextpow2(len(x))
	N = 2**N
	def dit(x):
		x = np.asarray(x, dtype=complex)
		N = nextpow2(len(x))
		N = 2**N
		x = np.append(x,np.zeros(N-len(x)))
		if N <= 2:
			return slow_idft(x)
		else:
			X_even = dit(x[::2])
			X_odd = dit(x[1::2])
			W_Nk = np.exp(2j * np.pi * np.arange(N//2) / N)
			return np.concatenate([X_even + (W_Nk*X_odd), X_even - (W_Nk*X_odd)])

	return (1/N)*dit(x)

# If h(t) is ... 		then, ...
# real				H(-f) = H*(f) # implemented in powspec
# imag				H(-f) = -H*(f) # implemented in powspec
# even				H(-f) = H(f)
# odd				H(-f) = -H(f)
# real and even			H(f) = real and even
# real and odd			H(f) = imag and odd
# imag and even			H(f) = imag and even
# imag and odd			H(f) = real and odd

def powspec(x,period=None,sym=True): # if sym is False, then the frequency range will be 0 to fs rather than -fs/2 to fs/2
	x = np.array(x, dtype = complex)
	real = np.isreal(x).all()
	imag = np.isreal(x*(-1j)).all()
	p = dit_fft(x)
	if(period is None):
		period = len(p)
		rate = 1.0/period
	else:
		rate = 1.0/period

	if(sym):
		p = p[:len(p)//2]
		if(real):	
			p = np.append(np.conjugate(p[1:][::-1]),p)
		if(imag):	
			p = np.append(-np.conjugate(p[1:][::-1]),p)
		p = np.abs(p)**2
		f = np.linspace(-rate/2, rate/2, len(p))
	else:
		p = np.abs(p)**2
		f = np.linspace(0,rate,len(p))
	return f,p

def get_fp(period,k=6,return_ratio = False):
	N = 2**k
	rate = 1/period
	t = np.linspace(0, (N-1)*period, N) # N
	x = np.cos(2*np.pi*3*t) 
	p = dit_fft(x)[:len(t)//2]
	#p = slow_dft(x)[:len(t)//2]
	p = np.append(np.conjugate(p[1:][::-1]),p)
	p = np.abs(p)**2
	p_sorted = np.sort(p)
	f = np.linspace(-rate/2, rate/2, len(p))	
	if(return_ratio):
		ratio = 100*(p_sorted[-3]/p_sorted[-1])
		return f, p, ratio
	else:
		return f, p

def save_plot(x,y):
	plt.stem(x,y)
	plt.xlabel(r"$\omega$")
	plt.ylabel(r'$|\hat{X}(\omega)|^2$')
	plt.title("Power Spectrum")
	plt.savefig("power_spectrum.png",bbox_inches="tight")

def get_optimal(k_arr):
	ratio = []
	for k in k_arr:
		_,_,a = get_fp(0.1,k,True)
		ratio.append(a)
	idx = np.argmin(ratio)
	f, p = get_fp(0.1,k_arr[idx])
	save_plot(f,p)
	return k_arr[idx]

def linear_convolution(x,h):
	N = len(x)+len(h)-1
	x = np.append(x,np.zeros(len(h)-1))
	h = np.append(h,np.zeros(N-len(h)))
	y = np.zeros(N)
	x = x[::-1]
	for i in range(N):
		tmph = np.append(h[i+1:], h[:i+1])
		y[i] = np.sum(x*tmph)
	return y

def cross_corr(x1,x2): # circular cross correlation of two discrete signals
	if(len(x1) == len(x2)):
		N = len(x1)
		y = np.zeros(N)
		for i in range(N):
			y[i] = np.dot(np.conj(x1),x2)
			x1 = np.append(x1[1:],x1[:1])
		return y
	else:
		print("Error: len(x1) != len(x2)")
		return None

def auto_corr(x,t,T=None):
	if(T is None):
		return cross_corr(x,x)
	if(len(x) == len(t)):
		if(T is not None):
			idx = np.where(t<=T)
			x1 = x[idx]
			return cross_corr(x1,x1)
	else:
		print("Error: len(x) != len(t)")
		return None

def dit_2dfft(x):
	x = np.array([dit_fft(x[i]) for i in range(x.shape[0])])
	x = np.array([dit_fft(x[:,i]) for i in range(x.shape[1])])
	return x

def slow_dft2d(x):
	x = np.array([slow_dft(x[i]) for i in range(x.shape[0])])
	x = np.array([slow_dft(x[:,i]) for i in range(x.shape[1])])
	return x

def circular_shift(x,n):
	if(len(x) == x.size):
		return np.append(x[n:],x[:n])
	else:
		return np.concatenate((x[n:],x[:n]))	

def fftshift(x,n):
	x = circular_shift(x,n)
	factor = np.arange(len(x))
	return x*np.exp((1j*2*n*np.pi*factor)/len(x))

def fftshift2d(x,n=None,m=None):
	if(n is None):
		if(len(x)%2 == 0):
			n = len(x)//2
		else:
			n = len(x)//2 + 1
	if(m is None):
		if(len(x)%2 == 0):
			m = len(x)//2
		else:
			m = len(x)//2 + 1			
	x = circular_shift(x,n)
	print(x.shape)
	x = circular_shift(x.T,m).T
	factor1 = np.arange(x.shape[0])
	factor1 = np.exp((-1j*2*n*np.pi*factor1)/len(x))
	factor2 = np.arange(x.shape[1])
	factor1 = np.exp((-1j*2*n*np.pi*factor2)/len(x))
	factor = np.outer(factor1,factor2)
	return x*factor

'''
x = np.ones((5,5))
#x = np.ones(8)
#X = dit_2dfft(x)
X = slow_dft2d(x)
#X = dit_fft(x)
X = fftshift2d(X)
#X = fftshift(X,4)
X = np.abs(X)**2
plt.imshow(X,extent=[-len(x)//2,len(x)//2,-len(x)//2,len(x)//2])
plt.colorbar()
plt.show()
'''
'''
eps = 0
fm = 4.0
fs = 2*fm + eps
period = 1.0/fs
N = 4
#t = np.arange(0,1/fm,period)
t = np.linspace(0, (N-1)*period, N) # N
x = np.cos(2*np.pi*fm*t)  
f,p = powspec(x,period=period)
print(f,p)
plt.plot(f,p)
plt.show()
#print(f)
x = lambda n: np.cos(2*np.pi*fm*(n/N))
X = lambda k: np.sum([x(n)*np.exp((-1j*2*np.pi*k)/N) for n in range(N)])
print(X(fm))
'''
