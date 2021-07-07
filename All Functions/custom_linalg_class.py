import numpy as np
import itertools

class linalg:

	def __init__(self,A=None):
		self.A = None

	def get_row_reduced_form(self,A,b=None):
		"""Get row reduced form of input matrx"""
		if(b is not None):
			A = np.column_stack((A,b)).astype(float)
			del b
		else:
			A = np.array(A).astype(float)
		argmax = np.argmax(A.T[0])
		tmp = A[argmax]
		A = np.delete(A,argmax,0)
		A = np.row_stack((tmp,A)).astype(float)
		idx = np.where(~A.any(axis=1)) # where the A matrix row is identically zeros -> case of infinte solutions --> A.any(axis=1) checks if there is any non-zero element in the row (because axis = 1), if axis= 0, checks for rows with non-zero column
		A = np.delete(A,idx,0)
		m,n = A.shape
		for i in range(m):
			for j in range(i+1,m):
				scale = (A[j][i]/A[i][i])
				for k in range(n):
					A[j][k] -= A[i][k]*scale

		A = np.row_stack((A, np.zeros((len(idx[0]),A.shape[1] )) )) 
		# forcing the lower diagonal elements to 0 to avoid round-off error
		for i in range(min(m,n)):
			for j in range(i+1):
				A[i][j] = 0

		if(b is not None):
			return A[:,:-1],A[:,-1],len(idx[0])
		else:
			return A, np.where(argmax!=0,1,0) # row reduce matrix, number of swaps

	def back_substitution_with_row_reduction(self,A,b, return_matrices = False, verbose=False):
		"""Returns value of the variables computed by back substitution algorithm"""
		A,b,n_z = self.get_row_reduced_form(A,b)
		if(0. in np.diag(A)):
			idx = np.where(np.diag(A) == 0.0)
			if(b[idx] != 0):
				print("The system of equations has NO SOLUTION!")
				return None, A
			if(b[idx] == 0):
				print("The system of equations has INFINITELY MANY SOLUTIONS!")
				print("Returning one of the possible solutions...")
				q = True
		else:
			if(verbose):
				print("The system of equations has a UNIQUE SOLUTION!")
		b = b.astype(float)
		m,n = A.shape
		if(b.size < n):
			print("The system of equations has INFINITELY MANY SOLUTIONS!")
			print("Returning one of the possible solutions...")
			b = np.append(b,np.zeros(n-b.size))
		n -= n_z
		p = min(m,n)
		for i in reversed(range(p)):
			for j in reversed(range(i+1,n)):
				b[i] -= A[i][j]*b[j]
			b[i] = b[i]/A[i][i]
		if(return_matrices):
			return b[:A.shape[1]], A
		else:
			return b[:A.shape[1]]

	def back_substitution(self,A,b, return_matrices = False, verbose=False):
		"""Returns value of the variables computed by back substitution algorithm"""
		if(0. in np.diag(A)):
			idx = np.where(np.diag(A) == 0.0)
			if(b[idx] != 0):
				print("The system of equations has NO SOLUTION!")
				return None, A
			if(b[idx] == 0):
				print("The system of equations has INFINITELY MANY SOLUTIONS!")
				print("Returning one of the possible solutions...")
				q = True
		else:
			if(verbose):
				print("The system of equations has a UNIQUE SOLUTION!")
		b = b.astype(float)
		m,n = A.shape
		if(b.size < n):
			print("The system of equations has INFINITELY MANY SOLUTIONS!")
			print("Returning one of the possible solutions...")
			b = np.append(b,np.zeros(n-b.size))
		p = min(m,n)
		for i in reversed(range(p)):
			for j in reversed(range(i+1,n)):
				b[i] -= A[i][j]*b[j]
			b[i] = b[i]/A[i][i]
		if(return_matrices):
			return b[:A.shape[1]], A
		else:
			return b[:A.shape[1]]


	def forward_substitution(self,A,b, return_matrices = False, verbose=False):
		"""Returns value of the variables computed by back substitution algorithm"""
		b = b.astype(float)
		m,n = A.shape
		p = min(m,n)
		for i in range(n):
			for j in range(i):
				b[i] -= A[i][j]*b[j]
			b[i] = b[i]/A[i][i]
		return b[:n]

	def solve_lin_equation(self,A,b,return_matrices=False): # disadvantage --> more time and memory needed 
		"""Another function that returns the solution of linear equation"""
		A,b,n_z = self.get_row_reduced_form(A,b)
		q = False
		if(0. in np.diag(A)):
			idx = np.where(np.diag(A) == 0.0)
			if(b[idx] != 0):
				print("The system of equations has NO SOLUTION!")
				return None, A, b
			if(b[idx] == 0):
				print("The system of equations has INFINITELY MANY SOLUTIONS!")
				q = True
		else:
			print("The system of equations has a UNIQUE SOLUTION!")

		n = A.shape[1]-n_z
		x = np.zeros(n)
		x = np.append(x,np.ones(A.shape[1]-n))

		for i in reversed(range(n)):
			x[i] = (b[i] - np.dot(A[i],x))/A[i][i]

		if(return_matrices):
			return x,A,b
		if(not return_matrices):
			return x

	def get_pivot_matrix(self,M):
		"""Returns the pivot matrix required to rearrange the rows of input matrix into the form having diagonal elements as the element of maximum value in each column"""
		m,n = M.shape
		P_dim = min(m,n)
		n_swaps = 0
		id_mat = np.array([[float(i ==j) for i in range(m)] for j in range(m)])
		for j in range(P_dim):
			row = np.argmax(M.T[j])
			if j != row:
				tmp = np.copy(id_mat[row])
				id_mat[row] = id_mat[j]
				id_mat[j] = tmp
				n_swaps += 1
		return id_mat, n_swaps

	def matmul(self,M, N):
		"""Returns product of matrix multiplication of input matrices"""
		if(M.shape[1] == N.shape[0]):
			return np.array([[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N.T] for row_m in M])
		else:
			print("Matrix multiplication not possible with input matrics in the given order")
			return 

	def lu_decomposition(self,A):
		"""Returns the pivot, lower and upper triangular matrices corresponding to the LU decompostion of the input matrx"""
		m,n = A.shape
		L = np.zeros((m,min(m,n))) 
		U = np.zeros((min(m,n),n)) 
		P,n_swaps = self.get_pivot_matrix(A)
		PA = self.matmul(P, A)
		for k in range(min(m,n)):
			U[k][k] = 1
			for i in range(k,m):
				s1 = sum(L[i][j] * U[j][k] for j in range(k))
				L[i][k] = PA[i][k] - s1
			for j in range(k+1, n):
				s2 = sum(L[k][i] * U[i][j] for i in range(k))
				U[k][j] = (PA[k][j] - s2) / L[k][k]
		return P,L,U

	def levicivita(self,*values):
		if(type(*values) is tuple):
			arr = list(*values)
		else:
			arr = [*values]
		unique = np.unique(arr)
		if(max(arr) > len(arr)): # check for inconsistencies
			return None

		for i in unique:
			if(arr.count(i) != 1):
				return 0 # return 0 if there are repeatitions
		arr = np.array(arr)
		idx = np.where(arr == np.amin(arr))[0][0]
		arr = np.append(arr[idx:], arr[:idx])
		if(np.allclose(np.sort(arr),arr)):
			return 1
		else:
			return -1

	def cross_prod(self,x,y):
		if(len(x) != len(y)):
			return None # Not possible to compute cross product
		n = len(x)
		z = np.zeros(n)
		z = [sum([self.levicivita(i,j,k)*x[j]*y[k] for k in range(n) for j in range(n)]) for i in range(n)]
		return z

	def det(self,A, gauss=False):
		n = len(A)
		x = list(x for x in itertools.product(range(n), repeat=n))
		s = sum([self.levicivita(i)*np.prod([row[j] for row,j in zip(A,i)]) for i in x])
		return s

	def inverse(self,A,method = "LU"): # or back_sub
		if(method == "back_sub"):
			b = np.identity(len(A))
			for i in range(len(b)):
			b[i] = self.back_substitution_with_row_reduction(A,b[i])
			return b.T
		if(method == "LU"):
			P,L,U = self.lu_decomposition(A)
			b = np.identity(len(A))
			z = self.forward_substitution(L,b)
			b = self.back_substitution(U,z)
			b = self.matmul(b,P)
			return b


