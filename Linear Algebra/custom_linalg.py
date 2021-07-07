import numpy as np

def get_row_reduced_form(A,b):
	"""Get row reduced form of input matrx"""
	A = np.column_stack((A,b)).astype(float)
	del b
	argmax = np.argmax(A.T[0])
	tmp = A[argmax]
	A = np.delete(A,argmax,0)
	A = np.row_stack((tmp,A))
	idx = np.where(~A.any(axis=1)) # where the A matrix row is identically zeros -> case of infinte solutions --> A.any(axis=1) checks if there is any non-zero element in the row (because axis = 1), if axis= 0, checks for rows with non-zero column
	A = np.delete(A,idx,0)
	m,n = A.shape
	for i in range(m):
		for j in range(m):
			if(j>i):
				A[j] -= A[i]*(A[j][i]/A[i][i]) # <--
				A[j][0] = 0 # trying to avoid round-off error
			else:
				continue
	A = np.row_stack((A, np.zeros((len(idx[0]),A.shape[1] )) )) 
	return A[:,:-1],A[:,-1],len(idx[0])

def back_substitution_with_row_reduction(A,b, return_matrices = False, verbose=False):
	"""Returns value of the variables computed by back substitution algorithm"""
	A,b,n_z = get_row_reduced_form(A,b)
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

def back_substitution(A,b, return_matrices = False, verbose=False):
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


def forward_substitution(A,b, return_matrices = False, verbose=False):
	"""Returns value of the variables computed by back substitution algorithm"""
	b = b.astype(float)
	m,n = A.shape
	p = min(m,n)
	for i in range(n):
		for j in range(i):
			b[i] -= A[i][j]*b[j]
		b[i] = b[i]/A[i][i]
	return b[:n]

def solve_lin_equation(A,b,return_matrices=False): # disadvantage --> more time and memory needed 
	"""Another function that returns the solution of linear equation"""
	A,b,n_z = get_row_reduced_form(A,b)
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

def get_pivot_matrix(M):
	"""Returns the pivot matrix required to rearrange the rows of input matrix into the form having diagonal elements as the element of maximum value in each column"""
	m,n = M.shape
	P_dim = min(m,n)
	id_mat = np.array([[float(i ==j) for i in range(m)] for j in range(m)])
	for j in range(P_dim):
		row = np.argmax(M.T[j])
		if j != row:
			tmp = np.copy(id_mat[row])
			id_mat[row] = id_mat[j]
			id_mat[j] = tmp
	return id_mat

def matmul(M, N):
	"""Returns product of matrix multiplication of input matrices"""
	if(M.shape[1] == N.shape[0]):
		return np.array([[sum(el_m * el_n for el_m, el_n in zip(row_m, col_n)) for col_n in N.T] for row_m in M])
	else:
		print("Matrix multiplication not possible with input matrics in the given order")
		return 

def lu_decomposition(A):
	"""Returns the pivot, lower and upper triangular matrices corresponding to the LU decompostion of the input matrx"""
	m,n = A.shape
	L = np.zeros((m,min(m,n))) 
	U = np.zeros((min(m,n),n)) 
	P = get_pivot_matrix(A)
	PA = matmul(P, A)
	for k in range(min(m,n)):
		U[k][k] = 1
		for i in range(k,m):
			s1 = sum(L[i][j] * U[j][k] for j in range(k))
			L[i][k] = PA[i][k] - s1
		for j in range(k+1, n):
			s2 = sum(L[k][i] * U[i][j] for i in range(k))
			U[k][j] = (PA[k][j] - s2) / L[k][k]
	return P,L,U

def levicivita(*values):
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

def cross_prod(x,y):
    if(len(x) != len(y)):
        return None # Not possible to compute cross product
    n = len(x)
    z = np.zeros(n)
    z = [sum([levicivita(i,j,k)*x[j]*y[k] for k in range(n) for j in range(n)]) for i in range(n)]
    return z

def inverse(A):
    P,L,U = lu_decomposition(A)
    b = np.identity(len(A))
    b = matmul(P,b)
    z = forward_substitution(L,b.T)
    b = back_substitution(U,z)
#    b = matmul(P,b.T)
    return b

