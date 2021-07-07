import numpy as np
import custom_linalg as linalg

# Unique Solution
A = np.array([[2,3,-4],[1,5,-1],[3,7,-3]])
b = np.array([12,12,20])
x,A = linalg.back_substitution(A,b, return_matrices = True)
print(x)
print(A)

# No Solution
A = np.array([[2,3],[2,3]])
b = np.array([10,12])
x,A = linalg.back_substitution(A,b, return_matrices = True)
print(x)
print(A)

# Infinitely Many Solutions
A = np.array([[1,0,0,-3],[0,1,0,2],[0,0,1,1],[0,0,0,0],[0,0,0,0]])
b = np.array([4,1,2,0,0])
x,A = linalg.back_substitution(A,b, return_matrices = True)
print(x)
print(A)

#x = linalg.solve_lin_equation(A,b,False)
#print(x)
