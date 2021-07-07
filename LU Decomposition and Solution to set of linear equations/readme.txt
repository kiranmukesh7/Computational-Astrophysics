The module "custom_linalg" contains the custom defined functions of linear algebra. The python script "get_lu.py" takes in input matrix saved in a txt file and computes its LU decomposition and saves the intermediate matrices - pivot matrix, lower and upper triangular matrix. 

How to use "get_lu.py":

1. The input matrix file can be specified using the "-m" key. It is necessary for the input matrix file to be in either ".npy" or ".txt" format (to ease loading the same as numpy array).

2. The intermediate matrices can be saved as a txt or npy file, as desired (using "-s" key).

3. The intermediate matrices can be output in the terminal by setting the "-v" key to 1.


