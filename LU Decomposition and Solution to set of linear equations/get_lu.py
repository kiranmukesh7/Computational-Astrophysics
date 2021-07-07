##################################
###--- Importing Libraries ---####
##################################

import numpy as np
import custom_linalg as la
import argparse
import sys

##################################
#######--- User Inputs ---########
##################################

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--verbose", required=False, help="Display all the matrics?", default=0)
ap.add_argument("-s", "--save", required=False, help="Save as txt or npy file?", default="txt")
ap.add_argument("-m", "--matrix", required=True, help="Input file containing the matrix in numpy readable format")

args = vars(ap.parse_args())

verbose = int(args["verbose"])
matrix = args["matrix"]
save = args["save"]
if(matrix[-3:] == "txt"):
	A = np.loadtxt(matrix)
elif(matrix[-3:] == "npy"):
	A = np.load(matrix)
else:
	print("Please input file with extension txt or npy, in a numpy readable format.")
	sys.exit()

##################################
#--- Compute LU Decomposition ---#
##################################

P,L,U = la.lu_decomposition(A)

##################################
#--- Display and Save Results ---#
##################################

if(verbose == 1):
	print("Input Matrix: \n",A)
	print("Pivot Matrix: \n", P)
	print("Pivoted Matrix: \n",la.matmul(P,A))
	print("L Matrix: \n",L)
	print("U Matrix: \n",U)
	print("Verification using numpy allclose function: ", np.allclose(la.matmul(P,A),la.matmul(L,U)))
if(save == "npy"):
	np.save("{}_P.npy".format(matrix[:-4]),P)
	np.save("{}_L.npy".format(matrix[:-4]),L)
	np.save("{}_U.npy".format(matrix[:-4]),U)
else:
	np.savetxt("{}_P.txt".format(matrix[:-4]),P)
	np.savetxt("{}_L.txt".format(matrix[:-4]),L)
	np.savetxt("{}_U.txt".format(matrix[:-4]),U)

##################################
#######--- End of code ---########
##################################
