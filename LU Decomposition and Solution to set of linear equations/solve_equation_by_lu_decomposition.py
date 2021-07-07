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
ap.add_argument("-v", "--verbose", required=False, help="Display all the matrics?", default=1)
ap.add_argument("-s", "--save", required=False, help="Save as txt or npy file?", default="txt")
ap.add_argument("-a", "--Amatrix", required=True, help="Input file containing the matrix in numpy readable format")
ap.add_argument("-b", "--bmatrix", required=True, help="Input file containing the matrix in numpy readable format")

args = vars(ap.parse_args())
verbose = int(args["verbose"])
Amatrix = args["Amatrix"]
bmatrix = args["bmatrix"]
save = args["save"]
if(Amatrix[-3:] == "txt"):
	A = np.loadtxt(Amatrix)
elif(Amatrix[-3:] == "npy"):
	A = np.load(Amatrix)
else:
	print("Please input A matrix file with extension txt or npy, in a numpy readable format.")
	sys.exit()
if(bmatrix[-3:] == "txt"):
	b = np.loadtxt(bmatrix)
elif(bmatrix[-3:] == "npy"):
	b = np.load(bmatrix)
else:
	print("Please input b matrix file with extension txt or npy, in a numpy readable format.")
	sys.exit()
A = A.astype(float)
b = b.astype(float)
##################################
#--- Compute LU Decomposition ---#
##################################

P,L,U = la.lu_decomposition(A)
b = la.matmul(P,b.reshape(-1,1))
b = b.squeeze()
PA = la.matmul(P,A)

####################################
#--- Solve by Back-substitution ---#
####################################

z = la.forward_substitution(L,b)
x = la.back_substitution(U,z)

#######################################
#--- Saving and Displaying Results ---#
#######################################

if(save == "npy"):
	np.save("{}_{}_x.npy".format(Amatrix[:-4],bmatrix[:-4]),x)
	np.save("{}_{}_z.npy".format(Amatrix[:-4],bmatrix[:-4]),z)
	np.save("{}_{}_P.npy".format(Amatrix[:-4],bmatrix[:-4]),P)
	np.save("{}_{}_L.npy".format(Amatrix[:-4],bmatrix[:-4]),L)
	np.save("{}_{}_U.npy".format(Amatrix[:-4],bmatrix[:-4]),U)

else:
	np.savetxt("{}_{}_x.txt".format(Amatrix[:-4],bmatrix[:-4]),x)
	np.savetxt("{}_{}_z.txt".format(Amatrix[:-4],bmatrix[:-4]),z)
	np.savetxt("{}_{}_P.txt".format(Amatrix[:-4],bmatrix[:-4]),P)
	np.savetxt("{}_{}_L.txt".format(Amatrix[:-4],bmatrix[:-4]),L)
	np.savetxt("{}_{}_U.txt".format(Amatrix[:-4],bmatrix[:-4]),U)

if(verbose == 1):
	print("Input Matrix: \n",A)
	print("Pivot Matrix: \n", P)
	print("Pivoted Matrix: \n",la.matmul(P,A))
	print("L Matrix: \n",L)
	print("U Matrix: \n",U)
	print("z Matrix: \n",z)
	print("x Matrix: \n",x)

##################################
#######--- End of code ---########
##################################
