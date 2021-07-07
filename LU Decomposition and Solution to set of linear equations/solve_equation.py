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
ap.add_argument("-s", "--save", required=False, help="Save as txt or npy file?", default="txt")
ap.add_argument("-a", "--Amatrix", required=True, help="Input file containing the matrix in numpy readable format")
ap.add_argument("-b", "--bmatrix", required=True, help="Input file containing the matrix in numpy readable format")

args = vars(ap.parse_args())

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

X = la.back_substitution(PA,b)
z = la.back_substitution(L,b)
x = la.back_substitution(U,z)
if(save == "npy"):
	np.save("{}_{}_x.npy".format(Amatrix[:-4]),bmatrix[:-4]),x)
	np.save("{}_{}_z.npy".format(Amatrix[:-4]),bmatrix[:-4]),z)
	np.save("{}_{}_P.npy".format(Amatrix[:-4]),bmatrix[:-4]),P)
	np.save("{}_{}_L.npy".format(Amatrix[:-4]),bmatrix[:-4]),L)
	np.save("{}_{}_U.npy".format(Amatrix[:-4]),bmatrix[:-4]),U)

else:
	np.save("{}_{}_x.txt".format(Amatrix[:-4]),bmatrix[:-4]),x)
	np.save("{}_{}_z.txt".format(Amatrix[:-4]),bmatrix[:-4]),z)
	np.save("{}_{}_P.txt".format(Amatrix[:-4]),bmatrix[:-4]),P)
	np.save("{}_{}_L.txt".format(Amatrix[:-4]),bmatrix[:-4]),L)
	np.save("{}_{}_U.txt".format(Amatrix[:-4]),bmatrix[:-4]),U)

##################################
#######--- End of code ---########
##################################
