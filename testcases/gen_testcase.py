#!/usr/bin/python3

#########################################################################
# Generate M x N matrix of real numbers and store                       #
# the the matrix in file named 'testcase_<M>_<N>'                       #
# Parameters:                                                           #
#   M               :no of rows (samples) in matrix                     #
#   N               :no of coulmns (features) in matrix                 #
#   lrange, urange  :range of matrix elements ie                        #
#                       forall  0<=i<M, 0<=j<N                          #
#                       lrange <= matrix[i][j] <= urange                #
# Format of output file:                                                #
#   -----------------------------------------------------------------   #
#   | M N                                                               #
#	| D[0][0] D[0][1] ... D[0][N-1] D[1][0] ... D[M-1][N-1]             #
#   -----------------------------------------------------------------   #
#########################################################################


from random import uniform
from sklearn.preprocessing import StandardScaler

M = 1000            # number of rows (samples) in input matrix D
N = 300             # number of columns (features) in input matrix
lrange = -100000    # lrange <= element of matrix
urange = 100000     # element of matrix <= urange

# generate the matrix
D = []
for i in range(M):
    temp = []
    for j in range(N):
        temp.append(uniform(lrange, urange))
    D.append(temp)

# standardize
X_std = StandardScaler().fit_transform(D)

filename = 'testcase_' + str(M) + '_' + str(N)     #output filename
file = open(filename, 'w')

# write size of matrix in first line of file
file.write(str(M) + ' ' +str(N) + '\n')

# write space separated matrix elements
for i in range(M):
    for j in range(N):
        file.write('%.7f ' %(X_std[i][j]))

file.close()
