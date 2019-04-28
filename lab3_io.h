#ifndef LAB3_IO_H
#define LAB3_IO_H

#include <stdio.h>
#include <malloc.h>

/*
	M : number of rows (samples) in input matrix D
    N : number of columns (features) in input matrix D
    D : 1D Array of M x N input matrix in row-major, 
        #elements in D is (M * N)
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
*/
void read_matrix (const char* input_filename, int* M, int *N, double** D);

/*
check correctess of Singular Vector Decomposition
Arguments:
    M : number of rows (samples) in input matrix D
    N : number of columns (features) in input matrix D
    D : 1D Array of M x N input matrix in row-major, 
        #elements in D is (M * N)
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
    U : 1D array of N x n real matrix (computed by SVD) in row-major
        --------------------------------------------------------------------------------------
        | U[0][0] | U[0][1] | ... | U[0][N-1] | U[1][0] | ... | U[1][N-1] | ... | U[N-1][N-1] |
        --------------------------------------------------------------------------------------
    SIGMA : 1D array of N x M (or M x N) diagonal matrix of positive real numbers (computed by SVD),
        format: consists only digonal elements
        #elements in SIGMA is N (digonals will be N in both cases)
        -------------------------------------------------------------------
        | SIGMA[0][0] | SIGMA[1][1] | SIGMA[2][2] | ... | SIGMA[N-1][N-1] |
        -------------------------------------------------------------------
    V_T : 1D array of M x M real matrix (computed by SVD) in row-major
        --------------------------------------------------------------------------------
        | V_T[0][0] | V_T[0][1] | ... | V_T[0][M-1] | V_T[1][0] | ...  | V_T[M-1][M-1] |
        --------------------------------------------------------------------------------
    SIGMAm: #rows in SIGMA, to be decided as per the dimentions of matrix used for SVD
    SIGMAn: #columns in SIGMA, to be decided as per the dimentions of matrix used for SVD
    K : number of coulmns (features) in reduced matrix D_HAT
    D_HAT : reduced matrix (computed by PCA) in row-major
        -------------------------------------------------------------------------------------
        | D_HAT[0][0] | D_HAT[0][1] | ... | D_HAT[0][K-1] | D_HAT[1][0] | ... | D[M-1][K-1] |
        -------------------------------------------------------------------------------------
    computation_time : Time elapsed in computing SVD and PCA
*/

void write_result (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
        int SIGMAm, 
        int SIGMAn, 
		int K, 
		double* D_HAT,
		double computation_time);

/* 
Function to check the format of output code.
You can call it from main to check if the dimensions of 
output matches with our expected dimensions.
This is a dummy function. It is not the function that 
will be used for evaluation.
*/
void format_checker (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
		int SIGMAm, 
		int SIGMAn, 
		int K, 
		double* D_HAT);

#endif