#ifndef LAB3_CUDA_H
#define LAB3_CUDA_H

#include <math.h>
#include <malloc.h>
#include <stdio.h>


/*
To be implemented
Note:
    Since PCA of matrix D can be computed by taking SVD of D or D_T, we will allow you to 
    make the choice. If you compute SVD of D then, U is MxM matrix, SIGMA is MxN matrix and 
    V_T is NxN matrix. On the other hand if you compute SVD of D_T, U is NxN, SIGMA is NxM, 
    and V_T is MxM matrices. Note that dimensions of SIGMA are same as that of matrix being 
    decomposed. For correctness checking, we need to know the matrix you chose for SVD. We 
    will use the dimensions of SIGMA for this purpose since the dimensions of SIGMA is same 
    as the matrix being decomposed. Variable SIGMAm and SIGMAn (to be computed by you) are 
    number of rows and columns in SIGMA as well as in the matrix used for SVD. We will check 
    the correctness of SVD accordingly assuming the dimensions of U, SIGMA and V_T as per 
    these variables. Since only N digonal elements in SIGMA are non-zero, it should be returned 
    as 1D vector of N elements (no need to store zeros in SIGMA).

Arguments:
    M : number of rows (samples) in input matrix D (input)
    N : number of columns (features) in input matrix D (input)
    D : 1D Array of M x N input matrix in row-major, (input)
        #elements in D is (M * N)
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
    U : 1D array of N x N (or M x M) real matrix in row-major (to be computed)
        --------------------------------------------------------------------------------------
        | U[0][0] | U[0][1] | ... | U[0][N-1] | U[1][0] | ... | U[1][N-1] | ... | U[N-1][N-1] |
        --------------------------------------------------------------------------------------
    SIGMA : 1D array of N x M (or M x N) diagonal matrix of positive real numbers (to be computed)
        format: consists only digonal elements
        #elements in SIGMA is N (digonals will be N in both cases)
        -------------------------------------------------------------------
        | SIGMA[0][0] | SIGMA[1][1] | SIGMA[2][2] | ... | SIGMA[N-1][N-1] |
        -------------------------------------------------------------------
    V_T : 1D array of M x M (or N x N) real matrix in row-major (to be computed)
        -------------------------------------------------------------------------------
        | V_T[0][0] | V_T[0][1] | ... | V_T[0][M-1] | V_T[1][0] | ... | V_T[M-1][M-1] |
        -------------------------------------------------------------------------------
    SIGMAm: #rows in SIGMA, to be decided as per the dimentions of matrix used for SVD
        (to be computed)
    SIGMAn: #columns in SIGMA, to be decided as per the dimentions of matrix used for SVD
        (to be computed)
    D_HAT : 1D array of reduced M x K real matrix in row-major (to be computed)
        -----------------------------------------------------------------------------------------
        | D_HAT[0][0] | D_HAT[0][1] | ... | D_HAT[0][K-1] | D_HAT[1][0] | ... | D_HAT[M-1][K-1] |
        -----------------------------------------------------------------------------------------
    K : number of columns (features) in reduced matrix (to be computed)
    retention : percentage of inpdormation to be retained by PCA
        retention = 90 means 90% of information should be retained
*/
void SVD_and_PCA (
    int M, 
    int N, 
    double* D, 
    double** U, 
    double** SIGMA, 
    double** V_T, 
    int *SIGMAm, 
    int *SIGMAn, 
    double** D_HAT, 
    int *K, 
    int retention);

#endif
