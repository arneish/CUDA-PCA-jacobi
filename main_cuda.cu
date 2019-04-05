#include "lab3_io.h"
#include "lab3_cuda.h"

#include <stdlib.h>
#include <omp.h>

/*
	Arguments:
		arg1: input filename (consist M, N and D)
		arg2: retention (percentage of information to be retained by PCA)
*/

int main(int argc, char const *argv[])
{
	if (argc < 3){
		printf("\nLess Arguments\n");
		return 0;
	}

	if (argc > 3){
		printf("\nTOO many Arguments\n");
		return 0;
	}

	//---------------------------------------------------------------------
	int M;			//no of rows (samples) in input matrix D (input)
	int N;			//no of columns (features) in input matrix D (input)
	double* D;		//1D array of M x N matrix to be reduced (input)
	double* U;		//1D array of N x N matrix U (to be computed by SVD)
	double* SIGMA;	//1D array of N x M diagonal matrix SIGMA (to be computed by SVD)
	double* V_T;		//1D array of M x M matrix V_T (to be computed by SVD)
	int K;			//no of coulmns (features) in reduced matrix D_HAT (to be computed by PCA)
	double *D_HAT;	//1D array of M x K reduced matrix (to be computed by PCA)
	int retention;	//percentage of information to be retained by PCA (command line input)
	//---------------------------------------------------------------------

	retention = atoi(argv[2]);	//retention = 90 means 90% of information should be retained

	float computation_time;

	/*
		-- Pre-defined function --
		reads matrix and its dimentions from input file and creats array D
	    #elements in D is M * N
        format - 
        --------------------------------------------------------------------------------------
        | D[0][0] | D[0][1] | ... | D[0][N-1] | D[1][0] | ... | D[1][N-1] | ... | D[M-1][N-1] |
        --------------------------------------------------------------------------------------
	*/
	read_matrix (argv[1], &M, &N, &D);

	printf("M, N: %d %d\n", M, N);

	U = (double*) malloc(sizeof(double) * N*N);
	SIGMA = (double*) malloc(sizeof(double) * N);
	V_T = (double*) malloc(sizeof(double) * M*M);

	cudaEvent_t start, stop;
	cudaEventRecord(start);
	
	// /*
	// 	*****************************************************
	// 		TODO -- You must implement this function
	// 	*****************************************************
	// */
	SVD_and_PCA(M, N, D, &U, &SIGMA, &V_T, &D_HAT, &K, retention);

	cudaEventRecord(stop);
	cudaEventElapsedTime(&computation_time, start, stop);
	
	/*
		--Pre-defined functions --
		checks for correctness of results computed by SVD and PCA
		and outputs the results
	*/
	write_result(M, N, D, U, SIGMA, V_T, K, D_HAT, computation_time);

	return 0;
}
