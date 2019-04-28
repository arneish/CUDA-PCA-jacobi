#include "lab3_io.h"

void read_matrix (const char* input_filename, int* M, int* N, double** D){
	FILE *fin = fopen(input_filename, "r");
	int i;

	fscanf(fin, "%d%d", M, N);
	
	int num_elements = (*M) * (*N);
	*D = (double*) malloc(sizeof(double)*(num_elements));
	
	for (i = 0; i < num_elements; i++){
		fscanf(fin, "%lf", (*D + i));
	}
	fclose(fin);
}

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
		double computation_time){
	// Will contain output code
}

void format_checker (int M, 
		int N, 
		double* D, 
		double* U, 
		double* SIGMA, 
		double* V_T,
		int SIGMAm, 
		int SIGMAn, 
		int K, 
		double* D_HAT){
	printf("checking format\n");
	if (SIGMAm==M && SIGMAn==N) {
		printf("SVD of D:\n");
	}
	else if (SIGMAm==N && SIGMAn==M) {
		printf("SVD of D_T:\n");
	}

	printf("Matrix U:\n");
	for (int i = 0; i < SIGMAm; i++) {
		for (int j = 0; j < SIGMAm; j++) {
			printf("%.2lf\t", U[i*SIGMAm+j]);
		}
		printf("\n");
	}

	printf("Matrix SIGMA:\n");
	for (int i = 0; i < SIGMAm; i++) {
		for( int j = 0; j < SIGMAn; j++) {
			if (i == j)
				printf("%.2lf ", SIGMA[i]);
			else printf("0\t");
		}
		printf("\n");
	}

	printf("Matrix V_T:\n");
	for (int i = 0; i < SIGMAn; i++) {
		for (int j = 0; j < SIGMAn; j++) {
			printf("%lf\t", V_T[i*SIGMAn+j]);
		}
		printf("\n");
	}

	printf("K = %d\n", K);

	printf("Matrix D_HAT:\n");
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < K; j++) {
			printf("%lf\t", D_HAT[i*K+j]);
		}
		printf("\n");
	}
}

