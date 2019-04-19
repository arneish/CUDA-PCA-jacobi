#include "lab3_cuda.h"
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define MAX_ITER 1000000
// /*
// 	*****************************************************
// 		TODO -- You must implement this function
// 	*****************************************************
// */
double *s_initialize_identity(int size)
{
    double *I = (double *)calloc(size * size, sizeof(double));
    for (int i = 0; i < size; i++)
        I[i * size + i] = 1.0;
    return I;
}

void s_transpose(double *M, int m, int n, double *M_T)
{
    int i, j;
    for (i = 0; i < m; i++)
    {
        for (j = 0; j < n; j++)
        {
            M_T[j * m + i] = M[i * n + j];
        }
    }
}

void s_multiply(double *M_1, int m1, int n1, double *M_2, int m2, int n2, double *result)
{
    double sum = 0.0;
    //compute M_2_T:
    double *M_2_T = (double *)malloc(sizeof(double) * n2 * m2);
    s_transpose(M_2, m2, n2, M_2_T);
    int i, j, k, temp1, temp2;
    for (i = 0; i < m1; i++)
    {
        temp1 = i * n1;
        for (j = 0; j < n2; j++)
        {
            sum = 0.0;
            temp2 = j * m2;
            for (k = 0; k < n1; k++)
            {
                sum += M_1[temp1 + k] * M_2_T[temp2 + k];
            }
            result[i * n2 + j] = sum;
        }
    }
    free(M_2_T);
}

double s_maxind(double *A, int size, int k)
{
    int m = k + 1;
    for (int i = k + 2; i < size; i++)
    {
        if (fabs(A[k * size + i]) > fabs(A[k * size + m]))
        {
            m = i;
        }
    }
    return m;
}

void s_update(int k, double t, double *e, bool *changed, int *state)
{
    double y = e[k]; e[k] = y + t;
    if (changed[k] && (y==e[k]))
    {
        changed[k] = false;
        (*state)--;
    }
    else if (!changed[k] && (y!=e[k]))
    {
        changed[k]=true;
        (*state)++;
    }
}

void s_rotate(int k, int l, int i, int j, double *A, int P, double c, double s)
{
    double k_l=c*A[k*P+l]-s*A[i*P+j];
    double i_j=s*A[k*P+l]+c*A[i*P+j];
    A[k*P+l]=k_l;
    A[i*P+j]=i_j;
}

double l2_matrix_diff_norm(double *E_, double *E, int M, int N)
{
    double sum = 0.0;
    for (int i=0; i<M; i++)
    {
        for (int j=0; j<N; j++)
            sum+=(E_[i*M+j]-E[i*M+j])*(E_[i*M+j]-E[i*M+j]);
    }
    return sqrt(sum);
}

double l2_diff_norm(double *e_, double *e, int len)
{
    double sum = 0.0;
    for (int i=0; i<len; i++)
    {
        sum+=(e_[i]-e[i])*(e_[i]-e[i]);
    }
    return sqrt(sum);
}

void s_compute_V(double **SIGMA, double *D_T, double **U, double **V_T, int N, int P)
{
    //V_T = INV-SIGMA * U_T * M
    double *INV_SIGMA = (double *)calloc(N * P, sizeof(double)); //|=NXP
    for (int i = 0; i < P; i++)
    {
        INV_SIGMA[i * P + i] = 1.0 / ((*SIGMA)[i]);
    }
    double *U_T = (double *)malloc(sizeof(double) * P * P);
    s_transpose(*U, P, P, U_T);
    //first, multiply INV-SIGMA X U_T |=(NXP)
    double *product = (double *)malloc(sizeof(double) * N * P);
    s_multiply(INV_SIGMA, N, P, U_T, P, P, product);
    //now, multiply product X D_T |=(NXN)
    s_multiply(product, N, P, D_T, P, N, *V_T);
    free(INV_SIGMA);
    free(U_T);
    free(product);
}

double s_matrix_similarity(double *M_1, int m, int n, double *M_2)
{
    double l2_diff = 0.0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            l2_diff += (M_1[i * n + j] - M_2[i * n + j]) * (M_1[i * n + j] - M_2[i * n + j]);
            //printf("%f ", l2_diff);
        }
    }
    l2_diff = sqrt(l2_diff);
    printf("L2-diff b/w D_T's: %f\n", l2_diff);
    return l2_diff;
}

void print_matrix(double *A, int M, int N, bool console)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (!console)
                fprintf(stderr, "%f ", A[i * N + j]);
            else
                printf("%f ", A[i * N + j]);
        }
        if (!console)
            fprintf(stderr, "\n");
        else
            printf("\n");
    }
}

double s_upper_triangular_sum(double *A, int P)
{
    double sum=0.0;
    for (int i=0; i<P; i++)
    {
        for (int j=i+1; j<P; j++)
        {
            sum+=A[i*P+j]*A[i*P+j];
        }
    }
    return sqrt(sum);
}


void SVD_and_PCA(int N,
                 int P,
                 double *D,
                 double **U,
                 double **SIGMA,
                 double **V_T,
                 double **D_HAT,
                 int *K,
                 int retention)
{
    /*1.Perform SVD for D_T*/
    // Get eigen-values & eigen-vectors for D_T*D
    double *D_T = (double *)malloc(sizeof(double) * P * N);
    s_transpose(D, N, P, D_T);
    double *A = (double *)calloc(P * P, sizeof(double));
    double *A_T = (double *)calloc(P * P, sizeof(double));
    s_multiply(D_T, P, N, D, N, P, A);

    //begin Jacobi eigenvalue algorithm:
    int state = P, num_iter = 0, m, k, l;              //m: pivot row identifier
    double p, y,d,r, c, s,t; //p: pivot element, c: cos, s: sin
    double *E = s_initialize_identity(P);                 //P*P
    double *E_ = (double *)malloc(sizeof(double)*P*P);
    double *e = (double *)malloc(sizeof(double) * P);     //init eigen-values array
    double *e_ = (double *)malloc(sizeof(double)*P);
    int *ind = (int *)malloc(sizeof(int) * P); //init maxindex array
    bool *changed = (bool *)malloc(sizeof(bool) * P);   //change in eigen_value[k]
    printf("printing A:\n");
    print_matrix(A, P, P, 1);
    for (int i = 0; i < P; i++)
    {
        ind[i] = s_maxind(A, P, i); //NOTE: undefined for last row
        e[i] = A[i * P + i];
        changed[i] = true;
        printf("%d, %d\n", i, ind[i]);
    }
    while (state && num_iter < MAX_ITER)
    {
        memcpy(E_, E, sizeof(double)*P*P);
        memcpy(e_, e, sizeof(double)*P);
        //find index (k,l) of pivot p
        m = 0;
        for (int i = 1; i < P-1; i++)
        {
            //printf("i:%d, %d, %f\n", i, ind[i], A[i*P+ind[i]]);
            if (fabs(A[i * P + ind[i]]) > fabs(A[m * P + ind[m]]))
            {
                m = i;
            }
        }
        k = m; l = ind[k]; p = A[k*P+l];
        y = 0.5*(e[l]-e[k]); d = fabs(y)+sqrt(p*p+y*y);
        r = sqrt(p*p+d*d); c = d/r; s = p/r; t = p*p/d;
        if (y<0) {s = -s; t=-t;}
        A[k*P+l]=0.0; s_update(k, -t, e, changed, &state); s_update(l, t, e, changed, &state);
        
        //rotate rows and cols k and l:
        for (int i=0; i<k; i++)
        {
            s_rotate(i, k, i, l, A, P, c, s);
        }
        for (int i=k+1; i<l; i++)
        {
            s_rotate(k, i, i, l, A, P, c, s);
        }
        for (int i=l+1; i<P; i++)
        {
            s_rotate(k, i, l, i, A, P, c, s);
        }
        //rotate eigenvectors:
        for (int i=0; i<P; i++)
        {
            double e_ik = c*E[i*P+k]-s*E[i*P+l];
            double e_il = s*E[i*P+k]+c*E[i*P+l];
            E[i*P+k] = e_ik;
            E[i*P+l] = e_il;
        }
        ind[k] = s_maxind(A, P, k); ind[l] = s_maxind(A, P, l);
        double diff = l2_diff_norm(e_, e, P);
        double diff_2 = l2_matrix_diff_norm(E_, E, P, P);
        double upper_triangular_sum = s_upper_triangular_sum(A, P);
        // printf("printing e_:");
        // for (int i=0; i<P; i++)
        // {
        //     printf("%f,", e_[i]);
        // }
        printf(" ITER:%d, state:%d, diff:%.10f up-sum:%f", num_iter, state, diff+diff_2, upper_triangular_sum);
        printf("\n");
        num_iter++;
    }
    //print_matrix(A, P, P, 1);
    //sort eigenvalues in desc:
    int *indices = (int *) malloc(sizeof(int)*P);
    for (int i=0; i<P; i++)
    {
        indices[i]=i;
    }
    for (int i=0; i<P; i++)
    {
        int m=i;
        for (int j=i+1; j<P; j++)
        {
            if (e[j]>e[m])
            {
                m=j;
            }
        }
        if (m!=i)
        {
            double temp = e[m];
            e[m] = e[i];
            e[i] = temp;
            temp = indices[m];
            indices[m] = indices[i];
            indices[i] = temp;
        }
    }
    printf("Indices arr:\n");
    for(int i=0; i<P; i++)
    {
        printf("%d,", indices[i]);
    }
    printf("\n");
    //computing SIGMA:
    printf("printing sigma:\n");
    for (int i=0; i<P; i++)
    {
        (*SIGMA)[i] = sqrt(e[i]);
        //printf("%f,", (*SIGMA)[i]);
    }
    printf("\n");
    //computing SIGMA_MATRIX:
    double *temp_sigma = (double *) calloc(P*N, sizeof(double));
    for (int i=0; i<P; i++)
    {
        temp_sigma[i*N+i] = (*SIGMA)[i];
    }
    printf("temp sigma:\n");
    //print_matrix(temp_sigma, P, N, 1);
    //eigenvectors matrix (U for D_T*D):
    printf("printing E:\n");
    //print_matrix(E, P, P, 1);
    printf("printing U:\n");
    for (int row=0; row<P; row++)
    {
        for (int col=0; col<P; col++)
        {
            (*U)[row*P+col] = E[row*P+indices[col]];
      //      printf("%f,", (*U)[row*P+col]);
        }
      //  printf("\n");
    }
    //compute V_T:
    s_compute_V(SIGMA, D_T, U, V_T, N, P);
    printf("printing V_T:\n");
    //print_matrix(*V_T, N, N, 1);
    /*SVD verification*/ //TO BE DELETED
    double *product_one = (double *)malloc(sizeof(double)*P*N);
    s_multiply(*U, P, P, temp_sigma, P, N, product_one);
    printf("PRODUCT ONE:\n");
    //print_matrix(product_one, P, N, 1);
    double *product_two = (double *)malloc(sizeof(double)*P*N);
    s_multiply(product_one, P, N, *V_T, N, N, product_two);
    printf("PRODUCT TWO:\n");
    //print_matrix(product_two, P, N, 1);

    printf("\nORIGINAL D_T:\n");
    //print_matrix(D_T, P, N, 1);
    //printf("\nORIGINAL D:\n");
    //print_matrix(D, N, P, 0);
    printf("\nVERIFIED D_T:\n");
    //print_matrix(product_two, P, N, 1);
    s_matrix_similarity(D_T, P, N, product_two);
    free(product_one);
    free(temp_sigma);
    free(product_two);
}
