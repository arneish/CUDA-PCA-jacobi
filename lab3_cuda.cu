#include "lab3_cuda.h"
#include <math.h>
#include <stdbool.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <chrono>
using namespace std::chrono;

#define EPSILON 1e-3
#define THRESHOLD 1e-3
#define MAX_BLOCK_SIZE 1024
#define MAX_SWEEPS 30
#define MAX_ITER 10000000
#define MULTIPLY_BLOCK_SIZE 64

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

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
    assert(n1 == m2);
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
    double y = e[k];
    e[k] = y + t;
    if (changed[k] && (y == e[k]))
    {
        changed[k] = false;
        (*state)--;
    }
    else if (!changed[k] && (y != e[k]))
    {
        changed[k] = true;
        (*state)++;
    }
}

void s_rotate(int k, int l, int i, int j, double *A, int P, double c, double s)
{
    double k_l = c * A[k * P + l] - s * A[i * P + j];
    double i_j = s * A[k * P + l] + c * A[i * P + j];
    A[k * P + l] = k_l;
    A[i * P + j] = i_j;
}

void s_merge(double *e, int *indices_e, int left_index, int mid, int right_index)
{
    int i = left_index, j = mid + 1, k = 0;
    double *sorted = (double *)malloc(sizeof(double) * (right_index - left_index + 1));
    int *sorted_indices = (int *)malloc(sizeof(int) * (right_index - left_index + 1));
    assert(sorted_indices!=NULL);
    while (i <= mid && j <= right_index)
    {
        if (fabs(e[i]) >= fabs(e[j]))
        {
            sorted_indices[k] = indices_e[i];
            sorted[k++] = e[i++];
        }
        else
        {
            sorted_indices[k] = indices_e[j];
            sorted[k++] = e[j++];
        }
    }
    while (i <= mid)
    {
        sorted_indices[k] = indices_e[i];
        sorted[k++] = e[i++];
    }
    while (j <= right_index)
    {
        sorted_indices[k] = indices_e[j];
        sorted[k++] = e[j++];
    }
    memcpy(e + left_index, sorted, sizeof(double)*(right_index-left_index+1));
    memcpy(indices_e + left_index, sorted_indices, sizeof(int)*(right_index-left_index+1));
    free(sorted);
    free(sorted_indices);
}

void s_mergesort(double *e, int e_len, int *indices_e, int left_index, int right_index)
{
    //sort e in desc based on abs value
    //rearrange corresponding indices_e appropriately
    assert(left_index <= right_index);
    if (left_index < right_index)
    {
        int mid = (left_index + right_index) / 2;
        s_mergesort(e, e_len, indices_e, left_index, mid);
        s_mergesort(e, e_len, indices_e, mid + 1, right_index);
        s_merge(e, indices_e, left_index, mid, right_index);
    }
}

double l2_matrix_diff_norm(double *E_, double *E, int M, int N)
{
    double sum = 0.0;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
            sum += (E_[i * M + j] - E[i * M + j]) * (E_[i * M + j] - E[i * M + j]);
    }
    return sqrt(sum);
}

double l2_diff_norm(double *e_, double *e, int len)
{
    double sum = 0.0;
    for (int i = 0; i < len; i++)
    {
        sum += (e_[i] - e[i]) * (e_[i] - e[i]);
    }
    return sqrt(sum);
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
    printf("first_product_s:\n");
    //print_matrix(product, N, P, 1);
    //now, multiply product X D_T |=(NXN)
    s_multiply(product, N, P, D_T, P, N, *V_T);
    free(INV_SIGMA);
    free(U_T);
    free(product);
}

double s_matrix_similarity_fabs(double *M_1, int m, int n, double *M_2)
{
    double l2_diff = 0.0;
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            l2_diff += (fabs(M_1[i * n + j]) - fabs(M_2[i * n + j])) * (fabs(M_1[i * n + j]) - fabs(M_2[i * n + j]));
            //printf("%f ", l2_diff);
        }
    }
    l2_diff = sqrt(l2_diff);
    //printf("L2-diff b/w matrices: %f\n", l2_diff);
    return l2_diff;
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
    //printf("L2-diff b/w matrices: %f\n", l2_diff);
    return l2_diff;
}


double s_upper_triangular_sum(double *A, int P)
{
    double sum = 0.0;
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            sum += A[i * P + j] * A[i * P + j];
        }
    }
    return sqrt(sum);
}

void s_set_array(double *A, int P, double *a)
{
    //copying all the A-diagonal elements:
    for (int i = 0; i < P; i++)
    {
        a[i] = A[i * P + i];
    }
    //copying upper triangular A elements:
    int index = P;
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            a[index++] = A[i * P + j];
        }
    }
}

__device__ int device_iter;

__device__ void chess_tourney_params(int P, int *row_pair, int iter)
{
    //NOTE: here, row_pair is thread-local
    int localID = threadIdx.x;
    int index1, index2;
    index1 = (localID + iter) % (P - 1);
    if (localID != 0)
    {
        index2 = (P - localID + iter - 1) % (P - 1);
    }
    else
    {
        index2 = P - 1;
    }
    row_pair[0] = min(index1, index2);
    row_pair[1] = max(index1, index2);
}

template <int BLOCK_SIZE>
__global__ void kernel_MatMul(double *A, int rA, int cA,
                              double *B, int rB, int cB, double *C)
{
    assert(cA == rB);
    int bIDx = blockIdx.x, bIDy = blockIdx.y, tIDx = threadIdx.x, tIDy = threadIdx.y;
    int row_ = bIDy * BLOCK_SIZE + tIDy;
    int col_ = bIDx * BLOCK_SIZE + tIDx;
    __shared__ double A_sub[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double B_sub[BLOCK_SIZE][BLOCK_SIZE];
    double C_sub = 0.0;
    for (int m = 0; m < (BLOCK_SIZE + cA - 1) / BLOCK_SIZE; m++)
    {
        if (m * BLOCK_SIZE + tIDx < cA && row_ < rA)
        {
            A_sub[tIDy][tIDx] = A[row_ * cA + m * BLOCK_SIZE + tIDx];
        }
        else
        {
            A_sub[tIDy][tIDx] = 0.0;
        }

        if (m * BLOCK_SIZE + tIDy < rB && col_ < cB)
        {
            B_sub[tIDy][tIDx] = B[(m * BLOCK_SIZE + tIDy) * cB + col_];
        }
        else
        {
            B_sub[tIDy][tIDx] = 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++)
            C_sub += A_sub[tIDy][k] * B_sub[k][tIDx];
        __syncthreads();
    }
    if (row_ < rA && col_ < cB)
    {
        C[cB * BLOCK_SIZE * bIDy + BLOCK_SIZE * bIDx + cB * tIDy + tIDx] = C_sub;
    }
}

__global__ void kernel_compute_params(double *device_A, int P, int iter, double *device_sine, double *device_cosine, int *device_BlockToElem)
{
    //NOTE: CHECKOUT CUDA OFFICIAL PAGE ON JACOBI (FAST HYPOTENUSE?)
    /*1 Block, P/2 threads: threadID t handles params for its alloted pair (for a particular device_iter)*/
    int localID = threadIdx.x;
    assert(localID < P / 2);
    int k, l;
    double elem, y, d, r, c, s; //,t
    int *row_pair = (int *)malloc(sizeof(int) * 2);
    chess_tourney_params(P, row_pair, iter);
    k = row_pair[0];
    l = row_pair[1];
    free(row_pair);
    device_BlockToElem[localID * 2] = k;     //row
    device_BlockToElem[localID * 2 + 1] = l; //col
    //printf("threadID:%d, (%d,%d)\n", localID, k, l);
    elem = device_A[k * P + l];
    y = (device_A[l * P + l] - device_A[k * P + k]) * 0.5;
    d = fabs(y) + sqrt(elem * elem + y * y);
    r = sqrt(elem * elem + d * d);
    if (r < EPSILON)
    {
        c = 1.0;
        s = 0.0;
    }
    else
    {
        c = d / r;
        s = y / fabs(y) * elem / r; //t=y/fabs(y)*p*p/d;
    }
    device_cosine[k * P + l] = c;
    device_sine[k * P + l] = s;
}

__global__ void kernel_row_update(double *device_A, int local_device_P, double *device_sine, double *device_cosine, int *device_BlockToElem)
{
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=P/2], compute the corresponding two rows: p,q for device_iter*/
    __shared__ int row_pair[2];
    __shared__ double params[2]; //[sin_, cos_]
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {
        row_pair[0] = device_BlockToElem[blockID * 2];
        row_pair[1] = device_BlockToElem[blockID * 2 + 1];
        params[0] = device_sine[row_pair[0] * local_device_P + row_pair[1]];
        params[1] = device_cosine[row_pair[0] * local_device_P + row_pair[1]];
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? Is this better than computing pair(p,q) all over again
    int k = row_pair[0], l = row_pair[1];
    double sin_ = params[0], cos_ = params[1];

    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/
    double new_elem_k, new_elem_l;

    new_elem_k = device_A[k * local_device_P + localID] * cos_ - device_A[l * local_device_P + localID] * sin_;
    new_elem_l = device_A[k * local_device_P + localID] * sin_ + device_A[l * local_device_P + localID] * cos_;
    device_A[k * local_device_P + localID] = new_elem_k;
    device_A[l * local_device_P + localID] = new_elem_l;
}

__global__ void kernel_col_eigenvec_update(double *device_A, int local_device_P, double *device_eigenvectors, double *device_sine, double *device_cosine, int *device_BlockToElem)
{
    int localID = threadIdx.x;
    int blockID = blockIdx.x;

    /*Based on blockID [total blocks=P/2], compute the corresponding two cols: p,q for device_iter*/
    __shared__ int col_pair[2];
    __shared__ double params[2]; //[sin_, cos_]
    if (localID == 0)            //to minimize global memory access latency at the cost of divergence
    {
        col_pair[0] = device_BlockToElem[blockID * 2];
        col_pair[1] = device_BlockToElem[blockID * 2 + 1];
        params[0] = device_sine[col_pair[0] * local_device_P + col_pair[1]];
        params[1] = device_cosine[col_pair[0] * local_device_P + col_pair[1]];
    }
    __syncthreads(); //all "P" threads in the block are synchronized and have access to row_pair(k,l) and params

    //CHECKPOINT: Can you reduce shared-memory bank conflicts here? Is this better than computing pair(p,q) all over again
    int k = col_pair[0], l = col_pair[1];
    double sin_ = params[0], cos_ = params[1];

    /*Concurrent modifications to all row pairs(k,l) [different blocks]*/
    /*Concurrent modifications to different-column elements of a row pair: ["P" threads of the block]*/
    double new_elem_k, new_elem_l, new_eigen_k, new_eigen_l;

    /* col-wise access (inefficient):*/
    new_elem_k = device_A[localID * local_device_P + k] * cos_ - device_A[localID * local_device_P + l] * sin_;
    new_elem_l = device_A[localID * local_device_P + k] * sin_ + device_A[localID * local_device_P + l] * cos_;
    device_A[localID * local_device_P + k] = new_elem_k;
    device_A[localID * local_device_P + l] = new_elem_l;

    new_eigen_k = device_eigenvectors[localID * local_device_P + k]*cos_ - device_eigenvectors[localID*local_device_P+l]*sin_;
    new_eigen_l = device_eigenvectors[localID * local_device_P+k]*sin_ + device_eigenvectors[localID*local_device_P+l]*cos_;
    device_eigenvectors[localID * local_device_P + k] = new_eigen_k;
    device_eigenvectors[localID * local_device_P+l] = new_eigen_l;

    /*row-wise access (efficient):*/
    
}

double compute_offset(double *A, int P)
{
    double sum = 0.0, sum_2 = 0.0;
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            sum += fabs(A[i * P + j]);
            sum_2 += fabs(A[j * P + i]);
        }
    }
    if (fabs(sum_2-sum)>1e-3)
        printf("fail:%f\n",fabs(sum_2-sum));
    assert(fabs(sum_2 - sum) < 1e-3);
    return sum;
}

double findmaxUT(double *A, int P)
{
    double temp = -1;
    for (int i = 0; i < P; i++)
    {
        for (int j = i + 1; j < P; j++)
        {
            temp = max(temp, fabs(A[i * P + j]));
        }
    }
    return temp;
}

void GPU_multiply(double *d_A, const int rA, const int cA, double *d_B, const int rB, const int cB, double *d_C, int block_size)
{
    dim3 threads(block_size, block_size);
    int gridX, gridY;
    if (cB % threads.x==0)
        gridX = cB/threads.x;
    else
        gridX = ceil(cB*1.0/threads.x);
    if (rA % threads.y==0)
        gridY = rA/threads.y;
    else
        gridY = ceil(rA*1.0/threads.y);
    
    dim3 grid(gridX, gridY);
    if (block_size == 16)
    {
        kernel_MatMul<16><<<grid, threads>>>(d_A, rA, cA, d_B, rB, cB, d_C);
    }
    else
    {
        kernel_MatMul<32><<<grid, threads>>>(d_A, rA, cA, d_B, rB, cB, d_C);
    }
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

void GPU_compute_V(double **SIGMA, double *d_D_T, double **U, double **V_T, int N, int P)
{
    //V_T = INV-SIGMA * U_T * M
    double *INV_SIGMA = (double *)calloc(N*P, sizeof(double)); //|=NXP
    for (int i=0; i<P; i++)
    {
        INV_SIGMA[i*P+i] = 1.0/((*SIGMA)[i]);
    }
    double *U_T = (double *)malloc(sizeof(double)*P*P);
    s_transpose(*U, P, P, U_T);

    //first, multiply INV-SIGMA X U_T |=(NXP)
    double *first_product = (double *)malloc(sizeof(double)*N*P);
    double *d_INV_SIGMA, *d_U_T, *d_first_product;
    cudaMalloc((void **)&d_INV_SIGMA, sizeof(double)*N*P);
    cudaMalloc((void **)&d_U_T, sizeof(double)*P*P);
    cudaMalloc((void **)&d_first_product, sizeof(double)*N*P); 
    cudaMemcpy(d_INV_SIGMA, INV_SIGMA, sizeof(double)*N*P, cudaMemcpyHostToDevice);
    cudaMemcpy(d_U_T, U_T, sizeof(double)*P*P, cudaMemcpyHostToDevice);
    
    GPU_multiply(d_INV_SIGMA, N, P, d_U_T, P, P, d_first_product, 32);

    cudaMemcpy(first_product, d_first_product, sizeof(double)*N*P, cudaMemcpyDeviceToHost);
    printf("first_product_GPU:\n");
    //print_matrix(first_product, N, P, 1);
    cudaFree(d_INV_SIGMA);
    cudaFree(d_U_T);
    
    //now, multiply product X D_T |=(NXN)
    double *d_V_T;
    cudaMalloc((void **)&d_V_T, sizeof(double)*N*N);

    GPU_multiply(d_first_product, N, P, d_D_T, P, N, d_V_T, 32);

    cudaMemcpy(*V_T, d_V_T, sizeof(double)*N*N, cudaMemcpyDeviceToHost);
    cudaFree(d_first_product);
    cudaFree(d_V_T);

    gpuErrchk(cudaPeekAtLastError());
    free(INV_SIGMA);
    free(U_T);
    free(first_product);
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
    /****************GPU-PARALLELIZED JACOBI EIGENVALUE ALGORITHM:****************/

    /*1.Perform SVD for D_T*/
    // Get eigen-values & eigen-vectors for D_T*D
    high_resolution_clock::time_point t_begin, t_end, t1;
    t_begin = high_resolution_clock::now();
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
    // limit/=10000;
    // cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit);
    // cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
    // printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);

    double *D_T = (double *)malloc(sizeof(double) * P * N);
    printf("starting transpose of D:\n");
    s_transpose(D, N, P, D_T);
    printf("ending transpose:\n");

    double *A = (double *)calloc(P * P, sizeof(double));
    double *device_D, *device_D_T, *device_A;
    cudaMalloc((void **)&device_D, sizeof(double) * N * P);
    cudaMalloc((void **)&device_D_T, sizeof(double) * P * N);
    cudaMalloc((void **)&device_A, sizeof(double) * P * P);
    cudaMemset(device_A, 0, P * P * sizeof(double));
    cudaMemcpy(device_D, D, sizeof(double) * N * P, cudaMemcpyHostToDevice);
    cudaMemcpy(device_D_T, D_T, sizeof(double) * P * N, cudaMemcpyHostToDevice);

    printf("starting multiplication of D_T*D=A:\n");

    /* Parallelized matrix multiplication (D_T*D=A) */
    t1 = high_resolution_clock::now();
    GPU_multiply(device_D_T, P, N, device_D, N, P, device_A, 32);
    cudaMemcpy(A, device_A, sizeof(double) * P * P, cudaMemcpyDeviceToHost);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    printf("GPU A:%f\n", time_span.count());
    t1 = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t1 - t2);
    printf("CPU A:%f\n", time_span.count());
    //s_matrix_similarity(A, P, P, A_s);
    printf("printing A:\n");
    //print_matrix(A, P, P, 1);
    printf("\nprinting A_s:\n");
    //print_matrix(A_s, P, P, 1);

    double *eigenvectors = s_initialize_identity(P);
    double *device_eigenvectors;
    gpuErrchk(cudaMalloc((void **)&device_eigenvectors, sizeof(double) * P * P));
    cudaMemcpy(device_eigenvectors, eigenvectors, sizeof(double)*P*P, cudaMemcpyHostToDevice);
    double *sine = (double *)calloc(P * P, sizeof(double));
    double *cosine = (double *)calloc(P * P, sizeof(double));
    double *device_sine;
    double *device_cosine;
    int *device_BlockToElem; //to store mapping of P/2 "blocks" to element at (p,q), computed in the first kernel call
    gpuErrchk(cudaMalloc((void **)&device_sine, sizeof(double) * P * P));
    gpuErrchk(cudaMalloc((void **)&device_cosine, sizeof(double) * P * P));
    gpuErrchk(cudaMalloc((void **)&device_BlockToElem, sizeof(int) * P / 2 * 2));
    cudaMemset(device_sine, 0, P * P * sizeof(double));
    cudaMemset(device_cosine, 0, P * P * sizeof(double));

    assert(P % 2 == 0); //only writing for even "f"

    int grid_size, block_size=P, iter = 0, counter = 0;
    double offset_ = THRESHOLD + 1;
    if (P%2==0)
        grid_size = P / 2;
    else
        grid_size = P/2+1;
    while (counter < MAX_SWEEPS && offset_ > THRESHOLD) //sweeps
    {
        t1 = high_resolution_clock::now();
        iter = 0;
        while (iter < P - 1)
        {
            //Compute rotation parameters for all (p,q): q>p
            kernel_compute_params<<<1, grid_size>>>(device_A, P, iter, device_sine, device_cosine, device_BlockToElem);
            cudaDeviceSynchronize();

            //row-update kernel
            kernel_row_update<<<grid_size, block_size>>>(device_A, P, device_sine, device_cosine, device_BlockToElem);
            cudaDeviceSynchronize();

            //col-update & eigen-vector update kernel
            kernel_col_eigenvec_update<<<grid_size, block_size>>>(device_A, P, device_eigenvectors, device_sine, device_cosine, device_BlockToElem);
            cudaDeviceSynchronize();
            iter++;
        }
        cudaMemcpy(A, device_A, sizeof(double) * P * P, cudaMemcpyDeviceToHost);
        offset_ = compute_offset(A, P);
        t2 = high_resolution_clock::now();
        time_span = (t2 - t1);
        printf("Sweep:%d, offset:%f, sweep-time:%f\n", counter, offset_, time_span.count());
        counter++;
    }

    cudaMemcpy(eigenvectors, device_eigenvectors, sizeof(double)*P*P, cudaMemcpyDeviceToHost);
    cudaMemcpy(cosine, device_cosine, sizeof(double) * P * P, cudaMemcpyDeviceToHost);
    cudaMemcpy(sine, device_sine, sizeof(double) * P * P, cudaMemcpyDeviceToHost);

    double *eigenvalues = (double *)malloc(sizeof(double) * P);
    for (int i = 0; i < P; i++)
    {
        eigenvalues[i] = A[i * P + i];
    	
	}

    //sort eigenvalues in desc:
    int *e_indices = (int *)malloc(sizeof(int) * P);
    for (int i = 0; i < P; i++)
    {
        e_indices[i] = i;
    }
    s_mergesort(eigenvalues, P, e_indices, 0, P - 1);
    printf("Indices arr:\n");
    // for (int i = 0; i < min(P,50); i++)
    // {
    //     printf("%d,", e_indices[i]);
    // }
    printf("\n");
    printf("e arr:\n");
    double temp_ = eigenvalues[0];
    for (int i = 0; i < P; i++)
    {
        assert(temp_>=eigenvalues[i]);
        temp_=eigenvalues[i];
	printf("%f,", eigenvalues[i]);
    }
    printf("\n");
    // printf("printing GPU E:\n");
    // for (int x=0; x<P; x++)
    // {
    //     for (int y=0; y<P; y++)
    //     {
    //         printf("%f,", eigenvectors[x*P+e_indices[y]]);
    //     }
    //     printf("\n");
    // }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    
    //computing SIGMA:
    printf("printing sigma:\n");
    double *temp_sigma_ = (double *)calloc(P * N, sizeof(double));
    double sum_eigenvalues = 0.0;

    for (int i = 0; i < P; i++)
    {
        (*SIGMA)[i] = sqrt(eigenvalues[i]);
        sum_eigenvalues+=(*SIGMA)[i];
        //printf("%f,", (*SIGMA)[i]);
        temp_sigma_[i * N + i] = (*SIGMA)[i];
    }
    printf("sum evals_G:%f\n", sum_eigenvalues);
    printf("\n");
    
    //computing U:
    printf("GPU U:\n");
    for (int row = 0; row < P; row++)
    {
        for (int col = 0; col < P; col++)
        {
            (*U)[row * P + col] = eigenvectors[row * P + e_indices[col]];
        }
    }

    //compute V_T:
    double *V_T_g = (double *)calloc(N*N, sizeof(double));

    GPU_compute_V(SIGMA, device_D_T, U, &V_T_g, N, P);
    printf("printing V_T:\n");

    
    //Parallelized (PCA):
    *K=0;
    double retention_ = 0.0;
    int count_ = 0;
    while((retention_<retention) && (count_ < P))
    {
        retention_+=((*SIGMA)[count_]/sum_eigenvalues)*100;
        (*K)++;
        count_++;
    }
    printf("K GPU:%d, retention:%f\n", *K, retention_);
    double *W = (double *)malloc(sizeof(double)*P*(*K));
    *D_HAT = (double *)malloc(sizeof(double)*N*(*K));
    for (int r=0; r<P; r++)
    {
        for (int c=0; c<(*K); c++)
        {
            W[r*(*K)+c] = (*U)[r*P+c];
        }
    }

    //now, multiply D*W |=(NxP.PxK=NxK)
    //void GPU_multiply(double *d_A, const int rA, const int cA, double *d_B, const int rB, const int cB, double *d_C, int block_size)
    double *device_W, *device_D_HAT;
    cudaMalloc((void **)&device_W, sizeof(double)*P*(*K));
    cudaMalloc((void**)&device_D_HAT, sizeof(double)*N*(*K));
    cudaMemcpy(device_W, W, sizeof(double)*P*(*K), cudaMemcpyHostToDevice);

    GPU_multiply(device_D, N, P, device_W, P, *K, device_D_HAT, 32);

    cudaMemcpy(*D_HAT, device_D_HAT, sizeof(double)*N*(*K), cudaMemcpyDeviceToHost);
    
    t_end = high_resolution_clock::now();
    time_span = duration_cast<duration<double>>(t_end - t_begin);
    printf("TOTAL TIME:%f\n", time_span.count());
    
    //return;

    /****************SERIAL JACOBI EIGENVALUE ALGORITHM:****************/

    //begin Jacobi eigenvalue algorithm:
    int state = P, num_iter = 0, m, k, l; //m: pivot row identifier
    double p, y, d, r, c, s, t;           //p: pivot element, c: cos, s: sin
    double *E = s_initialize_identity(P); //P*P
    double *E_ = (double *)malloc(sizeof(double) * P * P);
    double *e = (double *)malloc(sizeof(double) * P); //init eigen-values array
    double *e_ = (double *)malloc(sizeof(double) * P);
    int *ind = (int *)malloc(sizeof(int) * P);        //init maxindex array
    bool *changed = (bool *)malloc(sizeof(bool) * P); //change in eigen_value[k]
    double *A_s = (double *)calloc(P * P, sizeof(double));
    s_multiply(D_T, P, N, D, N, P, A_s);

    printf("printing A_s:\n");
    //print_matrix(A_s, P, P, 1);
    for (int i = 0; i < P; i++)
    {
        ind[i] = s_maxind(A_s, P, i); //NOTE: undefined for last row
        e[i] = A_s[i * P + i];
        changed[i] = true;
        //printf("%d, %d\n", i, ind[i]);
    }
    while (state && num_iter < MAX_ITER)
    {
        memcpy(E_, E, sizeof(double) * P * P);
        memcpy(e_, e, sizeof(double) * P);
        //find index (k,l) of pivot p
        m = 0;
        for (int i = 1; i < P - 1; i++)
        {
            //printf("i:%d, %d, %f\n", i, ind[i], A[i*P+ind[i]]);
            if (fabs(A_s[i * P + ind[i]]) > fabs(A_s[m * P + ind[m]]))
            {
                m = i;
            }
        }
        k = m;
        l = ind[k];
        p = A_s[k * P + l];
        y = 0.5 * (e[l] - e[k]);
        d = fabs(y) + sqrt(p * p + y * y);
        r = sqrt(p * p + d * d);
        c = d / r;
        s = p / r;
        t = p * p / d;
        if (y < 0)
        {
            s = -s;
            t = -t;
        }
        A_s[k * P + l] = 0.0;
        s_update(k, -t, e, changed, &state);
        s_update(l, t, e, changed, &state);

        //rotate rows and cols k and l:
        for (int i = 0; i < k; i++)
        {
            s_rotate(i, k, i, l, A_s, P, c, s);
        }
        for (int i = k + 1; i < l; i++)
        {
            s_rotate(k, i, i, l, A_s, P, c, s);
        }
        for (int i = l + 1; i < P; i++)
        {
            s_rotate(k, i, l, i, A_s, P, c, s);
        }
        //rotate eigenvectors:
        for (int i = 0; i < P; i++)
        {
            double e_ik = c * E[i * P + k] - s * E[i * P + l];
            double e_il = s * E[i * P + k] + c * E[i * P + l];
            E[i * P + k] = e_ik;
            E[i * P + l] = e_il;
        }
        ind[k] = s_maxind(A_s, P, k);
        ind[l] = s_maxind(A_s, P, l);
        double diff = l2_diff_norm(e_, e, P);
        double diff_2 = l2_matrix_diff_norm(E_, E, P, P);
        double upper_triangular_sum = s_upper_triangular_sum(A_s, P);
        printf("\rITER:%d, state:%d, diff:%.10f up-sum:%f", num_iter, state, diff + diff_2, upper_triangular_sum);
        fflush(stdout);
        num_iter++;
    }
    //sort eigenvalues in desc:
    int *indices = (int *)malloc(sizeof(int) * P);
    for (int i = 0; i < P; i++)
    {
        indices[i] = i;
    }
    s_mergesort(e, P, indices, 0, P - 1);
    printf("Indices arr:\n");
    for (int i = 0; i < P; i++)
    {
        printf("%d,", indices[i]);
    }
    printf("\n");
    printf("e arr:\n");
    for (int i = 0; i < P; i++)
    {
        printf("%f,", e[i]);
    }
    printf("\n");

    // //computing SIGMA:
    // printf("printing sigma:\n");
    double sum_eigenvalues_s=0.0;
    for (int i = 0; i < P; i++)
    {
        (*SIGMA)[i] = sqrt(e[i]);
        sum_eigenvalues_s+=(*SIGMA)[i];
        //printf("%f,", (*SIGMA)[i]);
    }
    printf("sum evals_s:%f\n", sum_eigenvalues_s);
    printf("\n");
    //computing SIGMA_MATRIX:
    double *temp_sigma = (double *)calloc(P * N, sizeof(double));
    for (int i = 0; i < P; i++)
    {
        assert(e[i]>=0);
        temp_sigma[i * N + i] = sqrt(e[i]);
    }

    //eigenvectors matrix (U for D_T*D):
    printf("printing E:\n");

    //L2
    double sum_temp=0.0;
    for (int x=0; x<P; x++)
    {
        for (int y=0; y<P; y++)
        {
            sum_temp+=(fabs(E[x*P+indices[y]])-fabs(eigenvectors[x*P+e_indices[y]]))*(fabs(E[x*P+indices[y]])-fabs(eigenvectors[x*P+e_indices[y]]));
        }
    }
    printf("L-2 fabs diff in E:%f\n", sqrt(sum_temp));
    
    printf("printing U:\n");
    double *u_s = (double *) malloc(sizeof(double)*P*P);
    for (int row = 0; row < P; row++)
    {
        for (int col = 0; col < P; col++)
        {
            // (*U)[row * P + col] = E[row * P + indices[col]];
            u_s[row * P + col] = E[row * P + indices[col]];
           // printf("%f,", (*U)[row*P+col]);
        }
        //printf("\n");
    }
    //compute V_T:
    double *V_T_s = (double *)calloc(N*N, sizeof(double));
    s_compute_V(SIGMA, D_T, &u_s, &V_T_s, N, P);
    printf("\nprinting V_T_G:\n");
    //print_matrix(V_T_g, N, N, 1);

    printf("\nprinting V_T:\n");
    double sim1=s_matrix_similarity_fabs(*U, P, P, u_s);
    printf("L2-matrix fabs sim bw U's:%.10f\n", sim1);
    double sim2 = s_matrix_similarity_fabs(V_T_g, N, N, V_T_s);
    printf("L2-matrix fabs sim bw V_T's:%.10f\n", sim2);
    sim2 = s_matrix_similarity_fabs(V_T_g, N, N, V_T_g);
    printf("L2-matrix fabs sim bw V_Tg's same:%.10f\n", sim2);
    printf("prinitng V_t_s:\n");
    //print_matrix(V_T_s, N, N, 1);
    
    //compute serial PCA:
     int K_s=0;
     double retention_s = 0.0;
     int count_s = 0;
     while((retention_s<retention) && (count_s < P))
     {
         retention_s+=((*SIGMA)[count_s]/sum_eigenvalues_s)*100;
         K_s++;
         count_s++;
     }
     printf("K_s CPU:%d, retention_S:%f\n", K_s, retention_s);
     assert(*K==K_s);
     double *W_s = (double *)malloc(sizeof(double)*P*K_s);
     double *D_HAT_s = (double *)malloc(sizeof(double)*N*K_s);
     for (int r=0; r<P; r++)
     {
         for (int c=0; c<K_s; c++)
         {
             W_s[r*K_s+c] = u_s[r*P+c];
         }
     }
 
     //now, serially multiply D*W |=(NxP.PxK=NxK)
     s_multiply(D, N, P, W_s, P, K_s, D_HAT_s);
    sim2 = s_matrix_similarity_fabs(D_HAT_s, N, K_s, *D_HAT);
    printf("L2-matrix fabs sim bw PCAs:%.10f\n", sim2);
    sim2 = s_matrix_similarity_fabs(*D_HAT, N, K_s, *D_HAT);
    printf("L2-matrix fabs sim bw same G PCAs:%.10f\n", sim2);

    return;

    
}
