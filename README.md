# col380_lab3_suite
Problem Statement: Implement Principal Component Analysis with Singular Vector Decomposition in CUDA

## Directories and files
`testcase/`: contains python script `gen_testcase.py` for sample testcase generation  
`lab3_io.h` and `lab3_io.cu`: functions to read matrix from file and check the correctness of the result  
`main_cuda.cu`: function `main()`  
`lab3_cuda.h`: header file for the functions to be implemented  
`lab3_cuda.cu`: implement the function in this file  
Refer to respective files for furthur details.  
**Do not change the directory structure and prototype of functions.**

## Building and Executing
```
nvcc -lm main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
```
#### Command Line Arguments
The program takes two command line arguments:
- arg1: input filename (consist M, N and D)  
- arg2: retention (percentage of information to be retained by PCA) 

Note that the retention percentage is integer.  Please refer to `main_cuda.cu` for more details.  
To run the program:
```
./pca <input filename> <retention>
```
Example:
```
./pca testcase/testcase_1000_1000 90
```

## Generating testcases
Script `gen_testcase.py` generates testcases as per the parameters and output the generated testcase in file `testcase_<M>_<N>` in the desired format. You might need to change the values of variables `M` and `N` in the script. Read the comments in the script for more information.
```
python3 gen_testcase.py
```

## Input-Output Specifications
#### Input dataset specifications
- M : number of rows (samples) in input matrix D
- N : number of columns (features) in input matrix D
- D : input matrix, #elements in D is (M * N)

The first line of the input file contains `M` followed by `N`. The second line contains elements of matrix `D`. All the values in one line are space separated.  

#### Output Specification
Your program should perform SVD and PCA on the given input and store the results in the variables given in the program. We will check the correctness by calling the functions from the program. You should compute following matrices and values:  
- U : N x N real matrix (to be computed by SVD)
- SIGMA : N x M diagonal matrix of positive real numbers ( to be computed by SVD)
- V_T : M x M real matrix (to be computed by SVD)
- K : number of columns (features) in reduced matrix D_HAT
- D_HAT : reduced matrix (to be computed by PCA)

Refer to `lab3_cuda.h` for more details. **Your program should not output anything on `stdout`.**  

## Submission Instructions
- You are supposed to submit only one file named `lab3_cuda.cu`. Please make sure all the functions you have used are in this file.
- Do not submit other files
- Your code should build and execute as per the instructions given above. Please make sure that your code doesn't need any Makefile.
- Your program should not output anything in `stdout`.

We will not consider the submissions that don't comply with these guidelines.
