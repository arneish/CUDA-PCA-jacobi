nvcc -g -lm -arch=sm_35 -std=c++11 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca
export CUDA_LAUNCH_BLOCKING=1
