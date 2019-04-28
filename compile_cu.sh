#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=3
nvcc -Xcompiler="--std=c++0x" -lm -arch=sm_35 -std=c++11 main_cuda.cu lab3_cuda.cu lab3_io.cu -o pca

#export CUDA_LAUNCH_BLOCKING=1
