# graph-bit-ops
* To compile the program `bmm32`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm32 bmm32.cu`
* To compile the program `bmm64`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm64 bmm64.cu`
* Note: it is verified on only CUDA 10.0