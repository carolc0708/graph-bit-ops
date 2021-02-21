# graph-bit-ops

## BSR BMV
* in directory `/bsrbmv`
* `make`
* `./bsrbmv test.mtx` or `./bsrbmv {/path/to/matrix/file}`
*  using `cuda/10.1.243`

## BMM (result not verified)
* in directory `/bmm`
* To compile the program `bmm32`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm32 bmm32.cu`
* To compile the program `bmm64`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm64 bmm64.cu`
* Note: it is verified on only CUDA 10.0