# graph-bit-ops

## Environment
* using `cuda/10.1.243`
* in `/data` directory, use `fetch_data.sh` to get all testing matrices

## BSR BMV (verified)
* in directory `/bsrbmv`
* `make` to get `bsrbmv32`, `bsrbmv64`
* `./bsrbmv32 {/path/to/matrix/file.mtx}`

## BSR BMM (verified)
* in directory `/bsrbmm`
* `make` to get `bsrbmm32`, `bsrbmm64`
* `./bsrbmm32 {/path/to/matrix/file.mtx}`

## BMM (result not verified)
* in directory `/bmm`
* To compile the program `bmm32`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm32 bmm32.cu`
* To compile the program `bmm64`: `nvcc -std=c++11 -O3 -w -arch=sm_60 -maxrregcount=64 -rdc=true -o bmm64 bmm64.cu`
* Note: it is tested on only CUDA 10.0