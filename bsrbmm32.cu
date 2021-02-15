#include <stdio.h>
#include <assert.h>
#include "bsrbmm32.h"
#include "bsrbmm32.cuh"

using namespace std;

__global__ void bfs(MulAParam* p)
{
    //grid_group grid = this_grid();
    MatMul(p);
    //grid.sync();

}

int bmv() {

    // config input
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned mat_length = 1000; // assume we already know
    const unsigned input_width = 1;

    // set input vector/matrix
    float* f = NULL;

    // f
    srand(3333);
    cudaMallocHost((void**) &f, sizeof(float)*mat_length*input_width);
    for (int i=0; i<mat_length; i++) {
        f[i] = rand() % 2;
    }

    // print f
    printf(" \n input: \n");
    for (int i=0; i<mat_length; i++) {
        printf("%d", f[i]>0?1:0);
    }
    printf("\n");

    // set iterations
    MulAParam* p = new MulAParam(mat_length, input_width);
    std::string filename = "G43";
    MulAParam* p_gpu = p->initialize(filename, f);


    // setup kernel
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties (&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, bfs, numThreads, 0);
    void* args[] = {&p_gpu};


    START_TIMER;
    cudaLaunchCooperativeKernel((void*)bfs, numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, args);
    STOP_TIMER;

    // output
    float* output = p->download_output();
    validate(output, FEIL(mat_length), FEIL(input_width)); // only printout for now


    // release
    delete p;
    cudaFreeHost(f);

    return 0;

}

int bmm () {

    // config input
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned mat_length = 1000; // assume we already know it
    const unsigned input_width = 1000;

    // set input vector/matrix
    float* f = NULL;

    // f
    srand(3333);
    cudaMallocHost((void**) &f, sizeof(unsigned)*mat_length*input_width);
    for (int i = 0; i < mat_length; ++i) {
        for (int j = 0; j < input_width; ++j) {
            f[i * mat_length + j] = rand() % 2;
        }
    }

    // print f
    printf(" \n input: \n");
    for (int i = 0; i < mat_length; ++i) {
        for (int j = 0; j < input_width; ++j) {
            printf("%d", (f[i * mat_length + j])>0?1:0);
        }
        printf("\n");
    }


    // set iterations
    MulAParam* p = new MulAParam(mat_length, input_width);
    std::string filename = "G43";
    MulAParam* p_gpu = p->initialize(filename, f);

    // setup kernel
    int numThreads = 1024;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties (&deviceProp, dev);
    int numBlocksPerSm;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, bfs, numThreads, 0);
    void* args[] = {&p_gpu};


    START_TIMER;
    cudaLaunchCooperativeKernel((void*)bfs, numBlocksPerSm*deviceProp.multiProcessorCount, numThreads, args);
    STOP_TIMER;

    // output
    //float* output = p->download_output();
    //validate(output, input_height, mat_length); // only printout for now


    // release
    delete p;
    cudaFreeHost(f);

    return 0;

}

int main()
{
    bmv();
    //bmm();
}