
#include <stdio.h>
#include <assert.h>
#include "bmm32.h"
#include "bmm32.cuh"

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
    const unsigned input_height = 1;
    const unsigned mat_length = 32;

    // set input vector/matrix
    float* f = NULL;

    // f
    srand(3333);
    cudaMallocHost((void**) &f, sizeof(unsigned)*input_height*mat_length);
    for (int i=0; i<mat_length; i++) {
        f[i] = rand() % 2;
    }

    // print f
    printf(" \n input: \n");
    for (int i = 0; i < mat_length; ++i) {
        printf("%d", f[i]>0?1:0);
    }

    // set iterations
    MulAParam* p = new MulAParam(input_height, mat_length, mat_length);
    FILE* config_file = fopen("matrix.csv", "r");
    MulAParam* p_gpu = p->initialize(config_file, f);

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
    float* output = p->download_full_output();
    validate(output, input_height, mat_length); // only printout for now


    // release
    delete p;
    cudaFreeHost(f);

    return 0;

}

int bmm () {

    // config input
    int dev = 0;
    cudaSetDevice(dev);
    const unsigned input_height = 32;
    const unsigned mat_length = 32;

    // set input vector/matrix
    float* f = NULL;

    // f
    srand(3333);
    cudaMallocHost((void**) &f, sizeof(unsigned)*input_height*mat_length);
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < mat_length; ++j) {
            f[i * mat_length + j] = rand() % 2;
        }
    }

    // print f
    printf(" \n input: \n");
    for (int i = 0; i < input_height; ++i) {
        for (int j = 0; j < mat_length; ++j) {
            printf("%d", (f[i * input_height + j])>0?1:0);
        }
        printf("\n");
    }


    // set iterations
    MulAParam* p = new MulAParam(input_height, mat_length, mat_length);
    FILE* config_file = fopen("matrix.csv", "r");
    MulAParam* p_gpu = p->initialize(config_file, f);

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
    float* output = p->download_full_output();
    validate(output, input_height, mat_length); // only printout for now


    // release
    delete p;
    cudaFreeHost(f);

    return 0;

}

int main()
{
    //bmv();
    bmm();
}