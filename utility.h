#ifndef UTILITY_H
#define UTILITY_H

#include <sys/time.h>

typedef unsigned long long ullong;

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

#define BITWIDTH64 64
#define LOG_BITWIDTH64 6
#define CEIL64(X) (((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)
#define FEIL64(X) ((((X)+BITWIDTH64-1)>>LOG_BITWIDTH64)<<LOG_BITWIDTH64)

#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid)); \
    unsigned warpid; asm("mov.u32 %0, %%warpid;":"=r"(warpid));

//Start Timer
#define START_TIMER cudaEvent_t start, stop;\
    cudaEventCreate(&start);\
    cudaEventCreate(&stop);\
    cudaEventRecord(start);

//Stop Timer
#define STOP_TIMER cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    float milliseconds = 0; \
    cudaEventElapsedTime(&milliseconds, start, stop); \
    printf("\n============================\n"); \
    printf("Inference_Time: %.3f ms.",milliseconds);\
    printf("\n============================\n");

// load adjacency matrix
void launch_array(FILE* cf, float* array, unsigned array_size)
{
    if (cf == NULL)
    {
        fprintf(stderr, "NULL pointer to the network configuration file.\n");
        exit(1);
    }
    for (int i=0; i<array_size; i++)
        fscanf(cf, "%f", &array[i]); // better keep the input as 1D
}

// validate output
void validate(float* output, unsigned output_height, unsigned output_width) {
    for (int i=0; i<output_height; i++) {
        for (int j=0; j<output_width; j++) {
            //printf("%.f ", (output[i *output_height + j]));
            printf("%d", (output[i * output_height + j])>0?1:0);
        }
        printf("\n");
    }
}

// cuda error report function
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPU_ERROR: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

#endif