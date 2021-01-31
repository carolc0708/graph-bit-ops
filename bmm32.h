#ifndef BMM32_H
#define BMM32_H

#include "utility.h"

// to print unsigned
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

// pack: Row-major-32
__global__ void PackTo32Row(const float* __restrict__ A, unsigned* B,
        const unsigned A_height, const unsigned A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval=0;
    #pragma unroll
    for (int i=0; i<32; i++)
    {
        float f0 = ((by*32+laneid<A_width)&&(bx*32+i<A_height))?A[(bx*32+i)*A_width+by*32+laneid]:0.0f;
        Bval = (Bval<<1)|(f0>0?1:0); // <- i change here
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

// pack: Col-major-32
__global__ void PackTo32Col(const float* __restrict__ A, unsigned* B,
        const unsigned A_height, const unsigned A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid)); // fetch lane-id
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    unsigned Bval=0;
    #pragma unroll
    for (int i=0; i<32; i++) {
        float f0 = A[(bx*32+i)*A_width + (by*32) + laneid];
        //rotate anticlockwise
        unsigned r0 = __brev(__ballot(f0>0)); // <- i change here
        if (laneid == i) Bval = r0;
    }
    B[by*A_height+bx*32+laneid] = Bval;
}

// unpack
__global__ void UnpackFcOutput32(const unsigned* __restrict__  A, float* B,
        const unsigned A_height, const unsigned A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    unsigned Aval = A[by*gridDim.x*32+bx*32+laneid];
    #pragma unroll
    for (int i=0; i<32; i++)
    {
        unsigned r0 = __shfl(Aval, i); //from lane-i
        if ((32*bx+i)<A_height && by*32+laneid<A_width)
        {
            B[(32*bx+i)*A_width+by*32+laneid] =  ((r0>>(31-laneid)) & 0x1); //
        }
    }
}

// vect/mat input, for each time pass this, multiply A
class MulAParam // a k2-tree matrix-mutiplication unit
{
public:
    // length
    unsigned input_height;
    unsigned input_width;
    unsigned weight_width;

    // Input
    float* input;
    unsigned* input_gpu;

    // A (treat like weight)
    float* weight;
    unsigned* weight_gpu;

    // Output
    unsigned* output;
    unsigned* output_gpu;

    // GPU shadow
    MulAParam* gpu;

public:
    MulAParam(unsigned _input_height, unsigned _input_width, unsigned _weight_width)
    {
        this->input_height = _input_height;
        this->input_width = _input_width;
        this->weight_width = _weight_width;

        this->input = NULL;
        this->input_gpu = NULL;

        this->weight = NULL;
        this->weight_gpu = NULL;

        this->output = NULL;
        this->output_gpu = NULL;

        this->gpu = NULL;

    }

    MulAParam* ready()
    {
        if (input_gpu == NULL)
        {
            fprintf(stderr, "Input data has not been uploaded to GPU.\n");
            exit(1);
        }
        if (output_gpu == NULL)
        {
            fprintf(stderr, "Output on GPU has not been allocated.\n");
            exit(1);
        }
        CUDA_SAFE_CALL(cudaMalloc((void**)&(this->gpu), sizeof(MulAParam)));
        CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(MulAParam), cudaMemcpyHostToDevice));
        return this->gpu;
    }

    MulAParam* initialize(FILE* config_file, float* prev_iter)
    {
        // process prev_iter (assume the initial input) =====
        this->input = prev_iter;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(this->input_gpu), input_bit_bytes()) );
        float* input_float = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(input_float), input_bytes()) );
        CUDA_SAFE_CALL( cudaMemcpy(input_float, this->input, input_bytes(), cudaMemcpyHostToDevice) );

        // Binarize and compact prev_iter
        // v: height = 1, weight = mat_length
        PackTo32Col<<<dim3(CEIL(this->input_height), CEIL(this->input_width)), 32>>>(
                    input_float, this->input_gpu, this->input_height, this->input_width);
        CUDA_SAFE_CALL(cudaFree(input_float));

//        // download to verify --
//        unsigned* input_verify = NULL;
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&(input_verify), input_bit_bytes()) );
//        CUDA_SAFE_CALL( cudaMemcpy(input_verify, this->input_gpu, input_bit_bytes(), cudaMemcpyDeviceToHost) );
//        printf("\n input verify: \n");
//        printf("%d\n", input_bit_size());
//        for(int i=0; i<input_bit_size(); i++) {
//            printf("D%d:", i); bin(input_verify[i]); printf("\n");
//        }
//        printf("\n");  // here, we printed this way
//        // [0...32] -> unsigned D0
//        // [0...32] -> D1
//        // ...
//        // [0...32] -> D31
//
//        CUDA_SAFE_CALL( cudaMemcpy(this->input_gpu, input_verify, input_bit_bytes(), cudaMemcpyHostToDevice) );
//        cudaFreeHost(input_verify);

        // process A =====
        this->weight = (float*) malloc(weight_bytes());
        launch_array(config_file, this->weight, weight_size());
        CUDA_SAFE_CALL(cudaMalloc((void**)&(this->weight_gpu), weight_bit_bytes()));
        printf(" \n weight: \n");
        for(int i=0; i<this->input_width; i++) {
            for (int j=0; j<this->weight_width; j++) {
                printf("%d", (this->weight[i * this->input_width + j])>0?1:0);
            }
            printf("\n");
        }
        printf("\n");

        float* weight_float = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(weight_float), weight_bytes()) );
        CUDA_SAFE_CALL( cudaMemcpy(weight_float, this->weight, weight_bytes(), cudaMemcpyHostToDevice) );

        // Binarize and compact A
        PackTo32Row<<<dim3( CEIL(this->input_width), CEIL(this->weight_width) ), 32>>>(
                                    weight_float, this->weight_gpu, this->input_width, this->weight_width);
        CUDA_SAFE_CALL( cudaFree(weight_float) );

//        // download to verify --
//        unsigned* weight_verify = NULL;
//        CUDA_SAFE_CALL( cudaMallocHost((void**)&(weight_verify), weight_bit_bytes()) );
//        CUDA_SAFE_CALL( cudaMemcpy(weight_verify, this->weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );
//
//        printf(" \n weight_verify: \n");
//        for(int i=0; i<weight_bit_size(); i++) {
//            printf("D%d:", i); bin(weight_verify[i]); printf("\n");
//        }
//        printf("\n"); // here, we printed this way
//        // [0...32] -> unsigned D0
//        // [0...32] -> D1
//        // ...
//        // [0...32] -> D31
//        CUDA_SAFE_CALL( cudaMemcpy(this->weight_gpu, weight_verify, weight_bit_bytes(), cudaMemcpyHostToDevice) );
//        cudaFreeHost(weight_verify);

        // allocate output
        CUDA_SAFE_CALL(cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()));
        CUDA_SAFE_CALL(cudaMemset(this->output_gpu, 0, output_bit_bytes()));

        // set input
        return this->ready();
    }

    // sizes
    //column-major, ceil row
    int input_size() { return this->input_height * this->input_width; }
    int input_bytes() { return this->input_size() * sizeof(float); }
    int input_bit_size() { return FEIL(this->input_height) * CEIL(this->input_width); }
    int input_bit_bytes() { return this->input_bit_size() * sizeof(unsigned); }

    //row-major, ceil column to 32bit
    int weight_size() { return this->input_width * this->weight_width; }
    int weight_bytes() { return this->weight_size()*sizeof(float); }
    int weight_bit_size() { return CEIL(this->input_width) * FEIL(this->weight_width); }
    int weight_bit_bytes() { return this->weight_bit_size()*sizeof(unsigned); }

    //column-major, ceil row to 32bit
    int output_size() { return this->input_height * this->weight_width; }
    int output_bytes() { return this->output_size() * sizeof(unsigned); }
    int output_bit_size() { return FEIL(this->input_height) * CEIL(this->weight_width); }
    int output_bit_bytes() { return this->output_bit_size() * sizeof(unsigned); }

    // output
    unsigned* download_output() // copy output_gpu to output, from device to host
    {
        if(output == NULL) output = (unsigned*)malloc(this->output_bit_bytes());
        CUDA_SAFE_CALL(cudaMemcpy(this->output, this->output_gpu, this->output_bytes(), cudaMemcpyDeviceToHost));

        return this->output;
    }
    float* download_full_output()
    {
        const int size = this->output_size()*sizeof(float);
        float* full_output = (float*)malloc(size);
        float* full_output_gpu = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), size) );
        CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
        UnpackFcOutput32<<<dim3( CEIL(this->input_height), CEIL(this->weight_width) ), 32>>>(
                this->output_gpu, full_output_gpu, this->input_height, this->weight_width);
        CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu, size, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
        return full_output;
    }

    // release
    void release()
    {
        if (this->output != NULL) { free(this->output); this->output = NULL; }
        if (this->output_gpu != NULL) { CUDA_SAFE_CALL(cudaFree(this->output_gpu)); this->output_gpu = NULL; }
        if (this->gpu != NULL) { CUDA_SAFE_CALL(cudaFree(this->gpu)); this->gpu = NULL; }
    }
    ~MulAParam() { release(); }

};

#endif