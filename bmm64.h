#ifndef BMM64_H
#define BMM64_H
#include "utility.h"

// to print ullong
void bin(ullong n)
{
    ullong i;
    for(i = 1ULL << 63; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

// pack to row-major-64
__global__ void PackTo64Row(const float* __restrict__ A, ullong* B,
        const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    ullong Bval=0;
    #pragma unroll
    for (int i=0; i<64; i++)
    {
        float f0 = ((by*32+laneid<A_width) && (bx*64+i<A_height))?A[(bx*64+i)*A_width+by*32+laneid]:0.0f;
        Bval = (Bval<<1)|(f0>0?1:0);
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}

// pack to col-major-64
__global__ void PackTo64Col(const float* __restrict__ A, ullong* B,
        const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned bx = blockIdx.x;
    const unsigned by = blockIdx.y;
    ullong Bval=0;
    #pragma unroll
    for (int i=0; i<64; i++)
    {
        float f0 = A[(bx*64+i)*A_width+by*32+laneid];
        float f1 = A[(bx*64+i)*A_width+by*32+32+laneid];
        unsigned r0 = __brev(__ballot(f0>0));
        unsigned r1 = __brev(__ballot(f1>0));
        ullong l0;
        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r1),"r"(r0)); //(low,high)
        if (laneid == i) Bval = l0;
    }
    if (laneid < A_height*A_width)
        B[bx*gridDim.y*32+by*32+laneid] = Bval;
}


// unpack row-major
__global__ void UnpackFcOutput64(const ullong* __restrict__ A, float* B,
        const int A_height, const int A_width)
{
    unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));
    const unsigned by = blockIdx.y;
    const unsigned bx = blockIdx.x;
    /*ullong Aval = A[by*A_height+bx*32+laneid];*/
    ullong Aval = A[by*gridDim.x*32+bx*32+laneid];
    unsigned r0, r1;
    asm volatile("mov.b64 {%0,%1}, %2;":"=r"(r1),"=r"(r0):"l"(Aval)); //lo,hi
    #pragma unroll
    for (int i=0; i<32; i++)
    {
        unsigned r2 = __shfl(r0, i); //from lane-i, only 32-bit shuffle is allowed
        unsigned r3 = __shfl(r1, i); //r2 left, r3 right
        if ((32*bx+i)<A_height)
        {
            if (by*64+laneid < A_width)
                B[(32*bx+i)*A_width+by*64+laneid] = (float)((r2>>(31-laneid)) & 0x1); // not sure
            if (by*64+32+laneid < A_width)
                B[(32*bx+i)*A_width+by*64+32+laneid] = (float)((r3>>(31-laneid)) & 0x1); // not sure, haven't verify
        }
    }
}

// class
class MulAParam64
{
public:
    //Input
    float* input;
    ullong* input_gpu;
    unsigned input_width;
    unsigned input_height;
    //Weight
    float* weight;
    ullong* weight_gpu;
    unsigned weight_width;
    unsigned weight_height;
    //Output
    ullong* output;
    ullong* output_gpu;
    unsigned output_width;
    unsigned output_height;
    //GPU shadow
    MulAParam64* gpu;

public:
    MulAParam64(unsigned _input_height, unsigned _input_width, unsigned _weight_width)
    {
        //length
        this->input_height = _input_height;
        this->input_width = _input_width;
        this->weight_width = _weight_width;

        this->input = NULL;
        this->input_gpu = NULL;
        this->weight = NULL; // this is adjacency matrix
        this->weight_gpu = NULL;
        this->output = NULL;
        this->output_gpu = NULL;
        this->gpu = NULL;
    }

    MulAParam64* ready()
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
        CUDA_SAFE_CALL( cudaMalloc((void**)&(this->gpu), sizeof(MulAParam64)));
        CUDA_SAFE_CALL( cudaMemcpy(this->gpu, this, sizeof(MulAParam64), cudaMemcpyHostToDevice));
        return this->gpu;
    }

    void set_input_gpu(ullong* input_gpu)
    {
        this->input_gpu = input_gpu;
    }

    MulAParam64* initialize(FILE* config_file, float* prev_iter)
    {
        // process prev_iter (assume the initial input) ===
        this->input = prev_iter;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(this->input_gpu), input_bit_bytes()) );
        float* input_float = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(input_float), input_bytes()) );
        CUDA_SAFE_CALL( cudaMemcpy(input_float, this->input, input_bytes(), cudaMemcpyHostToDevice) );

        // binarized and pack
        PackTo64Col<<<dim3( CEIL64(this->input_height), 2*CEIL64(this->input_width) ), 32>>>(
                    input_float, this->input_gpu, this->input_height, this->input_width);
        CUDA_SAFE_CALL(cudaFree(input_float));

        // download to verify --
        ullong* input_verify = NULL;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&(input_verify), input_bit_bytes()) );
        CUDA_SAFE_CALL( cudaMemcpy(input_verify, this->input_gpu, input_bit_bytes(), cudaMemcpyDeviceToHost) );
        printf("\n input verify: \n");
        for(int i=0; i<input_bit_size(); i++) {
            //printf("%llu ", input_verify[i]);
            printf("D%d:", i); bin(input_verify[i]); printf("\n");
        }
        printf("\n");  // here, we printed this way
        // [0...64] -> unsigned D0
        // [0...64] -> D1
        // ...
        // [0...64] -> D64

        CUDA_SAFE_CALL( cudaMemcpy(this->input_gpu, input_verify, input_bit_bytes(), cudaMemcpyHostToDevice) );
        cudaFreeHost(input_verify);

        //Process weight ===
        this->weight = (float*)malloc(weight_bytes());
        launch_array(config_file, this->weight, weight_size());
        CUDA_SAFE_CALL( cudaMalloc((void**)&(this->weight_gpu), weight_bit_bytes()) );
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
        CUDA_SAFE_CALL( cudaMemcpy(weight_float, weight, weight_bytes(), cudaMemcpyHostToDevice) );

        //Binarize and compact weight
        PackTo64Row<<<dim3( CEIL64(this->input_width), 2*CEIL64(this->weight_width) ), 32>>>(
                weight_float, this->weight_gpu, this->input_width, this->weight_width);
        CUDA_SAFE_CALL( cudaFree(weight_float) );

        // download to verify --
        ullong* weight_verify = NULL;
        CUDA_SAFE_CALL( cudaMallocHost((void**)&(weight_verify), weight_bit_bytes()) );
        CUDA_SAFE_CALL( cudaMemcpy(weight_verify, this->weight_gpu, weight_bit_bytes(), cudaMemcpyDeviceToHost) );

        printf(" \n weight_verify: \n");
        for(int i=0; i<weight_bit_size(); i++) {
            //printf("%llu ", weight_verify[i]);
            printf("D%d:", i); bin(weight_verify[i]); printf("\n");
        }
        printf("\n"); // here, we printed this way
        // [0...64] -> unsigned D0
        // [0...64] -> D1
        // ...
        // [0...64] -> D64
        CUDA_SAFE_CALL( cudaMemcpy(this->weight_gpu, weight_verify, weight_bit_bytes(), cudaMemcpyHostToDevice) );
        cudaFreeHost(weight_verify);

        //Allocate output gpu
        CUDA_SAFE_CALL( cudaMalloc((void**)&(this->output_gpu), output_bit_bytes()) );
        CUDA_SAFE_CALL( cudaMemset(this->output_gpu, 0, output_bit_bytes()) );

        //set_input_gpu(prev_layer_gpu);
        return this->ready();
    }

    //column-major, ceil row
    int input_size() { return this->input_height*this->input_width;}
    int input_bytes() { return input_size()*sizeof(float);}
    int input_bit_size() { return  FEIL64(this->input_height)*CEIL64(this->input_width);}
    int input_bit_bytes() { return input_bit_size()*sizeof(ullong);}

    //row-major, ceil column to 64bit
    int weight_size() { return this->input_width*this->weight_width;}
    int weight_bytes() { return weight_size()*sizeof(float);}
    int weight_bit_size() { return CEIL64(this->input_width)*FEIL64(this->weight_width);}
    int weight_bit_bytes() { return weight_bit_size()*sizeof(ullong);}

    //column-major, ceil row to 32bit
    int output_size() { return this->input_height*this->weight_width;}
    int output_bytes() { return output_size()*sizeof(float);}
    int output_bit_size() { return FEIL64(this->input_height)*CEIL64(this->weight_width);}
    int output_bit_bytes() { return output_bit_size()*sizeof(ullong);}

    ullong* get_output_gpu()
    {
        return this->output_gpu;
    }

    ullong* download_output()
    {
        if (output == NULL) output = (ullong*)malloc(output_bit_bytes());
        CUDA_SAFE_CALL( cudaMemcpy(output, output_gpu,
                    output_bit_bytes(), cudaMemcpyDeviceToHost) );

//        printf(" \n output_verify: \n");
//        for(int i=0; i<output_bit_size(); i++) {
//            //printf("%llu ", weight_verify[i]);
//            printf("D%d:", i); bin(output[i]); printf("\n");
//        }
//        printf("\n"); // here, we printed this way
//        // [0...64] -> unsigned D0
//        // [0...64] -> D1
//        // ...
//        // [0...64] -> D64
        return this->output;
    }

    float* download_full_output()
    {
        const int size = this->output_size()*sizeof(float);
        float* full_output = (float*)malloc(size);
        float* full_output_gpu = NULL;
        CUDA_SAFE_CALL( cudaMalloc((void**)&(full_output_gpu), size) );
        CUDA_SAFE_CALL( cudaMemset(full_output_gpu, 0, size) );
        UnpackFcOutput64<<<dim3( 2*CEIL64(this->input_height), CEIL64(this->weight_width) ), 32>>>(
                output_gpu, full_output_gpu, this->input_height, this->weight_width);
        CUDA_SAFE_CALL( cudaMemcpy(full_output, full_output_gpu,
                    size, cudaMemcpyDeviceToHost) );
        CUDA_SAFE_CALL( cudaFree(full_output_gpu) );
        return full_output;
    }

    void release()
    {
        if (this->weight != NULL) { free(this->weight); this->weight = NULL;}
        if (this->output != NULL) { free(this->output); this->output = NULL;}
        if (this->weight_gpu != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(weight_gpu) );
            weight_gpu = NULL;
        }
        if (this->output_gpu != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(this->output_gpu) );
            this->output_gpu = NULL;
        }
        if (this->gpu != NULL)
        {
            CUDA_SAFE_CALL( cudaFree(this->gpu));
            this->gpu = NULL;
        }

    }
    ~MulAParam64() { release(); }
};

#endif