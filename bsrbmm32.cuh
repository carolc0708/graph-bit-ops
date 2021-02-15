#ifndef BSRBMM32_CUH
#define BSRBMM32_CUH

__device__ __inline__ void store128(const void* addr, float a, float b, float c, float d)
{
    *((float4*)addr) = make_float4(*(float*)(&a),*(float*)(&b),*(float*)(&c),*(float*)(&d));
}


//template <typename T>
//__global__ void  __launch_bounds__(32,32)
//BMM32_Arow_Brow(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
//        T* C, const int A_height, const int A_width, const int B_width)

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

// the row-by-row model, dense
//template <typename T>
__device__ __inline__ void MatMuldense(MulAParam* p)
{
    GET_LANEID;

    unsigned A_height = FEIL(p->weight_height);
    unsigned A_width = CEIL(p->weight_width);
    unsigned B_width = FEIL(p->input_width);

    // load
    const unsigned* Asub = &(p->input_gpu[blockIdx.x*32]);
    const unsigned* Bsub = &(p->weight_gpu[blockIdx.y*32]);
    float* Csub = &(p->output_gpu[blockIdx.x*B_width*32+blockIdx.y*32]);
    register unsigned Cm[32] = {0}; // allocate 32 register for each lane

    // compute
    for (int i=0; i<A_width; i++) {
        unsigned r0 = Asub[i*A_height+laneid];
        unsigned r1 = Bsub[i*B_width+laneid];

        #pragma unroll
        for (int j=0; j<32; j++)
        {
            unsigned r2 = __shfl(r1, j); // broadcast from lane-j
            Cm[j] |= __popc(r0 & r2);
        }
    }

    // store
    for (int i=0; i<32; i++)
        Csub[laneid*B_width+i] = (float)(Cm[i]);

    // more efficient store
//    for (int i=0; i<32; i+=4)
//        store128((void*)&Csub[laneid*B_width+i],
//                    (T)(Cm[i+0]), (T)(Cm[i+1]),
//                    (T)(Cm[i+2]), (T)(Cm[i+3]));

}

// bsr bmm, bmv
__device__ __inline__ void MatMul(MulAParam* p) { // A (weight, A matrix) * B (input) = C

    if (blockIdx.x < p->bsr_nblockrow+1) {
        GET_LANEID;

        unsigned A_height = FEIL(p->weight_height); // 32 <-- not in use
        unsigned A_width = CEIL(p->weight_width); // 32*nblocks / 32 = nblocks <-- not in use
        unsigned B_width = FEIL(p->input_width); // can be 1 (feil to 32) or any possible output_width

        // load
        unsigned row_start = p->bsr_rowptr_gpu[blockIdx.x]; // 0 32 64 . . . 991
        unsigned row_end = p->bsr_rowptr_gpu[blockIdx.x+1]; // 32 64 96 . . . 991 1022
        const unsigned* Asub = &(p->weight_gpu[row_start]); // block is in continuous layout
        const unsigned* Bsub = &(p->input_gpu[blockIdx.y*32]); // 0, when it is mv
        float* Csub = &(p->output_gpu[blockIdx.x*B_width*32+blockIdx.y*32]);
        register unsigned Cm[32] = {0}; // allocate 32 register for each lane

        // compute
        // if that row has more than 1 col block
        for (int i=row_start; i<row_end; i++) {
            unsigned r0 = Asub[i*32+laneid]; // block is in continuous layout
            unsigned r1 = Bsub[(p->bsr_colind_gpu[i])*B_width+laneid];

            #pragma unroll
            for (int j=0; j<32; j++)
            {
                unsigned r2 = __shfl(r1, j); // broadcast from lane-j
                Cm[j] += __popc(r0 & r2);
            }
        }

        // store
//        for (int i=0; i<32; i++)
//            Csub[laneid*B_width+i] = (float)(Cm[i]>0?1:0);

        // more efficient store
        for (int i=0; i<32; i+=4)
            store128((void*)&Csub[laneid*B_width+i],
                        (float)(Cm[i+0]>0?1:0), (float)(Cm[i+1]>0?1:0),
                        (float)(Cm[i+2]>0?1:0), (float)(Cm[i+3]>0?1:0));

    }
}
#endif