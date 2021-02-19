#include <stdio.h>
#include <assert.h>
#include <sys/time.h>

#define BITWIDTH 32
#define LOG_BITWIDTH 5
#define CEIL(X) (((X)+BITWIDTH-1)>>LOG_BITWIDTH)
#define FEIL(X) ((((X)+BITWIDTH-1)>>LOG_BITWIDTH)<<LOG_BITWIDTH)

typedef unsigned long long ullong;

// A faster way to obtain lane id in a warp
#define GET_LANEID unsigned laneid; asm("mov.u32 %0, %%laneid;":"=r"(laneid));

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store64(const void* addr, T a, T b)
{
    *((float2*)addr) = make_float2(*(float*)(&a),*(float*)(&b));
}

//For higher memory access efficiency
template <typename T>
__device__ __inline__ void store128(const void* addr, T a, T b, T c, T d)
{
    *((float4*)addr) = make_float4(*(float*)(&a),*(float*)(&b),*(float*)(&c),*(float*)(&d));
}

// to print unsigned
void bin(unsigned n)
{
    unsigned i;
    for (i = 1 << 31; i > 0; i = i / 2)
        (n & i) ? printf("1") : printf("0");
}

// weight should be col-major packing, layout is 32 * (32*numofblocks)
// input should be row-major packing, layout is whatever it is originally

// col-major packing bit 32
template <typename T>
__global__ void ToBit32Col(const T* __restrict__ A, unsigned* B,
        const int A_height, const int A_width) // blocksize, nblocks * blocksize
{
    GET_LANEID;
    const unsigned by = blockIdx.y; // nblocks
    const unsigned bx = blockIdx.x; // 1
    unsigned Bval;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[by*32*32+i*32+laneid];
        unsigned r0 = __brev(__ballot(f0>0));
        if (laneid == i) Bval = r0;
    }
    B[by*32+laneid] = Bval;
}

// row-major packing bit 32
template <typename T>
__global__ void ToBit32Row(const T* __restrict__ A, unsigned* B,
        const int A_height, const int A_width) // no get lane?
{
    const unsigned bx = blockIdx.x; // blockrows
    const unsigned by = blockIdx.y; // 1
    unsigned Bval=0;
#pragma unroll
    for (int i=0; i<32; i++)
    {
        T f0 = A[bx*32+i];
        Bval = (Bval<<1) + (f0>0);
    }
    B[bx] = Bval;
}

//// col-major packing bit 64
//template <typename T>
//__global__ void ToBit64Col(const T* __restrict__ A, ullong* B, const int A_height, const int A_width)
//{
//    GET_LANEID;
//    const unsigned by = blockIdx.y;
//    const unsigned bx = blockIdx.x;
//    ullong Bval;
//#pragma unroll
//    for (int i=0; i<32; i++)
//    {
//        T f0 = A[(bx*32+i)*A_width+by*64+laneid];
//        T f1 = A[(bx*32+i)*A_width+by*64+32+laneid];
//        unsigned r0 = __ballot(f0>0);
//        unsigned r1 = __ballot(f1>0);
//        ullong l0;
//        asm volatile("mov.b64 %0, {%1,%2};":"=l"(l0):"r"(r0),"r"(r1)); //lo,hi
//        if (laneid == i) Bval = __brevll(l0);
//    }
//    B[by*A_height+bx*32+laneid] = Bval;
//}
//
//// row-major packing bit 64
//template <typename T>
//__global__ void ToBit64Row(const T* __restrict__  A, ullong* B,
//        const int A_height, const int A_width)
//{
//    GET_LANEID;
//    const unsigned bx = blockIdx.x;
//    const unsigned by = blockIdx.y;
//    ullong Bval = 0;
//#pragma unroll
//    for (int i=0; i<64; i++)
//    {
//        T f0 = A[(bx*64+i)*A_width+by*32+laneid];
//        Bval = (Bval<<1) | (f0>0?1:0);
//    }
//    B[bx*A_width+by*32+laneid] = Bval;
//}

//// the row-by-row model
//// dense bmm
//template <typename T>
//__device__ __inline__ void bmm32_dense(MulAParam* p)
//{
//    GET_LANEID;
//
//    unsigned A_height = FEIL(p->weight_height);
//    unsigned A_width = CEIL(p->weight_width);
//    unsigned B_width = FEIL(p->input_width);
//
//    // load
//    const unsigned* Asub = &(p->input_gpu[blockIdx.x*32]);
//    const unsigned* Bsub = &(p->weight_gpu[blockIdx.y*32]);
//    T* Csub = &(p->output_gpu[blockIdx.x*B_width*32+blockIdx.y*32]);
//    register unsigned Cm[32] = {0}; // allocate 32 register for each lane
//
//    // compute
//    for (int i=0; i<A_width; i++) {
//        unsigned r0 = Asub[i*A_height+laneid];
//        unsigned r1 = Bsub[i*B_width+laneid];
//
//        #pragma unroll
//        for (int j=0; j<32; j++)
//        {
//            unsigned r2 = __shfl(r1, j); // broadcast from lane-j
//            Cm[j] += __popc(r0 & r2);
//        }
//    }
//
//    // store
//    for (int i=0; i<32; i++)
//        Csub[laneid*B_width+i] = (T)(Cm[i]>0?1:0);
//
//    // more efficient store
////    for (int i=0; i<32; i+=4)
////        store128((void*)&Csub[laneid*B_width+i],
////                    (T)(Cm[i+0]>0?1:0), (T)(Cm[i+1]>0?1:0),
////                    (T)(Cm[i+2]>0?1:0), (T)(Cm[i+3]>0?1:0));
//
//}
//
//// dense bmv
//template <typename T>
//__device__ __inline__ void bmv32_dense(MulAParam* p)
//{
//    GET_LANEID;
//
//    unsigned A_height = FEIL(p->weight_height);
//    unsigned A_width = CEIL(p->weight_width);
//    unsigned B_width = FEIL(p->input_width);
//
//    // load
//    const unsigned* Asub = &(p->input_gpu[blockIdx.x*32]);
//    const unsigned* Bsub = &(p->weight_gpu[0]);
//    T* Csub = &(p->output_gpu[blockIdx.x*B_width*32+blockIdx.y*32]);
//    register unsigned Cm[1] = {0}; // allocate 1 register
//
//    // compute
//    for (int i=0; i<A_width; i++) {
//        unsigned r0 = Asub[i*A_height+laneid];
//        unsigned r1 = Bsub[i*B_width]; // only the lane-0 is required
//
//        Cm[0] += __popc(r0 & r1);
//    }
//
//    // store
//    Csub[laneid*B_width] = (T)(Cm[0]>0?1:0);
//}
//

//// bsr bmm
//template <typename T>
//__device__ __inline__ void bmm32_sparse(MulAParam* p) { // A (bsr matrix) * B (dense matrix) = C (dense matrix)
//
//    if (blockIdx.x < p->bsr_nblockrow+1) {
//        GET_LANEID;
//
//        unsigned A_height = FEIL(p->weight_height); // 32 <-- not in use
//        unsigned A_width = CEIL(p->weight_width); // 32*nblocks / 32 = nblocks <-- not in use
//        unsigned B_width = FEIL(p->input_width); // can be 1 (feil to 32) or any possible output_width
//
//        // load
//        unsigned row_start = p->bsr_rowptr_gpu[blockIdx.x]; // 0 32 64 . . . 991
//        unsigned row_end = p->bsr_rowptr_gpu[blockIdx.x+1]; // 32 64 96 . . . 991 1022
//        const unsigned* Asub = &(p->weight_gpu[row_start]); // block is in continuous layout
//        const unsigned* Bsub = &(p->input_gpu[blockIdx.y*32]);
//        T* Csub = &(p->output_gpu[blockIdx.x*B_width*32+blockIdx.y*32]);
//        register unsigned Cm[32] = {0}; // allocate 32 register for each lane at B
//
//        // compute
//        // if that row has more than 1 col block
//        for (int i=row_start; i<row_end; i++) {
//            unsigned r0 = Asub[i*32+laneid]; // block is in continuous layout
//            unsigned r1 = Bsub[(p->bsr_colind_gpu[i])*B_width+laneid];
//
//            #pragma unroll
//            for (int j=0; j<32; j++)
//            {
//                unsigned r2 = __shfl(r1, j); // broadcast from lane-j
//                Cm[j] += __popc(r0 & r2);
//            }
//        }
//
//        // store
////        for (int i=0; i<32; i++)
////            Csub[laneid*B_width+i] = (T)(Cm[i]>0?1:0);
//
//        // more efficient store
//        for (int i=0; i<32; i+=4)
//            store128((void*)&Csub[laneid*B_width+i],
//                        (T)(Cm[i+0]>0?1:0), (T)(Cm[i+1]>0?1:0),
//                        (T)(Cm[i+2]>0?1:0), (T)(Cm[i+3]>0?1:0));
//
//    }
//}

// bsr bmv32 no padding
// A (bsr matrix) * B (vector) = C (vector)
// col-bin(32 x (blocksize x nblocks)) * row-bin((nblockrows x nblocks) x 1) = (nblockrow x nblocks) x 1
template <typename Index, typename T>
__global__ void bmv32_sparse(const unsigned* __restrict__ A, const unsigned* __restrict__ B,
            T* C, const int A_height, const int A_width, const int B_width,
            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
            const Index nblockrows, const Index nblocks)
{
    GET_LANEID;

    const unsigned bx = blockIdx.x; // 1
    const unsigned by = blockIdx.y; // nblockrow
//    if (by == 0 && laneid == 0) printf("!!!!!");
//    if (by == 0 && laneid == 0) {
//        for(int i=0; i<nblockrows+1; i++) printf("%d ", rowptr[i]);
//        printf("\n");
//        for(int i=0; i<nblocks; i++) printf("%d ", colind[i]);
//        printf("\n");
//        for(int j=0; j<32; j++) { for(unsigned i = 1 << 31; i > 0; i = i / 2)
//        { (A[32*0+j]&i)?printf("1"):printf("0"); } printf("\n"); }
//    }

    // load
    int row_start = rowptr[blockIdx.y]; // 0 32 64 . . . 991
    int row_end = rowptr[blockIdx.y+1]; // 32 64 96 . . . 991 1022

    const unsigned* Asub = &(A[row_start*32]); // block is in continuous layout
    const unsigned* Bsub = &(B[0]); // 0, when it is mv
    T* Csub = &(C[blockIdx.y*32]);
    register unsigned Cm[1] = {0}; // allocate 1 register

    // compute
    // if that row has more than 1 col block
    for (int i=row_start; i<row_end; i++) {
        Cm[0] = 0;
        unsigned r0 = Asub[(i-row_start)*32+laneid]; // block is in continuous layout
        unsigned r1 = Bsub[(colind[i])]; // only first row is required

        Cm[0] += __popc(r0 & r1);
        // store
        Csub[laneid] += (T)(Cm[0]); //Csub[laneid] = (T)(Cm[0]>0);
    }
}

// haven't test
// bsr bmv64 no padding
// A (bsr matrix) * B (vector) = C (vector)
template <typename Index, typename T>
__global__ void bmv64_sparse(const ullong* __restrict__ A, const ullong* __restrict__ B,
                            T* C, const int A_height, const int A_width, const int B_width,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks)
{
    if (blockIdx.x < nblockrows + 1) {
        GET_LANEID;

        // load
        unsigned row_start = rowptr[blockIdx.x]; // not correct
        unsigned row_end = rowptr[blockIdx.x+1];
        const ullong* Asub = &(A[row_start]);
        const ullong* Bsub = &(B[0]);
        T* Csub = &(C[blockIdx.x*B_width*64]);
        register unsigned Cm[2] = {0};

        // compute
        for (int i=row_start; i<row_end; i++) {
            ullong a0 = Asub[i*A_height+laneid];
            ullong a1 = Asub[i*A_height+32+laneid];
            ullong b0 = Bsub[(colind[i])*B_width];

            Cm[0] += (__popcll(a0 & b0)<<16) + __popcll(a1 & b0);
        }

        // store
        short t0,t1,t2,t3;
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[0]));
        asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t3),"=h"(t2):"r"(Cm[1]));
        store64(&Csub[laneid*B_width], (T)(t0>0?1:0), (T)(t2>0?1:0));
        store64(&Csub[(laneid+32)*B_width], (T)(t1>0?1:0), (T)(t3>0?1:0));
    }
}