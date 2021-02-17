#ifndef BMM64_CUH
#define BMM64_CUH

// treat continous block as batch input so that they can be considered together (later)
//
__device__ __inline__ void MatMul64(MulAParam64* p)
{
    GET_LANEID;
    const int gdx = CEIL64(p->input_height); //vertical
    const int gdy = CEIL64(p->weight_width); //horizontal
    for (int bid=blockIdx.x; bid<gdx*gdy; bid+=gridDim.x)
    {
        int bx = bid / gdy;
        int by = bid % gdy;
        const ullong* input_sub = &(p->input_gpu[bx*64]);
        const ullong* weight_sub = &(p->weight_gpu[by*64]);
        ullong* output_sub = &(p->output_gpu[by*gdx*64+bx*64]);
        register unsigned Cm[64] = {0};
        for (int i=0; (i*64) < (p->input_width); i++)
        {
            ullong a0 = input_sub[i*64*gdx+laneid];
            ullong a1 = input_sub[i*64*gdx+32+laneid];
            ullong b0 = weight_sub[i*64*gdy+laneid];
            ullong b1 = weight_sub[i*64*gdy+32+laneid];
            for (int j=0; j<32; j++)
            {
                ullong l0 = __shfl(b0,j);
                ullong l1 = __shfl(b1,j);
                Cm[j] += (__popcll(a0 & l0)<<16) + __popcll(a1 & l0); // modified
                Cm[32+j] += (__popcll(a0 & l1)<<16) + __popcll(a1 & l1); // modified
            }
        }
        ullong C0 = 0;
        ullong C1 = 0;
        for (int i=0; i<64; i++)
        {
            //if (by*64+i<(p->weight_width)) //required when matrix size cannot divide 64
            {
                short t0,t1;
                asm volatile("mov.b32 {%0,%1}, %2;":"=h"(t1),"=h"(t0):"r"(Cm[i])); //lo, hi
                //if (bx*64+laneid<(p->input_height))
                {
                    C0 |= (((float)t0 > 0?(ullong)1:(ullong)0)<<(63-i));
                }
                //if (bx*64+32+laneid<(p->input_height))
                {
                    C1 |= (((float)t1 > 0?(ullong)1:(ullong)0)<<(63-i));
                }
            }
        }
        output_sub[laneid] = C0;
        output_sub[laneid+32] = C1;
    }
}

#endif