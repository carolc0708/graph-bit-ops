#ifndef BMM32_CUH
#define BMM32_CUH

// the row-by-row model
__device__ __inline__ void MatMul(MulAParam* p)
{
    GET_LANEID;
    const int gdx = (CEIL(p->input_height));
    const int gdy = (CEIL(p->weight_width));
    for (int bid=blockIdx.x*32+warpid; bid<gdx*gdy; bid+=gridDim.x*32) // can be seen as a 32x32 tile
    {
        // BMM
        unsigned bx = bid / gdy;
        unsigned by = bid % gdy;
        const unsigned* input_sub = &(p->input_gpu[bx*32]);
        const unsigned* weight_sub = &(p->weight_gpu[by*32]); // each col of weight (A)
        unsigned* output_sub = &(p->output_gpu[by*gdx*32+bx*32]);

        register int Cm[32] = {0};
        for (int i=0; (i*32)<(p->input_width); i++)
        {
            unsigned r0 = input_sub[i*32*gdx+laneid];
            unsigned r1 = weight_sub[i*32*gdy+laneid];

            #pragma unroll
            for (int j=0; j<32; j++)
            {
                unsigned r2 = __shfl(r1, j); //from lane-j, r1 of weight matrix
                Cm[j] += __popc(r0 & r2);
            }
        }

        unsigned C = 0;
        if ((bx*32+laneid)<(p->input_height))
        {
            for (int i=0; i<32; i++)
            {
                if (by*32+i<(p->weight_width))
                {
                    C = (C << 1) | (Cm[i] > 0);
                }
            }
        }
        output_sub[laneid] = C;
    }
}

#endif