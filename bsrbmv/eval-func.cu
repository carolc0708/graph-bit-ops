#include <stdio.h>
#include <assert.h>

//======================================================================================
// count
//======================================================================================
template <typename Index, typename T>
__global__ void printBlockReport(const T* __restrict__ A,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks, const Index blocksize)
{
    Index maxnbpbr = -1, minnbpbr = 2147483647;
    Index sumnnztilerow = 0;
    for(int i=0; i<nblockrows; i++) {
        Index row_start = rowptr[i];
        Index row_end = rowptr[i+1];
        Index nbpbr = row_end - row_start;
//        load[i] = nbpbr;
        maxnbpbr = max(maxnbpbr, nbpbr);
        minnbpbr = min(nbpbr, minnbpbr);

        for(int nb=0; nb<nbpbr; nb++) {
            for(int j=0; j<blocksize; j++) {
                sumnnztilerow += (Index)(__popc((unsigned)(A[(row_start+nb)*blocksize+j])) > 0);
            }
        }

    }

    float avgnnztilerow = (float)sumnnztilerow / (nblocks * blocksize);

    printf("----------------------------------------\n");
    printf("Avg nnz-tile-row: %.4f\n", avgnnztilerow);
    printf("Max nblocks per blockrow: %d\n", (int)maxnbpbr);
    printf("Min nblocks per blockrow: %d\n", (int)minnbpbr);

    printf("----------------------------------------\n");
}

template <typename Index, typename T>
__global__ void printBlockReportUllong(const T* __restrict__ A,
                            const Index* __restrict__ rowptr, const Index* __restrict__ colind,
                            const Index nblockrows, const Index nblocks, const Index blocksize)
{
    Index maxnbpbr = -1, minnbpbr = 2147483647;
    Index sumnnztilerow = 0;
    for(int i=0; i<nblockrows; i++) {
        Index row_start = rowptr[i];
        Index row_end = rowptr[i+1];
        Index nbpbr = row_end - row_start;
//        load[i] = nbpbr;
        maxnbpbr = max(maxnbpbr, nbpbr);
        minnbpbr = min(nbpbr, minnbpbr);

        for(int nb=0; nb<nbpbr; nb++) {
            for(int j=0; j<blocksize; j++) {
                sumnnztilerow += (Index)(__popcll((A[(row_start+nb)*blocksize+j])) > 0);
            }
        }

    }

    float avgnnztilerow = (float)sumnnztilerow / (nblocks * blocksize);

    printf("----------------------------------------\n");
    printf("Avg nnz-tile-row: %.4f\n", avgnnztilerow);
    printf("Max nblocks per blockrow: %d\n", (int)maxnbpbr);
    printf("Min nblocks per blockrow: %d\n", (int)minnbpbr);

    printf("----------------------------------------\n");
}