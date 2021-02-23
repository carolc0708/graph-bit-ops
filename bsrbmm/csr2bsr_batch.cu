#include <cusparse_v2.h>
#include <iostream>
#include "bsrbmm.cu"

const int MAX_SIZE = 10;
typedef char carr[MAX_SIZE];

/* Error Checking for cuSparse library */
#define CHECK_CUSPARSE( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )

inline void __cusparseSafeCall( cusparseStatus_t err, const char *file, const int line )
{
#ifdef CUDA_ERROR_CHECK
    if ( CUSPARSE_STATUS_SUCCESS != err )
    {
        fprintf(stderr, "CUSPARSE API failed at %s:%i : %d\n",
                 file, line, err);
        exit(-1);
    }
#endif
    return;
}

void printHostIndArr(const int* indarr, const int N)
{
    for(int i=0; i<N; i++) printf("[%d]%d ", i, indarr[i]);
    printf("\n");
}

__global__ void printDeviceIndArr(const int* indarr, const int N)
{
    for(int i=0; i<N; i++) printf("%d ", indarr[i]);
    printf("\n");
}

__global__ void printGlobalBSR32(const int* bsrrowptr, const int* bsrcolind, const unsigned* bsrval,
                               const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr --- \n");
    printf("bsrrowptr: \n"); for(int i=0; i<(nblockrows+1); i++) { printf("%d ", bsrrowptr[i]); } printf("\n");
    printf("bsrcolind: \n"); for(int i=0; i<nblocks; i++) { printf("%d ", bsrcolind[i]); } printf("\n");
    printf("bsrval: \n");
    printf("[0] "); for(int j=0; j<blocksize; j++) { for(unsigned i = 1 << 31; i > 0; i = i / 2)
    { (bsrval[0*blocksize+j]&i)?printf("1"):printf("0"); } printf(" "); } printf("\n");
    printf("[%d] ", nblocks-1); for(int j=0; j<blocksize; j++) { for(unsigned i = 1 << 31; i > 0; i = i / 2)
    { (bsrval[(nblocks-1)*blocksize+j]&i)?printf("1"):printf("0"); } printf(" "); } printf("\n");
}

__global__ void printGlobalBSRBlock32(const unsigned* bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr block (bitmap) --- \n");
    for(int b=0; b<nblocks; b++) {
        printf("[%d]\n", b); for(int j=0; j<blocksize; j++) { for(unsigned i = 1 << 31; i > 0; i = i / 2)
        { (bsrval[b*blocksize+j]&i)?printf("1"):printf("0"); } printf("\n"); }
    }
}

__global__ void printTempBSRVal(const float* bsrval, const int blocksize, const int nblocks)
{
    printf("TempBSRVal: \n");
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            for(int k=0; k<blocksize; k++) {
                printf(bsrval[i*blocksize*blocksize+j*blocksize+k]>0?"1":"0");
            }
            printf("\n");
        }
        printf("\n");
    }
}

__global__ void setDeviceIndArr(int *arr, const int N, const int val)
{
    for (int i=0; i<N; i++) arr[i] = val;
}

__global__ void setDeviceIndArrElem(int *arr, const int ind, const int val)
{
    arr[ind] = val;
}

__global__ void setDeviceValArr(float *arr, const int N, const float val)
{
    for (int i=0; i<N; i++) arr[i] = val;
}

__global__ void setDeviceValArr(int *arr, const int N, const int val)
{
    for (int i=0; i<N; i++) arr[i] = val;
}

__global__ void setDeviceValArr(unsigned *arr, const int N, const unsigned val)
{
    for (int i=0; i<N; i++) arr[i] = val;
}

__global__ void setDeviceValArr(ullong *arr, const int N, const ullong val)
{
    for (int i=0; i<N; i++) arr[i] = val;
}

__global__ void offsetDeviceIndArr(int* indarr, const int N, const int temp_rowstart)
{
    for(int i=0; i<N; i++) indarr[i] -= temp_rowstart;
}

__global__ void padDeviceIndArr(int* indarr, const int startind, const int endind, const int temp_nnz)
{
    for(int i=startind; i<endind; i++) indarr[i] = temp_nnz;
}

/**
* batch the process of csr2bsr, blocksize = 32
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_32(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, unsigned* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
    // check h_csr
//    printf("h_csrRowPtr:\n"); printHostIndArr(h_csrRowPtr, (nrows+1));
//    printf("h_csrColInd:\n"); printHostIndArr(h_csrColInd, nnz);

    // global result
    setDeviceIndArr<<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<<<1,1>>>(bsrVal, nblocks*blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    unsigned *temp_bsrval_packed;

    for(int i=0; i<nblockrows; i++) { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i+1)*32<nrows?h_csrRowPtr[(i+1)*32]:nnz), temp_rowstart = h_csrRowPtr[i*32];
        int temp_nnz = temp_rowend - temp_rowstart;
//        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&csr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO) );
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&bsr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO) );

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE( cusparseCreate(&handle) );
            CHECK_CUSPARSE( cusparseSetStream(handle, streamId) );

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void**)&temp_csrrowptr, sizeof(int) * (32+1));
            if (i == nblockrows-1) { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * ((nrows+1)-(i*32)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), (32+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * (32+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, (32+1), temp_rowstart); // offset rowptr
            }
            // printf("temp_csrrowptr: \n"); printDeviceIndArr<<<1,1>>>(temp_csrrowptr, (32+1));

            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
            // printf("temp_csrcolind: \n"); printDeviceIndArr<<<1,1>>>(temp_csrcolind, temp_nnz);

            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<<<1,1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void**)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE( cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                temp_bsrrowptr, &temp_nblocks) );
            cudaMalloc((void**)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void**)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE( cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind) );
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void**)&temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize);
            ToBit32Col<float><<<dim3(1, temp_nblocks), 32>>>(temp_bsrval,
                                                temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
            cudaMemcpy(bsrColInd+last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal+last_bsrrowind*blocksize, temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr); temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind); temp_csrcolind = NULL;
            cudaFree(temp_csrval); temp_csrval = NULL;
            cudaFree(temp_bsrrowptr); temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind); temp_bsrcolind = NULL;
            cudaFree(temp_bsrval); temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed); temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE( cusparseDestroyMatDescr(csr_descr) );
            CHECK_CUSPARSE( cusparseDestroyMatDescr(bsr_descr) );
            CHECK_CUSPARSE( cusparseDestroy(handle) );

        } else { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);

        } // if (temp_nnz != 0)
//        // printout global bsr to verify
//        printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//        int k; std::cin >> k;

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

    // printout global bsr to verify
//    printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//    printGlobalBSRBlock32<<<1,1>>>(bsrVal, blocksize, nblocks);
}

void csc2bsr_batch_32(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, unsigned* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
    // check h_csr
//    printf("h_csrRowPtr:\n"); printHostIndArr(h_csrRowPtr, (nrows+1));
//    printf("h_csrColInd:\n"); printHostIndArr(h_csrColInd, nnz);

    // global result
    setDeviceIndArr<<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<<<1,1>>>(bsrVal, nblocks*blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    unsigned *temp_bsrval_packed;

    for(int i=0; i<nblockrows; i++) { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i+1)*32<nrows?h_csrRowPtr[(i+1)*32]:nnz), temp_rowstart = h_csrRowPtr[i*32];
        int temp_nnz = temp_rowend - temp_rowstart;
//        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&csr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO) );
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&bsr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO) );

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE( cusparseCreate(&handle) );
            CHECK_CUSPARSE( cusparseSetStream(handle, streamId) );

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void**)&temp_csrrowptr, sizeof(int) * (32+1));
            if (i == nblockrows-1) { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * ((nrows+1)-(i*32)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), (32+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * (32+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, (32+1), temp_rowstart); // offset rowptr
            }
            // printf("temp_csrrowptr: \n"); printDeviceIndArr<<<1,1>>>(temp_csrrowptr, (32+1));

            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
            // printf("temp_csrcolind: \n"); printDeviceIndArr<<<1,1>>>(temp_csrcolind, temp_nnz);

            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<<<1,1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void**)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE( cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                temp_bsrrowptr, &temp_nblocks) );
            cudaMalloc((void**)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void**)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE( cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind) );
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
            cudaMalloc((void**)&temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize);
            ToBit32Col<float><<<dim3(1, temp_nblocks), 32>>>(temp_bsrval,
                                                temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
            cudaMemcpy(bsrColInd+last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal+last_bsrrowind*blocksize, temp_bsrval_packed, sizeof(unsigned) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr); temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind); temp_csrcolind = NULL;
            cudaFree(temp_csrval); temp_csrval = NULL;
            cudaFree(temp_bsrrowptr); temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind); temp_bsrcolind = NULL;
            cudaFree(temp_bsrval); temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed); temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE( cusparseDestroyMatDescr(csr_descr) );
            CHECK_CUSPARSE( cusparseDestroyMatDescr(bsr_descr) );
            CHECK_CUSPARSE( cusparseDestroy(handle) );

        } else { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);

        } // if (temp_nnz != 0)
//        // printout global bsr to verify
//        printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//        int k; std::cin >> k;

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

    // printout global bsr to verify
//    printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//    printGlobalBSRBlock32<<<1,1>>>(bsrVal, blocksize, nblocks);
}

/**
* batch the process of csr2bsr, blocksize = 64
* assume csr val are only 0 or 1
*/
// still should change to ullong
void csr2bsr_batch_64(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, ullong* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
    // check h_csr
//    printf("h_csrRowPtr:\n"); printHostIndArr(h_csrRowPtr, (nrows+1));
//    printf("h_csrColInd:\n"); printHostIndArr(h_csrColInd, nnz);

    // global result
    setDeviceIndArr<<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<<<1,1>>>(bsrVal, nblocks*blocksize, 0);
    int total_nblocks = 0;

    // buffer storage
    int *temp_csrrowptr, *temp_csrcolind;
    float *temp_csrval;
    int *temp_bsrrowptr, *temp_bsrcolind;
    float *temp_bsrval;
    ullong *temp_bsrval_packed;

    for(int i=0; i<nblockrows; i++) { // serial processing, 1 blockrow at a time

        // calculate nnz in this blockrow
        int temp_rowend = ((i+1)*64<nrows?h_csrRowPtr[(i+1)*64]:nnz), temp_rowstart = h_csrRowPtr[i*64];
        int temp_nnz = temp_rowend - temp_rowstart;
//        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);

        if (temp_nnz != 0)
        {
            // set up cusparse metadata
            // 1) create cusparsematdescr for csr, bsr
            cusparseMatDescr_t csr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&csr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(csr_descr,CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(csr_descr,CUSPARSE_INDEX_BASE_ZERO) );
            cusparseMatDescr_t bsr_descr = 0;
            CHECK_CUSPARSE( cusparseCreateMatDescr(&bsr_descr) );
            CHECK_CUSPARSE( cusparseSetMatType(bsr_descr, CUSPARSE_MATRIX_TYPE_GENERAL) );
            CHECK_CUSPARSE( cusparseSetMatIndexBase(bsr_descr, CUSPARSE_INDEX_BASE_ZERO) );

            // 2) cusparse handle, stream
            cudaStream_t streamId = 0;
            cusparseHandle_t handle = 0;
            CHECK_CUSPARSE( cusparseCreate(&handle) );
            CHECK_CUSPARSE( cusparseSetStream(handle, streamId) );

            // 3) cusparse direction
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_COLUMN;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void**)&temp_csrrowptr, sizeof(int) * (64+1));
            if (i == nblockrows-1) { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*64, sizeof(int) * ((nrows+1)-(i*64)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), temp_rowstart); // offset rowptr
                padDeviceIndArr<<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), (64+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*64, sizeof(int) * (64+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<<<1,1>>>(temp_csrrowptr, (64+1), temp_rowstart); // offset rowptr
            }
            // printf("temp_csrrowptr: \n"); printDeviceIndArr<<<1,1>>>(temp_csrrowptr, (64+1));

            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
            // printf("temp_csrcolind: \n"); printDeviceIndArr<<<1,1>>>(temp_csrcolind, temp_nnz);

            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<<<1,1>>>(temp_csrval, temp_nnz, 1.0);

            // calculate nnzb & allocate buffer bsr storage
            int temp_nblocks;
            cudaMalloc((void**)&temp_bsrrowptr, sizeof(int) * 2);
            CHECK_CUSPARSE( cusparseXcsr2bsrNnz(handle, dirA, blocksize, ncols, csr_descr,
                                temp_csrrowptr, temp_csrcolind, blocksize, bsr_descr,
                                temp_bsrrowptr, &temp_nblocks) );
            cudaMalloc((void**)&temp_bsrcolind, sizeof(int) * temp_nblocks);
            cudaMalloc((void**)&temp_bsrval, sizeof(float) * temp_nblocks * blocksize * blocksize);

            // csr2bsr (nrows = blocksize, ncols = ncols)
            CHECK_CUSPARSE( cusparseScsr2bsr(handle, dirA, blocksize, ncols, csr_descr,
                            temp_csrval, temp_csrrowptr, temp_csrcolind,
                            blocksize, bsr_descr, temp_bsrval, temp_bsrrowptr, temp_bsrcolind) );
            if (i % 1000 == 0)
                printf("current total_nblocks: %d, temp_nblocks: %d\n", total_nblocks, temp_nblocks);
            total_nblocks += temp_nblocks;

            // pack buffer bsrval to binary
//            printTempBSRVal<<<1,1>>>(temp_bsrval, blocksize, temp_nblocks);
            cudaMalloc((void**)&temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize);
            ToBit64Col<float><<<dim3(2, temp_nblocks), 32>>>(temp_bsrval,
                                                temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
            cudaMemcpy(bsrColInd+last_bsrrowind, temp_bsrcolind, sizeof(int) * temp_nblocks, cudaMemcpyDeviceToDevice);
            cudaMemcpy(bsrVal+last_bsrrowind*blocksize, temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize, cudaMemcpyDeviceToDevice);

            // clean buffer
            cudaFree(temp_csrrowptr); temp_csrrowptr = NULL;
            cudaFree(temp_csrcolind); temp_csrcolind = NULL;
            cudaFree(temp_csrval); temp_csrval = NULL;
            cudaFree(temp_bsrrowptr); temp_bsrrowptr = NULL;
            cudaFree(temp_bsrcolind); temp_bsrcolind = NULL;
            cudaFree(temp_bsrval); temp_bsrval = NULL;
            cudaFree(temp_bsrval_packed); temp_bsrval_packed = NULL;

            // free descr and handle memory
            CHECK_CUSPARSE( cusparseDestroyMatDescr(csr_descr) );
            CHECK_CUSPARSE( cusparseDestroyMatDescr(bsr_descr) );
            CHECK_CUSPARSE( cusparseDestroy(handle) );

        } else { // only update global bsr's rowptr
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
//            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);

        } // if (temp_nnz != 0)
//        // printout global bsr to verify
//        printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//        int k; std::cin >> k;

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

    // printout global bsr to verify
//    printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
//    printGlobalBSRBlock64<<<1,1>>>(bsrVal, blocksize, nblocks);
}



