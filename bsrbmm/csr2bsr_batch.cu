#include "bsrbmm.cu"
#include "utility.cu"

/**
* batch the process of csr2bsr, blocksize = 32
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_32_row(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, unsigned* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n"); printHostIndArr<int>(h_csrRowPtr, (nrows+1));
    printf("h_csrColInd:\n"); printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<int><<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, unsigned><<<1,1>>>(bsrVal, nblocks*blocksize, 0);
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
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
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
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void**)&temp_csrrowptr, sizeof(int) * (32+1));
            if (i == nblockrows-1) { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * ((nrows+1)-(i*32)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), (32+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * (32+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (32+1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
             printf("temp_csrrowptr: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (32+1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
             printf("temp_csrcolind: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1,1>>>(temp_csrval, temp_nnz, 1.0);

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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
#endif
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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k; std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

#ifdef DEBUG
    printout global bsr to verify
    printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock32<<<1,1>>>(bsrVal, blocksize, nblocks);
#endif
}

void csr2bsr_batch_32_col(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, unsigned* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n"); printHostIndArr<int>(h_csrRowPtr, (nrows+1));
    printf("h_csrColInd:\n"); printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<int><<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, unsigned><<<1,1>>>(bsrVal, nblocks*blocksize, 0);
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
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
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
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*32)), (32+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*32, sizeof(int) * (32+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (32+1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
             printf("temp_csrrowptr: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (32+1));
#endif

            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
             printf("temp_csrcolind: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrcolind, temp_nnz);
#endif

            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1,1>>>(temp_csrval, temp_nnz, 1.0);

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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
#endif
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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k; std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

#ifdef DEBUG
    printout global bsr to verify
    printGlobalBSR32<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock32<<<1,1>>>(bsrVal, blocksize, nblocks);
#endif
}

/**
* batch the process of csr2bsr, blocksize = 64
* assume csr val are only 0 or 1
*/
void csr2bsr_batch_64_row(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, ullong* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n"); printHostIndArr<int>(h_csrRowPtr, (nrows+1));
    printf("h_csrColInd:\n"); printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<int><<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, ullong><<<1,1>>>(bsrVal, nblocks*blocksize, 0);
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
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
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
            cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

            /* **************************************** */
            // set buffer csr storage
            // 1) set buffer csr rowptr & offset
            cudaMalloc((void**)&temp_csrrowptr, sizeof(int) * (64+1));
            if (i == nblockrows-1) { // last iteration
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*64, sizeof(int) * ((nrows+1)-(i*64)), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), (64+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*64, sizeof(int) * (64+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (64+1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
             printf("temp_csrrowptr: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (64+1));
#endif
            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
             printf("temp_csrcolind: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrcolind, temp_nnz);
#endif
            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1,1>>>(temp_csrval, temp_nnz, 1.0);

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
#ifdef DEBUG
            printTempBSRVal<<<1,1>>>(temp_bsrval, blocksize, temp_nblocks);
#endif
            cudaMalloc((void**)&temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize);
            ToBit64Col<float><<<dim3(2, temp_nblocks), 32>>>(temp_bsrval,
                                                temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
#endif
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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k; std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

#ifdef DEBUG
    // printout global bsr to verify
    printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock64<<<1,1>>>(bsrVal, blocksize, nblocks);
#endif DEBUG

}

void csr2bsr_batch_64_col(const int* h_csrRowPtr, const int* h_csrColInd,
                    const int nrows, const int ncols, const int nnz,
                    int* bsrRowPtr, int* bsrColInd, ullong* bsrVal,
                    const int blocksize, const int nblockrows, const int nblocks)
{
#ifdef DEBUG
    // check h_csr
    printf("h_csrRowPtr:\n"); printHostIndArr<int>(h_csrRowPtr, (nrows+1));
    printf("h_csrColInd:\n"); printHostIndArr<int>(h_csrColInd, nnz);
#endif

    // global result
    setDeviceIndArr<int><<<1,1>>>(bsrRowPtr, (nblockrows+1), 0);
    setDeviceIndArr<int><<<1,1>>>(bsrColInd, nblocks, 0);
    setDeviceValArr<int, ullong><<<1,1>>>(bsrVal, nblocks*blocksize, 0);
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
#ifdef DEBUG
        printf("[blockrow %d] temp_nnz: %d, temp_rowstart: %d, temp_rowend: %d\n", i, temp_nnz, temp_rowstart, temp_rowend);
#endif
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
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), temp_rowstart); // offset rowptr
                padDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, ((nrows+1)-(i*64)), (64+1), temp_nnz);
            }
            else { // all other iteration except last
                cudaMemcpy(temp_csrrowptr, h_csrRowPtr+i*64, sizeof(int) * (64+1), cudaMemcpyHostToDevice);
                offsetDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (64+1), temp_rowstart); // offset rowptr
            }
#ifdef DEBUG
             printf("temp_csrrowptr: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrrowptr, (64+1));
#endif
            // 2) set buffer csr colind
            cudaMalloc((void**)&temp_csrcolind, sizeof(int) * temp_nnz);
            cudaMemcpy(temp_csrcolind, h_csrColInd+temp_rowstart, sizeof(int) * temp_nnz, cudaMemcpyHostToDevice);
#ifdef DEBUG
             printf("temp_csrcolind: \n"); printDeviceIndArr<int><<<1,1>>>(temp_csrcolind, temp_nnz);
#endif
            // 3) set buffer csr val
            cudaMalloc((void**)&temp_csrval, sizeof(float) * temp_nnz);
            setDeviceValArr<int, float><<<1,1>>>(temp_csrval, temp_nnz, 1.0);

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
#ifdef DEBUG
            printTempBSRVal<<<1,1>>>(temp_bsrval, blocksize, temp_nblocks);
#endif
            cudaMalloc((void**)&temp_bsrval_packed, sizeof(ullong) * temp_nblocks * blocksize);
            ToBit64Col<float><<<dim3(2, temp_nblocks), 32>>>(temp_bsrval,
                                                temp_bsrval_packed, blocksize, temp_nblocks * blocksize);

            // concat to global bsr result
            int last_bsrrowind;
            cudaMemcpy(&last_bsrrowind, bsrRowPtr+i, sizeof(int) * 1, cudaMemcpyDeviceToHost);
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind+temp_nblocks); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind+temp_nblocks);
#endif
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
            setDeviceIndArrElem<int><<<1,1>>>(bsrRowPtr, (i+1), last_bsrrowind); // add on offset
#ifdef DEBUG
            printf("set global bsrRowPtr[%d] = %d\n", (i+1), last_bsrrowind);
#endif

        } // if (temp_nnz != 0)
#ifdef DEBUG
        // printout global bsr to verify
        printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
        int k; std::cin >> k;
#endif

    } // for (i < nblockrows)

    // final check
    if (total_nblocks != nblocks)
        printf("[fail] total nblocks %d do not match nblocks %d!\n", total_nblocks, nblocks);
    else printf("[success] total nblocks %d match nblocks %d!\n", total_nblocks, nblocks);

#ifdef DEBUG
    // printout global bsr to verify
    printGlobalBSR64<<<1,1>>>(bsrRowPtr, bsrColInd, bsrVal, blocksize, nblockrows, nblocks);
    printGlobalBSRBlock64<<<1,1>>>(bsrVal, blocksize, nblocks);
#endif DEBUG

}