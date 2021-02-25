/**
* Filename: utility.cu
*
* Description: utility functions for debugging, profiling on bsrbmv.
*
*/
#include <cusparse_v2.h>
#include <iostream>
#include <sys/time.h>

//======================================================================================
// Error checking for cuda libraries' APIs
//======================================================================================
/**
* Error Checking for cuSparse library
*/
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

//======================================================================================
// Timing functions
//======================================================================================
/**
* The CPU Timer used in GraphBlast
*/
struct CpuTimer {
#if defined(CLOCK_PROCESS_CPUTIME_ID)

    double start;
    double stop;

    void Start() {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        start = tv.tv_sec + 1.e-6*tv.tv_usec;
    }

    void Stop() {
        static struct timeval tv;
        static struct timezone tz;
        gettimeofday(&tv, &tz);
        stop = tv.tv_sec + 1.e-6*tv.tv_usec;
    }

    double ElapsedMillis() {
        return 1000*(stop - start);
    }

#else

    rusage start;
    rusage stop;

    void Start() {
        getrusage(RUSAGE_SELF, &start);
    }

    void Stop() {
        getrusage(RUSAGE_SELF, &stop);
    }

    float ElapsedMillis() {
        float sec = stop.ru_utime.tv_sec - start.ru_utime.tv_sec;
        float usec = stop.ru_utime.tv_usec - start.ru_utime.tv_usec;

        return (sec * 1000) + (usec /1000);
    }

#endif
};

/**
* GPU Timer for better measurement granularity
*/
struct GpuTimer {

    cudaEvent_t start, stop;
    float milliseconds = 0;

    void Start() {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
    }

    void Stop() {
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
    }

    float ElapsedMillis() {
        milliseconds = 0;
        cudaEventElapsedTime(&milliseconds,start,stop);

        return milliseconds;
    }
};

//======================================================================================
// Verification function for result vector
//======================================================================================
/**
* verify bsrbmv result vector with cuSPARSE baseline
*/
template <typename T>
bool checkResult(T* vec1, T* vec2, const int N)
{
    bool flag = true;
    for (int i = 0; i < N; i ++) {
        T diff = vec1[i] - vec2[i];
        if (fabs(diff) > 1e-6) {
//            printf("[%d](%.f,%.f),", i, vec1[i], vec2[i]);
            flag = false;
        }
    }
    return flag;
}

template <typename T>
int countNnzinVec(const T* vec, const int N)
{
    int counter = 0;
    for (int i=0; i<N; i++) if (vec[i] != 0) counter += 1;
    return counter;
}

template <typename T>
__global__ void printResVec(const T* vec, const int N)
{
    for(int i=0; i<N; i++) {printf("%d ", (int)vec[i]); }
    printf("\n");
}

//======================================================================================
// Print function for host (random) vector
//======================================================================================
void printHostVec(const float* vec, const int N)
{
    for(int i=0; i<N; i++) printf(vec[i]>0?"1":"0");
    printf("\n");
}

//======================================================================================
// Print function for binarized vector in device
//======================================================================================
__global__ void printBin32Vec (const unsigned* packvec, const int N)
{
    for(int i=0; i<N; i++) {
        unsigned j;
        for(j = 1 << 31; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
        printf("\n");
    }
    printf("\n");
}

__global__ void printBin32Block (const unsigned* packbsrval, const int nblocks, const int blocksize)
{
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            unsigned k;
            for(k = 1 << 31; k > 0; j = k / 2)
                (packbsrval[i*blocksize+j] & k) ? printf("1") : printf("0");

            printf("\n");
        }
        printf("\n");
    }
}

__global__ void printBin64Vec (const ullong* packvec, const int N)
{
    for(int i=0; i<N; i++) {
        ullong j;
        for(j = 1ULL << 63; j > 0; j = j / 2)
            (packvec[i] & j) ? printf("1") : printf("0");
    }
    printf("\n");
}

__global__ void printBin64Block (const ullong* packbsrval, const int nblocks, const int blocksize)
{
    for(int i=0; i<nblocks; i++) {
        printf("[%d]\n", i);
        for(int j=0; j<blocksize; j++) {
            ullong k;
            for(k = 1ULL << 63; k > 0; k = k / 2)
                (packbsrval[i*blocksize+j] & k) ? printf("1") : printf("0");

            printf("\n");
        }
        printf("\n");
    }
}

//======================================================================================
// Print function for csr and bsr
//======================================================================================
template <typename Index>
void printHostIndArr(const Index* indarr, const Index N)
{
    for(Index i=0; i<N; i++) printf("[%d]%d ", i, indarr[i]);
    printf("\n");
}

template <typename Index>
__global__ void printDeviceIndArr(const Index* indarr, const Index N)
{
    for(Index i=0; i<N; i++) printf("%d ", indarr[i]);
    printf("\n");
}
__global__ void printGlobalBSR32(const int* bsrrowptr, const int* bsrcolind, const unsigned* bsrval,
                               const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 32 --- \n");
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
    printf("--- global bsr 32 block (bitmap) --- \n");
    for(int b=0; b<nblocks; b++) {
        printf("[%d]\n", b); for(int j=0; j<blocksize; j++) { for(unsigned i = 1 << 31; i > 0; i = i / 2)
        { (bsrval[b*blocksize+j]&i)?printf("1"):printf("0"); } printf("\n"); }
    }
}

__global__ void printGlobalBSR64(const int* bsrrowptr, const int* bsrcolind, const ullong* bsrval,
                               const int blocksize, const int nblockrows, const int nblocks)
{
    printf("--- global bsr 64 --- \n");
    printf("bsrrowptr: \n"); for(int i=0; i<(nblockrows+1); i++) { printf("%d ", bsrrowptr[i]); } printf("\n");
    printf("bsrcolind: \n"); for(int i=0; i<nblocks; i++) { printf("%d ", bsrcolind[i]); } printf("\n");
    printf("bsrval: \n");
    printf("[0] "); for(int j=0; j<blocksize; j++) { for(ullong i = 1ULL << 63; i > 0; i = i / 2)
    { (bsrval[0*blocksize+j]&i)?printf("1"):printf("0"); } printf(" "); } printf("\n");
    printf("[%d] ", nblocks-1); for(int j=0; j<blocksize; j++) { for(ullong i = 1ULL << 63; i > 0; i = i / 2)
    { (bsrval[(nblocks-1)*blocksize+j]&i)?printf("1"):printf("0"); } printf(" "); } printf("\n");
}

__global__ void printGlobalBSRBlock64(const ullong* bsrval, const int blocksize, const int nblocks)
{
    printf("--- global bsr 32 block (bitmap) --- \n");
    for(int b=0; b<nblocks; b++) {
        printf("[%d]\n", b); for(int j=0; j<blocksize; j++) { for(ullong i = 1ULL << 63; i > 0; i = i / 2)
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

//======================================================================================
// Set function for device array
//======================================================================================
template <typename Index>
__global__ void setDeviceIndArr(Index *arr, const Index N, const Index val)
{
    for (Index i=0; i<N; i++) arr[i] = val;
}

template <typename Index>
__global__ void setDeviceIndArrElem(Index *arr, const Index ind, const Index val)
{
    arr[ind] = val;
}

template <typename Index>
__global__ void offsetDeviceIndArr(Index* indarr, const Index N, const Index temp_rowstart)
{
    for(Index i=0; i<N; i++) indarr[i] -= temp_rowstart;
}

template <typename Index>
__global__ void padDeviceIndArr(Index* indarr, const Index startind, const Index endind, const Index temp_nnz)
{
    for(Index i=startind; i<endind; i++) indarr[i] = temp_nnz;
}

template <typename Index, typename T>
__global__ void setDeviceValArr(T *arr, const Index N, const T val)
{
    for (Index i=0; i<N; i++) arr[i] = val;
}
