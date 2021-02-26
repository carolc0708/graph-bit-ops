CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

all: bsrbmm32 bsrbmm64

# default
bsrbmm: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmm.cu -o $@

# default: use gpu timer
bsrbmm32: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=32 benchmark-bsrbmm.cu -o $@

bsrbmm64: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=64 benchmark-bsrbmm.cu -o $@

clean:
	rm -f bsrbmm bsrbmm32 bsrbmm64 *.o