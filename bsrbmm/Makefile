CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
# FLAGS = -arch=sm_70 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

all: bsrbmm4 bsrbmm8 bsrbmm16 bsrbmm32 bsrbmm64
tc: tc4 tc8 tc16 tc32

# tc
tc4: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=4 tc.cu -o $@
tc8: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=8 tc.cu -o $@
tc16: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=16 tc.cu -o $@
tc32: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=32 tc.cu -o $@

# default
bsrbmm4: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=4 benchmark-bsrbmm.cu -o $@

bsrbmm8: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=8 benchmark-bsrbmm.cu -o $@

bsrbmm16: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=16 benchmark-bsrbmm.cu -o $@

bsrbmm32: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=32 benchmark-bsrbmm.cu -o $@

bsrbmm64: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=64 benchmark-bsrbmm.cu -o $@

# prof each thread block
bsrbmm32prof: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=32 -DPROF benchmark-bsrbmm.cu -o $@

bsrbmm64prof: benchmark-bsrbmm.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) -DBLOCKSIZE=64 -DPROF benchmark-bsrbmm.cu -o $@

clean:
	rm -f bsrbmm4 bsrbmm8 bsrbmm16 bsrbmm32 bsrbmm64 tc4 tc8 tc16 tc32 *.o