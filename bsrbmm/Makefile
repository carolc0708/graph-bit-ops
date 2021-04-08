CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
# FLAGS = -arch=sm_70 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96

PFLAGS = -arch=sm_60 -O3 -std=c++11 -w
VFLAGS = -arch=sm_70 -O3 -std=c++11 -w

LINK = -lcublas -lcusparse -lcudart

all: bsrbmm4 bsrbmm8 bsrbmm16 bsrbmm32 bsrbmm64

tc-p100: tc4p tc8p tc16p tc32p
tc-v100: tc4v tc8v tc16v tc32v


# tc-p100
tc4p: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(PFLAGS) $(LINK) -DBLOCKSIZE=4 tc.cu -o $@
tc8p: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(PFLAGS) $(LINK) -DBLOCKSIZE=8 tc.cu -o $@
tc16p: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(PFLAGS) $(LINK) -DBLOCKSIZE=16 tc.cu -o $@
tc32p: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(PFLAGS) $(LINK) -DBLOCKSIZE=32 tc.cu -o $@

# tc-v100
tc4v: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(VFLAGS) $(LINK) -DBLOCKSIZE=4 tc.cu -o $@
tc8v: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(VFLAGS) $(LINK) -DBLOCKSIZE=8 tc.cu -o $@
tc16v: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(VFLAGS) $(LINK) -DBLOCKSIZE=16 tc.cu -o $@
tc32v: tc.cu bsrbmm.cu utility.cu csr2bsr_batch.cu
	$(CC) $(VFLAGS) $(LINK) -DBLOCKSIZE=32 tc.cu -o $@

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
	rm -f bsrbmm4 bsrbmm8 bsrbmm16 bsrbmm32 bsrbmm64 \
	tc4 tc8 tc16 tc32 tc4p tc8p tc16p tc32p tc4v tc8v tc16v tc32v *.o