CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

all: bsrbmv8 bsrbmv16 bsrbmv32 bsrbmv64

# default
bsrbmv8: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=8 -o $@

bsrbmv16: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=16 -o $@

bsrbmv32: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -o $@

bsrbmv64: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -o $@

# prof each thread block
bsrbmv32prof: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -DPROF -o $@

bsrbmv64prof: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -DPROF -o $@

# workload-1 (splitonly)
bsrbmv32split: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=32 -DSPLIT -o $@

# workload-3 (bcoo)
bsrbmv32bcoo: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) opt-bsrbmv.cu -DBLOCKSIZE=32 -o $@

# workload-2 (split & merge)
bsrbmv32ws: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=32 -o $@

bsrbmv64ws: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=64 -o $@

# workload-2 (split & merge), prof each thread block
bsrbmv32wsprof: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=32 -DPROF -o $@

bsrbmv64wsprof: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=64 -DPROF -o $@


# === not currently in use ===

# nonbatch csr2bsr for small matrices read
bsrbmv32nonbatch: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -DNONBATCH -o $@

bsrbmv64nonbatch: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -DNONBATCH -o $@

# debug csr2bsr_batch.cu
bsrbmv32debug: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -DDEBUG -o $@

bsrbmv64debug: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -DDEBUG -o $@

clean:
	rm -f bsrbmv8 bsrbmv16 bsrbmv32 bsrbmv64 *.o