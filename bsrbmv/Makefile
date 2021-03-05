CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

# all: bsrbmv32 bsrbmv64 bsrbmv32prof bsrbmv64prof

# all: bsrbmv32ws bsrbmv64ws bsrbmv32wsprof bsrbmv64wsprof bsrbmv32 bsrbmv64 bsrbmv32prof bsrbmv64prof

all: bsrbmv32split

# default
bsrbmv: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -o $@

# default: use gpu timer
bsrbmv32: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -o $@

bsrbmv64: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -o $@

# prof each thread block
bsrbmv32prof: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=32 -DPROF -o $@

bsrbmv64prof: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=64 -DPROF -o $@

# workload split only
bsrbmv32split: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=32 -DSPLIT -o $@

# workload splitnmerge
bsrbmv32ws: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=32 -o $@

bsrbmv64ws: workload-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) workload-bsrbmv.cu -DBLOCKSIZE=64 -o $@

# workload splitnmerge, prof each thread block
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
	rm -f bsrbmv bsrbmv32 bsrbmv64 bsrbmv32prof bsrbmv64prof bsrbmv32ws bsrbmv32split \
	    bsrbmv64ws bsrbmv32wsprof bsrbmv64wsprof *.o