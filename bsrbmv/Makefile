CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
# FLAGS = -arch=sm_70 -O3 -std=c++11 -w
LINK = -lcublas -lcusparse -lcudart

#all: bsrbmv8 bsrbmv16 bsrbmv32 bsrbmv64
all: bsrbmv8-sharedallunsigned bsrbmv8-allunsigned bsrbmv8-new bsrbmv8-baseline \
bsrbmv16-sharedallunsigned bsrbmv16-allunsigned bsrbmv16-new bsrbmv16-baseline \
bsrbmv32-shared bsrbmv32-new bsrbmv32-baseline \
bsrbmv64-shared bsrbmv64-new bsrbmv64-baseline

# bsrbmv4
bsrbmv4: benchmark-bsrbmv.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -DBLOCKSIZE=4 -o $@

# new versions
bsrbmv8-sharedallunsigned: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=8 -DSHARED -DALLUNSIGNED -o $@

bsrbmv8-allunsigned: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=8 -DALLUNSIGNED -o $@

bsrbmv8-new: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=8 -DNEW -o $@

bsrbmv8-baseline: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=8 -o $@

bsrbmv16-sharedallunsigned: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=16 -DSHARED -DALLUNSIGNED -o $@

bsrbmv16-allunsigned: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=16 -DALLUNSIGNED -o $@

bsrbmv16-new: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=16 -DNEW -o $@

bsrbmv16-baseline: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=16 -o $@

bsrbmv32-shared: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=32 -DSHARED -DALLUNSIGNED -o $@
bsrbmv32-new: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=32 -DNEW -o $@
bsrbmv32-baseline: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=32 -o $@

bsrbmv64-shared: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=64 -DSHARED -DALLUNSIGNED -o $@
bsrbmv64-new: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=64 -DNEW -o $@
bsrbmv64-baseline: benchmark-bsrbmv-all.cu bsrbmv.cu utility.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv-all.cu -DBLOCKSIZE=64 -o $@

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
	rm -f bsrbmv4 bsrbmv8 bsrbmv16 bsrbmv32 bsrbmv64 \
	bsrbmv8-sharedallunsigned bsrbmv8-allunsigned bsrbmv8-new bsrbmv8-baseline \
	bsrbmv16-sharedallunsigned bsrbmv16-allunsigned bsrbmv16-new bsrbmv16-baseline \
	bsrbmv32-shared bsrbmv32-new bsrbmv32-baseline \
	bsrbmv64-shared bsrbmv64-new bsrbmv64-baseline *.o