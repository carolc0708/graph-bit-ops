CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
# FLAGS = -arch=sm_70 -O3 -std=c++11 -w
LINK = -lcublas -lcusparse -lcudart

all: bfs4

bfs: bfs4 bfs32
sssp: sssp4 sssp32
pr: pr4 pr32

# pr
bfs4: bfs.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) bfs.cu -DBLOCKSIZE=4 -o $@
bfs32: bfs.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) bfs.cu -DBLOCKSIZE=32 -o $@

# sssp
sssp4: sssp.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) sssp.cu -DBLOCKSIZE=4 -o $@
sssp32: sssp.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) sssp.cu -DBLOCKSIZE=32 -o $@

# pr
pr4: pr.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) pr.cu -DBLOCKSIZE=4 -o $@
pr32: pr.cu bsrbmv.cu utility.cu csr2bsr_batch_bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) pr.cu -DBLOCKSIZE=32 -o $@

clean:
	rm -f bfs4 bfs32 sssp4 sssp32 pr4 pr32