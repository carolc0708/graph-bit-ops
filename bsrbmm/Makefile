CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

bsrbmm: benchmark-bsrbmm.cu bsrbmm.cu csr2bsr_batch.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmm.cu -o $@

clean:
	rm -f bsrbmm *.o