CC = nvcc

FLAGS = -arch=sm_60 -O3 -std=c++11 -w # -Xptxas --maxrregcount=96
LINK = -lcublas -lcusparse -lcudart

bsrbmv: benchmark-bsrbmv.cu bsrbmv.cu
	$(CC) $(FLAGS) $(LINK) benchmark-bsrbmv.cu -o $@

clean:
	rm -f bsrbmv *.o