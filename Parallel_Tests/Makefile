CC   	:= armclang++
omp  	:= BSM_omp.cpp
omppar1 := BSM_omp_1parallel.cpp
omppar2 := BSM_omp_2parallel_for.cpp

BIN1 := BSM_omp
BIN2 := BSM_omp2
BIN3 := BSM_omp3

COMMON_FLAGS := -Ofast -finline-functions -g -fno-omit-frame-pointer -std=c++20
LDFLAGS :=
OPTION := 

ARM_FLAGS := -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp -lamath -lm -larmpl

all: omp1 omp2 omp3

omp: $(omp)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN1) $(LDFLAGS) $(OPTION)

omp_1_region: $(omppar1)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN2) $(LDFLAGS) $(OPTION)

omp_2_for: $(omppar2)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN3) $(LDFLAGS) $(OPTION)

clean:
	@rm -rf $(BIN1) $(BIN2) $(BIN3)
