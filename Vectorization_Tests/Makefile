1loop   := BSM_base_1loop.cpp
extract := BSM_base_extract.cpp
unroll2 := BSM_base_unroll_2.cpp
unroll4 := BSM_base_unroll_4.cpp
unroll8 := BSM_base_unroll_8.cpp

BIN1 := one_loop
BIN2 := extract_loop
BIN3 := unroll_2
BIN4 := unroll_4
BIN5 := unroll_8

COMMON_FLAGS := -Ofast -finline-functions -g -fno-omit-frame-pointer -std=c++20
LDFLAGS :=
OPTION := 

ARM_FLAGS := -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp -lamath -lm -larmpl

all: loop1 loop2 loop3 loop4 loop5

loop1: $(1loop)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN1) $(LDFLAGS) $(OPTION)

loop2: $(extract)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN2) $(LDFLAGS) $(OPTION)

loop3: $(unroll2)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN3) $(LDFLAGS) $(OPTION)

loop4: $(unroll4)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN4) $(LDFLAGS) $(OPTION)

loop5: $(unroll8)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN5) $(LDFLAGS) $(OPTION)

clean:
	@rm -rf $(BIN1) $(BIN2) $(BIN3) $(BIN4) $(BIN5)
