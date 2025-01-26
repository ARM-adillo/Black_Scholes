cusrand := Base_custom_rand.cpp
armrand := Base_rand_armpl.cpp
randpar := Base_rand_par.cpp
randpool := Base_rand_pool.cpp

BIN1 := Custom_rand
BIN2 := Armpl_rand
BIN3 := STL_rand
BIN4 := Pool_rand

COMMON_FLAGS := -Ofast -finline-functions -g -fno-omit-frame-pointer -std=c++20
LDFLAGS :=
OPTION := 

ARM_FLAGS := -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp -lamath -lm -larmpl

all: rand1 rand2 rand3 

rand1: $(cusrand)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN1) $(LDFLAGS) $(OPTION)

rand2: $(armrand)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN2) $(LDFLAGS) $(OPTION)

rand3: $(randpar)
	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN3) $(LDFLAGS) $(OPTION)

#rand4: $(randpool)
#	armclang++ -mcpu=native -mtune=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN4) $(LDFLAGS) $(OPTION)

clean:
	@rm -rf $(BIN1) $(BIN2) $(BIN3)
