SRC := BSM_2.cpp
BIN := BSM_2
COMMON_FLAGS := -O3 -finline-functions -g -fno-omit-frame-pointer -std=c++20
LDFLAGS :=
OPTION := 

all: $(BIN)

$(BIN): $(SRC)
	g++ -march=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -ftree-loop-vectorize -fopenmp $? -o $(BIN) $(LDFLAGS) $(OPTION)

llvm: $(SRC)
	clang++ -march=native $(COMMON_FLAGS) -funroll-loops -fopenmp $? -o $(BIN) $(LDFLAGS) $(OPTION)

arm: $(SRC)
	armclang++ -mcpu=native $(COMMON_FLAGS) -funroll-loops -ftree-vectorize -fopenmp -armpl $? -o $(BIN) $(LDFLAGS) $(OPTION)

nvc: $(SRC)
	nvc++ -march=neoverse-v1 $(COMMON_FLAGS) -mp $? -o $(BIN) $(LDFLAGS) $(OPTION)

intel: $(SRC)
	icpx -xHost $(COMMON_FLAGS) -mavx -funroll-loops -qopenmp -qmkl $? -o $(BIN)

run: $(BIN)
	@./stencil 100 100 100 5
	
check: $(BIN)
	@python3 scripts/assert.py $(ARGS)

prof:
	@maqao oneview -R1 --config=cfg.lua -xp=maqao_arm --replace

clean:
	@rm -rf $(BIN) maqao-arm*
