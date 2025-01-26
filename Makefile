CC := armclang++

SRC	:= BSMC.cxx
BIN := BSMC.exe

COMMON_FLAGS 	:= -finline-functions -g -fno-omit-frame-pointer -std=c++20 -fstrict-aliasing -lamath -lm
AGGRESSIVE 		:= -Ofast -funroll-loops -ftree-vectorize 
MATH 			:= -fassociative-math -fno-signed-zeros -fno-trapping-math -freciprocal-math -ffinite-math-only

LDFLAGS := -flto
OPTION := 

all: BSMC 

BSMC: $(SRC)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN) $(LDFLAGS) $(OPTION)

clean:
	@rm -rf $(BIN)
