CC := armclang++

REF	:= BSM_ref.cpp
BIN_REF = ref.exe

BIG_BRAIN := BSM_BigBrain.cpp
BIN_BRAIN := bigbrain.exe

MEGAMIND := BSM_Megamind.cpp
BIN_MEGA := megamind.exe

MASTERMIND := BSM_MasterMind.cpp
BIN_MASTER := mastermind.exe

INEXACT := BSM_Inexact_Maths.cpp
BIN_INEX := inexact.exe

STREAMS := BSM_Inexact_Streams.cpp
BIN_STREAMS := streams.exe

COMMON_FLAGS := -finline-functions -g -fno-omit-frame-pointer -std=c++20 -fstrict-aliasing -lamath -lm
AGGRESSIVE := -Ofast -funroll-loops -ftree-vectorize 
MATH := -fassociative-math -fno-signed-zeros -fno-trapping-math -freciprocal-math -ffinite-math-only

LDFLAGS := -flto
OPTION := 

all: ref big mega master inex streams

ref: $(REF)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_REF) $(LDFLAGS) $(OPTION)

big: $(BIG_BRAIN)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_BRAIN) $(LDFLAGS) $(OPTION)

mega: $(MEGAMIND)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_MEGA) $(LDFLAGS) $(OPTION)

master: $(MASTERMIND)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_MASTER) $(LDFLAGS) $(OPTION)

inex: $(INEXACT)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_INEX) $(LDFLAGS) $(OPTION)

streams: $(STREAMS)
	$(CC) -mcpu=native -mtune=native $(COMMON_FLAGS) $(AGGRESSIVE) $(MATH) -fopenmp -armpl $? -o $(BIN_STREAMS) $(LDFLAGS) $(OPTION)

clean:
	@rm -rf $(BIN) $(BIG_BRAIN) $(MEGAMIND) $(MASTERMIND) $(INEXACT) $(STREAMS)
