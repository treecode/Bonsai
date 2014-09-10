CXX = mpicxx
CC  = mpicc
LD  = mpicxx

OMPFLAGS  = -fopenmp 
#OMPFLAGS += -D_GLIBCXX_PARALLEL

OFLAGS = -O3 -g -Wall 
# OMPFLAGS=

CXXFLAGS =  -fPIC $(OFLAGS) $(OMPFLAGS) -std=c++11



LDFLAGS = 

SRCPATH = ./

SRC1 = cvt_tipsy2bonsai.cpp
SRC2 = readBonsai.cpp
SRC3 = benchmark.cpp
SRC4 = cvt_jamie2bonsai.cpp
SRC5 = cvt_tomo2bonsai.cpp
OBJ1 = $(SRC1:%.cpp=%.o)
OBJ2 = $(SRC2:%.cpp=%.o)
OBJ3 = $(SRC3:%.cpp=%.o)
OBJ4 = $(SRC4:%.cpp=%.o)
OBJ5 = $(SRC5:%.cpp=%.o)

PROG1 = cvt_tipsy2bonsai
PROG2 = readBonsai
PROG3 = benchmark
PROG4 = cvt_jamie2bonsai
PROG5 = cvt_tomo2bonsai

RM = /bin/rm

all:	  $(PROG1) $(PROG2) $(PROG3) $(PROG4) $(PROG5)


$(PROG1): $(OBJ1) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG2): $(OBJ2) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG3): $(OBJ3) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG4): $(OBJ4) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)
$(PROG5): $(OBJ5) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)




%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o $(PROG1) $(PROG2) $(PROG3) $(PROG4) $(PROG5) $(OBJ1) $(OBJ2) $(OBJ3) $(OBJ4) $(OBJ5)

$(OBJ1): BonsaiIO.h  read_tipsy.h
$(OBJ2): BonsaiIO.h
$(OBJ3): BonsaiIO.h
$(OBJ4): BonsaiIO.h
$(OBJ5): BonsaiIO.h