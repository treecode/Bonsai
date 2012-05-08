CXX = g++
CC  = gcc
LD  = g++
F90 = gfortran

OMPFLAGS  = -fopenmp
OMPFLAGS += -D_GLIBCXX_PARALLEL

OFLAGS = -O3 -g -Wall
# OMPFLAGS=

CXXFLAGS =  -fPIC $(OFLAGS) -Wstrict-aliasing=2 $(OMPFLAGS)



LDFLAGS = 

SRCPATH = ./
SRC = add_dust.cpp anyoption.cpp
OBJ = $(SRC:%.cpp=%.o)

SRC1 = initorbit.cpp anyoption.cpp
OBJ1 = $(SRC1:%.cpp=%.o)

PROG  = add_dust
PROG1 = initorbit

RM = /bin/rm

all:	  $(PROG) $(PROG1)


$(PROG): $(OBJ) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)

$(PROG1): $(OBJ1) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o $(PROG)  $(PROG1)

$(OBJ): DustRing.h vector3.h tipsydefs.h







