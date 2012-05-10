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

SRC2 = initM31_MW.cpp anyoption.cpp
OBJ2 = $(SRC2:%.cpp=%.o)

PROG  = add_dust
PROG1 = initorbit
PROG2 = initM31_MW

RM = /bin/rm

all:	  $(PROG) $(PROG1) $(PROG2)


$(PROG): $(OBJ) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)

$(PROG1): $(OBJ1) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)

$(PROG2): $(OBJ2) 
	$(LD) $(LDFLAGS) $^ -o $@ $(OMPFLAGS)

%.o: $(SRCPATH)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


clean:
	/bin/rm -rf *.o $(PROG)  $(PROG1) $(PROG2)

$(OBJ): DustRing.h vector3.h tipsydefs.h
$(OBJ1): vector3.h tipsydefs.h 
$(OBJ2): vector3.h tipsydefs.h kepler.h







