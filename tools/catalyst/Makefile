CXX = mpicxx
CC  = mpicc
LD  = mpicxx
F90  = mpif90

CFLAGS =  -Wall -Werror
CUDA_TK  = /usr/local/cuda

CXXFLAGS = -Wall -fopenmp -std=c++11  -fPIE 
CXXFLAGS += -Werror
#ignore warnings caused by openmpi
CXXFLAGS += -Wno-literal-suffix
#ignore warnings in a constructor when intialization is not done in the order the variables are declared
CXXFLAGS += -Wno-reorder
CXXFLAGS += -Wno-main  # ignore warrnings on main()


OPT += -O3 -g

CFLAGS += $(OPT) -fPIE
CXXFLAGS += $(OPT)

CXXFLAGS += -mavx


NVCC=$(CUDATK)/bin/nvcc
NVCCFLAGS=-arch=sm_35

INCLUDE_FLAGS= -I./ -I./$(ENGINE)/
INCLUDE_FLAGS += -I$(CUDA_TK)/include 
LDFLAGS = -lGL -lGLU -lglut -lGLEW -fopenmp -lrt
LDFLAGS += -pie 
LDFLAGS += -rdynamic

ifeq ($(ICET),1)
	ICETPATH=../../IceT
	INCLUDE_FLAGS += -I$(ICETPATH)/include  
	LDFLAGS += -L$(ICETPATH)/lib -lIceTCore -lIceTMPI  
	CXXFLAGS += -DUSE_ICET
endif

ifeq ($(APPLEOSX),1)
	INCLUDE_FLAGS += -I/opt/local/include
	CXXFLAGS +=-Wno-deprecated-declarations 
	LDFLAGS = -framework OpenGL -L/opt/local/lib -lglut -lglew -fopenmp -pie
endif

OBJ = main.o anyoption.o RendererData.o
PROG= catalyst

all: $(PROG)

$(PROG): $(OBJ)
	$(LD) $^ -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

%.o: %.c
	$(CC) $(CFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

%.cu_o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDE_FLAGS) -c $< -o $@

clean:
	/bin/rm -rf $(OBJ) $(PROG)

clean_bak:
	find . -name '*~' -exec /bin/rm -rf '{}' \;



$(OBJ): anyoption.h  RendererData.h


