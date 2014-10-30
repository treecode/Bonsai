CXX = mpicxx
CC  = mpicc
LD  = mpicxx
F90  = mpif90

CFLAGS =  -Wall -Werror
CUDA_TK  = /usr/local/cuda
# CUDA_TK = $(HOME)/usr/local/cuda-5.5

CXXFLAGS += -Wall -fopenmp -std=c++11  -fPIE 
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

ENGINE=particles
ENGINE=smoke
ENGINE=density
#ENGINE=splotch



INCLUDE_FLAGS= -I./ -I./$(ENGINE)/
INCLUDE_FLAGS += -I$(CUDA_TK)/include 
LDFLAGS = -lGL -lGLU -lglut -lGLEW -fopenmp -lrt
LDFLAGS += -pie -rdynamic
#-L$(PWD)/lib/linux -lGLEW_x86_64


ifeq ($(PROFILE),1)
	CXXFLAGS += -D__COMPOSITE_PROFILE
endif

ifeq ($(ICET),1)
	ICETPATH=../../IceT
	INCLUDE_FLAGS += -I$(ICETPATH)/include  
	LDFLAGS += -L$(ICETPATH)/lib -lIceTCore -lIceTMPI  
	CXXFLAGS += -DUSE_ICET
endif

ifeq ($(APPLEOSX),1)
	INCLUDE_FLAGS += -I/opt/local/include
	CXXFLAGS +=-Wno-deprecated-declarations 
	LDFLAGS = -framework OpenGL -L/opt/local/lib -lglut -lglew -fopenmp
endif

ifeq ($(PNG),1)
	CXXFLAGS += -D_PNG
	LDFLAGS += -lpng
endif

OBJ = main.o $(ENGINE)/renderloop.o anyoption.o $(ENGINE)/renderer.o paramgl.o RendererData.o RendererDataDistribute.o
ifeq ($(ENGINE),smoke)
	OBJ += tr.o
	OBJ += smoke/GLSLProgram.o
	OBJ += smoke/SmokeShaders.o
	OBJ += smoke/framebufferObject.o
#	OBJ += smoke/depthSort.cu_o
	OBJ += smoke/depthSort.o
endif
ifeq ($(ENGINE),density)
	OBJ += tr.o
	OBJ += density/GLSLProgram.o
	OBJ += density/SmokeShaders.o
	OBJ += density/framebufferObject.o
#	OBJ += density/depthSort.cu_o
	OBJ += density/depthSort.o
endif
ifeq ($(ENGINE),splotch)
	OBJ += splotch/GLSLProgram.o
	OBJ += splotch/splotch.o
endif
ifeq ($(ENGINE),splotchGL)
	OBJ += splotch/GLSLProgram.o
	OBJ += splotchGL/splotch.o
endif

PROG= renderer

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



$(OBJ): CameraPath.h anyoption.h  paramgl.h  param.h   RendererData.h  $(ENGINE)/renderloop.h  $(ENGINE)/renderer.h  vector_math.h \
	splotch/Blending.h  splotch/GLSLProgram.h  splotch/MathArray.h  splotch/renderer.h  splotch/renderloop.h  splotch/Splotch.h  splotch/Texture.h  splotch/Vertex.h 


