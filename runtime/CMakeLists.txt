PROJECT(bonsai2)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

#Searching CUDA
FIND_PACKAGE(CUDA REQUIRED)

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )
  
option(CUDA_DEVICE_DEBUGGING
  "On to enable debugging device code."
  OFF
  )  

option(USE_MPI
  "On to build support for MPI"
  OFF
  )

set(CMAKE_DEBUG_POSTFIX "D")

include_directories(${CMAKE_SOURCE_DIR}/include)

set (CCFILES
  src/build.cpp 
  src/compute_properties.cpp 
  src/gpu_iterate.cpp 
  src/libraryInterface.cpp
  src/load_kernels.cpp
  src/main.cpp
  src/octree.cpp
  src/parallel.cpp
  src/sort_bodies_gpu.cpp
  src/renderloop.cpp
  )

set (HFILES
  include/my_cuda_rt.h
  include/my_ocl.h
  include/node_specs.h
  include/octree.h
  include/tipsydefs.h
  )

set (CUFILES
  CUDAkernels/build_tree.cu
  CUDAkernels/compute_propertiesD.cu
  CUDAkernels/dev_approximate_gravity.cu
  CUDAkernels/dev_approximate_gravity_let.cu
  CUDAkernels/parallel.cu
  CUDAkernels/sortKernels.cu
  CUDAkernels/timestep.cu
  )

set (CUHFILES
  CUDAkernels/support_kernels.cu
  CUDAkernels/scanKernels.cu
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Include Files" FILES ${CUHFILES})

set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

if (CUDA_DEVICE_DEBUGGING)
  set(DEVICE_DEBUGGING -G0)
endif (CUDA_DEVICE_DEBUGGING)

if (WIN32)
  add_definitions(-D_CRT_SECURE_NO_DEPRECATE)
endif (WIN32)

add_definitions(-DTIPSYOUTPUT)

cuda_add_executable(bonsai2
  ${CCFILES} 
  ${HFILES}
  ${CUFILES}
  OPTIONS ${GENCODE_SM20} ${VERBOSE_PTXAS} ${DEVICE_DEBUGGING}
  )

