PROJECT(bonsai2)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

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

option(USE_OPENGL
  "On to build support for OpenGL Rendering"
  OFF
  )

FIND_PACKAGE(CUDA REQUIRED)

if (USE_MPI)
  add_definitions(-DUSE_MPI)
endif (USE_MPI)

set(CMAKE_DEBUG_POSTFIX "D")

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/renderer)

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
  )

set (HFILES
  include/my_cuda_rt.h
  include/my_ocl.h
  include/node_specs.h
  include/octree.h
  include/tipsydefs.h
  include/vector_math.h
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

if (USE_OPENGL) 
  set(CCFILES ${CCFILES} src/renderloop.cpp src/render_particles.cpp)
  set(HFILES ${HFILES} include/renderloop.h include/render_particles.h)
  
  set (CCFILES ${CCFILES} 
  renderer/SmokeRenderer.cpp
  renderer/SmokeShaders.cpp
  renderer/paramgl.cpp
  renderer/param.cpp
  renderer/GLSLProgram.cpp
  renderer/framebufferObject.cpp
  )

  set (HFILES ${HFILES} 
  renderer/SmokeRenderer.h
  renderer/SmokeShaders.h
  renderer/paramgl.h
  renderer/param.h
  renderer/GLSLProgram.h
  renderer/framebufferObject.h
  )

  add_definitions(-DUSE_OPENGL)
  FIND_PACKAGE(OpenGL REQUIRED)

  # Link the right freeglut
  if (WIN32)
    if( CMAKE_SIZEOF_VOID_P EQUAL 8 )
      set( LIB_PATH ${CMAKE_SOURCE_DIR}/lib/win64/ )
	  set( GLEW_NAME glew64 )
    else( CMAKE_SIZEOF_VOID_P EQUAL 8 )
	  set( LIB_PATH ${CMAKE_SOURCE_DIR}/lib/win32/ )
	  set( GLEW_NAME glew32 )
    endif( CMAKE_SIZEOF_VOID_P EQUAL 8 )
  endif (WIN32)

  #freeglut
  FIND_PATH(FREEGLUT_INCLUDE_DIR NAMES GL/freeglut.h PATHS ${CMAKE_SOURCE_DIR}/include)
  FIND_LIBRARY(FREEGLUT_LIBRARY NAMES freeglut PATHS ${LIB_PATH})

  #glew
  FIND_PATH(GLEW_INCUDE_DIR NAMES GL/glew.h PATHS ${CMAKE_SOURCE_DIR}/include)
  FIND_LIBRARY(GLEW_LIBRARY NAMES ${GLEW_NAME} PATHS ${LIB_PATH})

  set(ALL_LIBRARIES ${FREEGLUT_LIBRARY} ${GLEW_LIBRARY})

  # Copy DLLs
  file(COPY ${LIB_PATH}/freeglut.dll DESTINATION ${CMAKE_BINARY_DIR}) 
  file(COPY ${LIB_PATH}/${GLEW_NAME}.dll DESTINATION ${CMAKE_BINARY_DIR}) 

endif (USE_OPENGL)


cuda_add_executable(bonsai2
  ${CCFILES} 
  ${HFILES}
  ${CUFILES}
  OPTIONS ${GENCODE_SM20} ${VERBOSE_PTXAS} ${DEVICE_DEBUGGING}
  )

target_link_libraries(bonsai2 ${ALL_LIBRARIES})

#copy test data file
file(COPY ${CMAKE_SOURCE_DIR}/model3_child_compact.tipsy DESTINATION ${CMAKE_BINARY_DIR})

