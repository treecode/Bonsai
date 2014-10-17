cmake_minimum_required(VERSION 2.8)
project(bonsai_catalyst)

#--------------------------------------------------
#  MPI  
#--------------------------------------------------
if (BONSAI_CATALYST_OSX_GCC)
  #
  # special mpi for JB's GCC compile on mac
  # can be ignored as it is only used if -DBONSAI_CATALYST_OSX_GCC
  # is set on command line
  #
  set(MPI_ROOT /Users/biddisco/apps/mpich-3.1)
  set(MPI_DIR /Users/biddisco/apps/mpich-3.1)
  set(MPI_C_INCLUDE_PATH /Users/biddisco/apps/mpich-3.1/include)
  set(MPI_C_LIBRARIES /Users/biddisco/apps/mpich-3.1/lib/libmpich.dylib)
  set(MPI_CXX_INCLUDE_PATH  /Users/biddisco/apps/mpich-3.1/include)
  set(MPI_CXX_LIBRARIES     /Users/biddisco/apps/mpich-3.1/lib/libmpichcxx.dylib;/Users/biddisco/apps/mpich-3.1/lib/libp)
  set(MPI_C_COMPILER        /usr/local/bin/mpicc)
  set(MPI_EXTRA_LIBRARY     /Users/biddisco/apps/mpich-3.1/lib/libpmpich.dylib;/Users/biddisco/apps/mpich-3.1/lib/libmpi)
  set(MPI_LIBRARY           /Users/biddisco/apps/mpich-3.1/lib/libmpichcxx.dylib)
else()
  # 
  # Don't need this if using mpicxx
  #
  find_package(MPI)
endif()

include_directories(${MPI_C_INCLUDE_PATH})

#--------------------------------------------------
# ParaView
#--------------------------------------------------
option(BONSAI_CATALYST_PARAVIEW ON "Link to ParaView")
if(BONSAI_CATALYST_PARAVIEW)

  find_package(ParaView 4.1 REQUIRED COMPONENTS vtkPVCatalyst vtkPVVTKExtensionsDefault)
  include("${PARAVIEW_USE_FILE}")
  message("Paraview cmake dir is ${ParaView_CMAKE_DIR}")
  include_directories(
    ${ParaView_CMAKE_DIR}/../CoProcessing/PythonCatalyst
    ${ParaView_DIR}/CoProcessing/PythonCatalyst  
  )
  set(CATALYST_ADAPTOR_SRCS
    BonsaiCatalystData.cpp
    vtkBonsaiPipeline.cpp
  )
endif()

#--------------------------------------------------
# Flags to get code compiling with clang on mac
#--------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  add_definitions(-DBONSAI_CATALYST_CLANG)
endif()

#--------------------------------------------------
# Make sure c++11 is selected, but don't clobber user flags
# if they have been set by hand. Assume user knows what they're doing
#--------------------------------------------------
if ("${CMAKE_CXX_FLAGS}" STREQUAL "")
  set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "CXX flags" FORCE)
endif()

#--------------------------------------------------
# On mac, if we use cuda we need stdlibc++
# add an option to use boost where std:: isn't available
#--------------------------------------------------
option(BONSAI_CATALYST_USE_STDLIB OFF "Use stdlibc++ instead of libc++")
if(BONSAI_CATALYST_USE_STDLIB)
  add_definitions(-DBONSAI_CATALYST_STDLIB)
  #-----------------------------------------------
  # Boost
  #-----------------------------------------------
  find_package( Boost 1.54.0 )
  include_directories(${Boost_INCLUDE_DIR})
  LINK_DIRECTORIES(${Boost_LIBRARY_DIR})
endif()
  
#--------------------------------------------------
# source files
#--------------------------------------------------
set(SRC_FILES 
  RendererData.cpp
  main.cpp
  ${CATALYST_ADAPTOR_SRCS}
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# exe
#
add_executable(catalyst ${SRC_FILES})
if(BONSAI_CATALYST_PARAVIEW)
  message("Adding catalyst links")
  target_link_libraries(catalyst vtkPVCatalyst vtkPVVTKExtensionsDefault)
endif()
target_link_libraries(catalyst bonsai_tools_common ${MPI_C_LIBRARIES})
