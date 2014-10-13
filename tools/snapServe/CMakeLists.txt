cmake_minimum_required(VERSION 2.8)
project(bonsai_snapserve)


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

#
# Flags to get code compiling with clang on mac
#
message("compiler is ${CMAKE_CXX_COMPILER_ID}")
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  add_definitions(-DBONSAI_CATALYST_CLANG)
else()
  add_definitions("-Wno-literal-suffix") # FIX for OpenMPI with gcc491
  add_definitions("-Wno-main") # Do not generate warrning on main
  add_definitions("-Werror")
endif()

#
# Make sure c++11 is selected, but don't clobber user flags
# if they have been set by hand. Assume user knows what they're doing
#
if ("${CMAKE_CXX_FLAGS}" STREQUAL "")
  set(CMAKE_CXX_FLAGS "-std=c++11" CACHE STRING "CXX flags" FORCE)
endif()
  
#
# source files
#
set(SRC_FILES 
  anyoption.cpp
  main.cpp
)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

#
# exe
#
add_executable(snapServe ${SRC_FILES})
target_link_libraries(snapServe ${MPI_C_LIBRARIES})
