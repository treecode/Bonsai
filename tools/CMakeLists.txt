cmake_minimum_required(VERSION 2.8)
project(bonsai_tools)

#--------------------------------------------------
# Setup Output directories
#--------------------------------------------------
SET (CMAKE_RUNTIME_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Executables."
)
SET (CMAKE_LIBRARY_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all Libraries"
)
SET (CMAKE_ARCHIVE_OUTPUT_DIRECTORY
    ${PROJECT_BINARY_DIR}/bin CACHE PATH "Single Directory for all static libraries."
)

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/common)

add_library(bonsai_tools_common
  common/anyoption.cpp
)

add_subdirectory(catalyst)
add_subdirectory(snapServe)

