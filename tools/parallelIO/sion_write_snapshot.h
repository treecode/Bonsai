#pragma once
#include <iostream>
#include <fstream>
#include "tipsydefs.h"

size_t sion_write_snapshot(
    const real4 *bodyPositions, 
    const real4 *bodyVelocities, 
    const int *bodyIds, 
    const int n, 
    const std::string &fileName, 
    const float time,
    const int rank, 
    const int nrank, 
    const MPI_Comm &comm)
{
  assert(0);
  return 0;
}
