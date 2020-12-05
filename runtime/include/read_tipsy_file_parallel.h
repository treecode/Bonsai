/*
 * read_tipsy_file_parallel.h
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#pragma once

#include <my_cuda_rt.h>
#include <octree.h>
#include <vector>

void read_tipsy_file_parallel(std::vector<real4> &bodyPositions, std::vector<real4> &bodyVelocities,
                              std::vector<ullong> &bodiesIDs,  float eps2, string fileName,
                              int rank, int procs, int &NTotal2, int &NFirst,
                              int &NSecond, int &NThird, octree *tree,
                              std::vector<real4> &dustPositions, std::vector<real4> &dustVelocities,
                              std::vector<ullong> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor,
                              const bool restart);
