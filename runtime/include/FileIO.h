/*
 * FileIO.h
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef FILEIO_H_
#define FILEIO_H_

#include <my_cuda_rt.h>
#include <octree.h>
#include <vector>

void read_tipsy_file_parallel(std::vector<real4> &bodyPositions, std::vector<real4> &bodyVelocities,
                              std::vector<int> &bodiesIDs,  float eps2, string fileName,
                              int rank, int procs, int &NTotal2, int &NFirst,
                              int &NSecond, int &NThird, octree *tree,
                              std::vector<real4> &dustPositions, std::vector<real4> &dustVelocities,
                              std::vector<int> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor,
                              const bool restart);

#endif /* FILEIO_H_ */
