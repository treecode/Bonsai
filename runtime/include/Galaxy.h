/*
 * Galaxies.h
 *
 *  Created on: Apr 12, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef GALAXY_H_
#define GALAXY_H_

#include <my_cuda_rt.h>
#include <vector>

/// One defined galaxy
struct Galaxy
{
  /// Center of mass
  real4 getCenterOfMass() const;

  /// Total Velocity
  real4 getTotalVelocity() const;

  /// Move center of mass to origin (0,0,0)
  void centering();

  /// Remove total velocity
  void steady();

  std::vector<real4> pos;
  std::vector<real4> vel;
  std::vector<int> ids;

  std::vector<real4> pos_dust;
  std::vector<real4> vel_dust;
  std::vector<int> ids_dust;
};

#endif /* GALAXY_H_ */
