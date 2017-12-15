/*
 * Galaxies.h
 *
 *  Created on: Apr 12, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#pragma once

#include <my_cuda_rt.h>
#include <vector>

typedef unsigned long long ullong;

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

  /// Move center of the galaxy
  void translate(real4 w);

  /// Accelerate the galaxy
  void accelerate(real4 w);

  std::vector<real4> pos;
  std::vector<real4> vel;
  std::vector<ullong> ids;

  std::vector<real4> pos_dust;
  std::vector<real4> vel_dust;
  std::vector<ullong> ids_dust;
};
