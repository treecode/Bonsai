/*
 * Galaxies.h
 *
 *  Created on: Apr 12, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef GALAXY_H_
#define GALAXY_H_

#include <vector>

typedef float4 real4;
typedef float real;

/// One defined galaxy
class Galaxy
{
 public:
  Galaxy(std::vector<real4> const& bodyPositions,
		 std::vector<real4> const& bodyVelocities,
		 std::vector<int> const& bodyIDs)
   : bodyPositions(bodyPositions),
     bodyVelocities(bodyVelocities),
     bodyIDs(bodyIDs)
  {}

  //real4 getTotalVelocity();

 private:

  std::vector<real4> bodyPositions;
  std::vector<real4> bodyVelocities;
  std::vector<int> bodyIDs;
};

#endif /* GALAXY_H_ */
