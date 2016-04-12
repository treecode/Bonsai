/*
 * Galaxies.h
 *
 *  Created on: Apr 12, 2016
 *      Author: Bernd Doser <bernd.doser@hits.org>
 */

#ifndef GALAXIES_H_
#define GALAXIES_H_

#include <vector>

typedef float4 real4;
typedef float real;

struct Galaxy
{
  Galaxy(std::vector<real4> const& bodyPositions,
		 std::vector<real4> const& bodyVelocities,
		 std::vector<int> const& bodyIDs)
   : bodyPositions(bodyPositions),
     bodyVelocities(bodyVelocities),
     bodyIDs(bodyIDs)
  {}

  std::vector<real4> bodyPositions;
  std::vector<real4> bodyVelocities;
  std::vector<int> bodyIDs;
};

typedef std::vector<Galaxy> Galaxies;

#endif /* GALAXIES_H_ */
