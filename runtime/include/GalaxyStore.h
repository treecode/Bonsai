/*
 * GalaxyStore.h
 *
 *  Created on: May 3, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef GALAXYSTORE_H_
#define GALAXYSTORE_H_

#include "Galaxy.h"
#include <vector>

class GalaxyStore
{
 public:

  void init(std::string const& path, octree *tree);

  Galaxy getGalaxy(int user_id, int galaxy_id, double angle, double velocity) const;

 private:

  std::vector<Galaxy> galaxies;

};

#endif /* GALAXYSTORE_H_ */
