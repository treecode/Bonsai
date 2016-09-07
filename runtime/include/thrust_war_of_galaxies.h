/*
 * thrust_war_of_galaxies.h
 *
 *  Created on: Jun 15, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef THRUST_WAR_OF_GALAXIES_H_
#define THRUST_WAR_OF_GALAXIES_H_

#include <octree.h>

/// Remove particles behind visualization sphere
extern "C" void remove_particles(tree_structure &tree,
  real deletion_radius_square, my_dev::dev_mem<uint> &user_particles, int number_of_users);

#endif /* THRUST_WAR_OF_GALAXIES_H_ */
