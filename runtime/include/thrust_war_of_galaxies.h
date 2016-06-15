/*
 * thrust_war_of_galaxies.h
 *
 *  Created on: Jun 15, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef THRUST_WAR_OF_GALAXIES_H_
#define THRUST_WAR_OF_GALAXIES_H_

/// Remove particles behind visualization sphere
extern "C" void remove_particles(const int n_bodies,
                                 my_dev::dev_mem<real4> &pos,
                                 my_dev::dev_mem<real4> &vel,
                                 my_dev::dev_mem<real4> &acc);

#endif /* THRUST_WAR_OF_GALAXIES_H_ */
