/*
 * war_of_galaxies.cu
 *
 *  Created on: Jun 14, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include <thrust/remove.h>
#include <thrust/device_ptr.h>

// Remove particles behind visualization sphere
extern "C" void remove_particles(const int n_bodies,
                                 my_dev::dev_mem<real4> &pos,
                                 my_dev::dev_mem<real4> &vel,
                                 my_dev::dev_mem<real4> &acc)
{
  thrust::device_ptr<real4> thrust_pos = thrust::device_pointer_cast(pos.raw_p());
  thrust::device_ptr<real4> thrust_vel = thrust::device_pointer_cast(vel.raw_p());
  thrust::device_ptr<real4> thrust_acc = thrust::device_pointer_cast(acc.raw_p());

  //thrust::remove_if();
}
