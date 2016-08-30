/*
 * war_of_galaxies.cu
 *
 *  Created on: Jun 14, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifdef USE_THRUST

#include "bonsai.h"
#include "thrust_war_of_galaxies.h"
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

/// Functor return true if position is out of sphere
struct OutOfSphereChecker
{
  OutOfSphereChecker(real deletion_radius_square, thrust::device_ptr<uint> user_particles)
   : deletion_radius_square(deletion_radius_square), user_particles(user_particles)
  {}

  __device__
  bool operator()(thrust::tuple<real4, real4, int> const& t) const
  {
    real4 position = thrust::get<0>(t);
    int user_id = thrust::get<2>(t) % 10;
    if (position.x * position.x + position.y * position.y + position.z * position.z > deletion_radius_square and user_id != 9)
    {
      atomicDec(user_particles.get() + user_id, UINT_MAX);
      return true;
    }
    return false;
  }

  real deletion_radius_square;

  thrust::device_ptr<uint> user_particles;
};

// Remove particles out of sphere
extern "C" void remove_particles(tree_structure &tree,
  real deletion_radius_square, my_dev::dev_mem<uint> user_particles)
{
  thrust::device_ptr<real4> thrust_pos = thrust::device_pointer_cast(tree.bodies_pos.raw_p());
  thrust::device_ptr<real4> thrust_vel = thrust::device_pointer_cast(tree.bodies_vel.raw_p());
  thrust::device_ptr<int> thrust_ids = thrust::device_pointer_cast(tree.bodies_ids.raw_p());
  thrust::device_ptr<uint> thrust_user_particles = thrust::device_pointer_cast(user_particles.raw_p());

  try {
    // auto is not working, compiler assume int
    thrust::zip_iterator< thrust::tuple<thrust::device_ptr<real4>, thrust::device_ptr<real4>, thrust::device_ptr<int> > > new_end =
      thrust::remove_if(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(thrust_pos, thrust_vel, thrust_ids)),
        thrust::make_zip_iterator(thrust::make_tuple(thrust_pos + tree.n, thrust_vel + tree.n, thrust_ids + tree.n)),
        OutOfSphereChecker(deletion_radius_square, thrust_user_particles)
      );

    // Set new number of particles
    tree.n = thrust::get<0>(new_end.get_iterator_tuple()) - thrust_pos;
  }
  catch(thrust::system_error &e)
  {
    std::cerr << "Error accessing vector element: " << e.what() << std::endl;
    exit(-1);
  }
}

#endif