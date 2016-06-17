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
#include <thrust/remove.h>

/// Functor return true if position is out of sphere
struct OutOfSphereChecker
{
  OutOfSphereChecker(real deletion_radius_square, std::vector<int> &user_particles)
   : deletion_radius_square(deletion_radius_square), user_particles(user_particles)
  {}

  __host__ __device__
  bool operator()(thrust::tuple<real4, real4, int> const& t) const
  {
    real4 position = thrust::get<0>(t);
    bool result = position.x * position.x + position.y * position.y + position.z * position.z > deletion_radius_square
      and thrust::get<2>(t) % 10 != 9;
    if (result) --user_particles[thrust::get<2>(t) % 10];
    return result;
  }

  real deletion_radius_square;

  std::vector<int> &user_particles;
};

// Remove particles out of sphere
extern "C" void remove_particles(tree_structure &tree,
  real deletion_radius_square, std::vector<int> &user_particles)
{
  thrust::device_ptr<real4> thrust_pos = thrust::device_pointer_cast(tree.bodies_pos.raw_p());
  thrust::device_ptr<real4> thrust_vel = thrust::device_pointer_cast(tree.bodies_vel.raw_p());
  thrust::device_ptr<int> thrust_ids = thrust::device_pointer_cast(tree.bodies_ids.raw_p());

  // auto is not working, compiler assume int
  thrust::zip_iterator<thrust::tuple<thrust::device_ptr<real4>, thrust::device_ptr<real4>, thrust::device_ptr<int> > >
    new_end = thrust::remove_if(thrust::device, thrust::make_zip_iterator(thrust::make_tuple(thrust_pos, thrust_vel, thrust_ids)),
    thrust::make_zip_iterator(thrust::make_tuple(thrust_pos + tree.n, thrust_vel + tree.n, thrust_ids + tree.n)),
    OutOfSphereChecker(deletion_radius_square, user_particles));

  int new_nb_of_particles = thrust::get<0>(new_end.get_iterator_tuple()) - thrust_pos;
  std::cout << "new_nb_of_particles = " << new_nb_of_particles << std::endl;
  tree.n = new_nb_of_particles;
}

#endif
