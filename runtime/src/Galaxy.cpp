/*
 * Galaxy.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include <Galaxy.h>

real4 Galaxy::getCenterOfMass() const
{
  real mass;
  real4 result = make_real4(0.0, 0.0, 0.0, 0.0);
  for (auto const& p : pos)
  {
  	mass = p.w;
    result.x += mass * p.x;
    result.y += mass * p.y;
    result.z += mass * p.z;
    result.w += mass;
  }

  result.x /= result.w;
  result.y /= result.w;
  result.z /= result.w;
  return result;
}

real4 Galaxy::getTotalVelocity() const
{
  real4 result = make_real4(0.0, 0.0, 0.0, 0.0);
  for (auto const& v : vel)
  {
    result.x += v.x;
    result.y += v.y;
    result.z += v.z;
  }

  result.x /= vel.size();
  result.y /= vel.size();
  result.z /= vel.size();
  return result;
}

void Galaxy::centering()
{
  real4 center_of_mass = getCenterOfMass();
  for (auto &p : pos)
  {
    p.x -= center_of_mass.x;
    p.y -= center_of_mass.y;
    p.z -= center_of_mass.z;
  }
}

void Galaxy::steady()
{
  real4 total_velocity = getTotalVelocity();
  for (auto &v : vel)
  {
    v.x -= total_velocity.x;
    v.y -= total_velocity.y;
    v.z -= total_velocity.z;
  }
}

void Galaxy::translate(real4 w)
{
  for (auto &p : pos)
  {
    p.x += w.x;
    p.y += w.y;
    p.z += w.z;
  }
}

void Galaxy::accelerate(real4 w)
{
  for (auto &v : vel)
  {
    v.x += w.x;
    v.y += w.y;
    v.z += w.z;
  }
}
