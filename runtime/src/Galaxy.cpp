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
    real4 result;
    result.x = 0.0;
    result.y = 0.0;
    result.z = 0.0;
    result.w = 0.0;
    for (size_t i(0); i != pos.size(); ++i)
    {
    	mass = pos[i].w;
        result.x += mass * pos[i].x;
        result.y += mass * pos[i].y;
        result.z += mass * pos[i].z;
        result.w += mass;
    }
    result.x /= result.w;
    result.y /= result.w;
    result.z /= result.w;
    return result;
}

real4 Galaxy::getTotalVelocity() const
{
    real4 result;
    result.x = 0.0;
    result.y = 0.0;
    result.z = 0.0;
    result.w = 0.0;
    for (size_t i(0); i != vel.size(); ++i)
    {
        result.x += vel[i].x;
        result.y += vel[i].y;
        result.z += vel[i].z;
    }
    result.x /= vel.size();
    result.y /= vel.size();
    result.z /= vel.size();
    return result;
}

void Galaxy::centering()
{
    real4 center_of_mass = getCenterOfMass();
    for (size_t i(0); i != pos.size(); ++i)
    {
        pos[i].x -= center_of_mass.x;
        pos[i].y -= center_of_mass.y;
        pos[i].z -= center_of_mass.z;
    }
}

void Galaxy::steady()
{
    real4 total_velocity = getTotalVelocity();
    for (size_t i(0); i != vel.size(); ++i)
    {
        vel[i].x -= total_velocity.x;
        vel[i].y -= total_velocity.y;
        vel[i].z -= total_velocity.z;
    }
}

void Galaxy::add_position(real x, real y, real z)
{
    for (size_t i(0); i != pos.size(); ++i)
    {
        pos[i].x += x;
        pos[i].y += y;
        pos[i].z += z;
    }
}

void Galaxy::add_velocity(real x, real y, real z)
{
    for (size_t i(0); i != vel.size(); ++i)
    {
        vel[i].x += x;
        vel[i].y += y;
        vel[i].z += z;
    }
}
