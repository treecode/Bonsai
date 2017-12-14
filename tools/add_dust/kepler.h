#ifndef __KEPLER_H__
#define __KEPLER_H__

#include <iostream>
#include <cassert>
#include "vector3.h"

struct Kepler
{
	typedef vector3<double> vec3;
  vec3 pos, vel;
	double M;

	double get_dt(const double eps) const
	{
		const double R = pos.abs();
		return eps * std::sqrt(R*R*R/M);
	}

	vec3 get_acc() const
	{
		const double R = pos.abs();
		return -M/(R*R*R)*pos;
	}
 
  Kepler(
			const vec3 &_pos, 
			const vec3 &_vel, 
			const double _M, 
			const double T,  
			const double eps = 1.0e-5) :
    pos(_pos), vel(_vel), M(_M)
  {
		double t = 0;
		double dt = get_dt(eps);
		while (t < T)
		{
			pos += vel*dt*0.5;
			vel += get_acc()*dt;
			pos += vel*dt*0.5;
			t += dt;
			dt = get_dt(eps);
		}
	}

};


#endif // __KEPLER_H__
