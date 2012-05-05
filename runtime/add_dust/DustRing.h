#ifndef _DUST_RING_H_
#define _DUST_RING_H_

#include <iostream>
#include <vector>
#include <cassert>
#include <algorithm>
#include "vector3.h"

  
typedef float real;
typedef vector3<real> vec3;

inline real SQR(const real x) {return x*x;}

struct Vel1D
{
  typedef std::vector<Vel1D> Vector;
  typedef std::vector<Vel1D>::      iterator      Iterator;
  typedef std::vector<Vel1D>::const_iterator constIterator;
  real R, V;
  Vel1D(const real _R = 0, const real _V = 0) : R(_R), V(_V) {}
};

struct cmp_Vel1D
{
  bool operator() (const Vel1D &a, const Vel1D &b)
  {
    return a.R < b.R;
  }
};

struct Particle
{
  typedef std::vector<Particle> Vector;
  typedef std::vector<Particle>::      iterator      Iterator;
  typedef std::vector<Particle>::const_iterator constIterator;
  vec3 pos, vel;
  Particle(const vec3 &_pos, const vec3 &_vel = 0.0) : pos(_pos), vel(_vel) {}
};


struct Rotation
{
  real Axx, Axy, Axz;
  real Ayx, Ayy, Ayz;
  real Azx, Azy, Azz;
  Rotation(const real I = 0.0)
  {
    const real costh = cos(I);
    const real sinth = sin(I);
    const real costhm = 1.0 - costh;

    const vec3 u(0.0, 1.0, 0.0);  /* rotation around Y-axis, by angle I */

    Axx = u.x*u.x*costhm +     costh;
    Axy = u.x*u.y*costhm - u.z*sinth;
    Axz = u.x*u.z*costhm + u.y*sinth;

    Ayx = u.y*u.x*costhm + u.z*sinth;
    Ayy = u.y*u.y*costhm +     costh;
    Ayz = u.y*u.z*costhm - u.x*sinth;

    Azx = u.z*u.x*costhm - u.y*sinth;
    Azy = u.z*u.y*costhm + u.x*sinth;
    Azz = u.z*u.z*costhm +     costh;
  }

  vec3 rotate(const vec3 &v) const
  {
    return vec3(
        Axx*v.x + Axy*v.y + Axz*v.z,
        Ayx*v.x + Ayy*v.y + Ayz*v.z,
        Azx*v.x + Azy*v.y + Azz*v.z);
  }

  void rotate(Particle::Vector &ptcl) const
  {
    for (Particle::Iterator it = ptcl.begin(); it != ptcl.end(); it++)
    {
      it->pos = rotate(it->pos);
      it->vel = rotate(it->vel);
    }
  }

};
  
inline real GaussianDistribution(const real x, const real sigma)
{
  return std::exp((-0.5)*x*x/(sigma*sigma));
}


struct DustRing
{
  enum RingType {CYLINDER, TORUS};
  real Ro;   /* ring central  region*/
  real D;    /* ring radial   extent */
  real H;    /* ring vertical extend */
  real I;    /* ring inclination */
  Vel1D::Vector VelCurve; 
  RingType ring_type;
  Particle::Vector ptcl;

  DustRing(
      const int N, 
      const real _Ro,              /* ring's central region */
      const real _D,               /* density scale height in radial direction */
      const real _H,               /* density scale height in vertical direction */
      const real _I,               /* ring inclination */
      const Vel1D::Vector &_VelCurve, /* velocity curve to assign velocities to dust particles */
      const real nrScale = 3.0,    /* number of scale-height in z-direction */
      const real nzScale = 3.0,    /* number of scale-height in R-direction */
      const RingType &type = CYLINDER) : 
    Ro(_Ro), D(_D), H(_H), I(_I), VelCurve(_VelCurve), ring_type(type)
  {
    assert(N > 0);
    std::sort(VelCurve.begin(), VelCurve.end(), cmp_Vel1D());

#if 1
    const real Rmin = VelCurve[0].R;
    const real Rmax = VelCurve.back().R;

    assert(Rmax       > Rmin); /* sanity check, outer edge of the disk must be outside the inner */
    assert(Ro-nzScale > Rmin); /* check that there are enough velocity data points */
    assert(Ro+nzScale < Rmax); /* to cover the full ring particle distribution */
#endif

    ptcl.reserve(N);
    switch(type)
    {
      case CYLINDER:
        CylinderDust(nrScale, nzScale);
        break;
      case TORUS:
        TorusDust(nrScale, nzScale);
        break;
      default:
        CylinderDust(nrScale, nzScale);
    };

    const Rotation Mat(I);
    Mat.rotate(ptcl);
  }

  protected:

  void CylinderDust(const int nrScale, const int nzScale)
  {
    fprintf(stderr, " Generating ring of disk with CYLINDER particle distribution \n");

    while (ptcl.size() != ptcl.capacity())
    {
      const real M = 2.0*Ro;
      const real f = drand48() * M;

      const real R = Ro + nrScale * D * (1.0 - 2.0*drand48());
      const real Z =      nzScale * H * (1.0 - 2.0*drand48());

      const real nR = GaussianDistribution(R-Ro, D);
      const real nZ = GaussianDistribution(Z,    H);
      const real n  = nR*nZ;
      const real dV = R;
      const real dN = n * dV;

      assert(dN < M);

      if (f < n)  /* use von Neuman rejection technique to test whether to accept the position */
      {
        const real phi = drand48() * 2.0 * M_PI;
        ptcl.push_back(Particle(vec3(R*cos(phi), R*sin(phi), Z)));
      }
    }

#if 1
    for (Particle::Iterator it = ptcl.begin(); it != ptcl.end(); it++)
    {
      Particle &p = *it;

      const real R = std::sqrt(p.pos.x*p.pos.x + p.pos.y*p.pos.y);
      const Vel1D &lo = *(std::lower_bound(VelCurve.begin(), VelCurve.end(), R, cmp_Vel1D())-1);
      const Vel1D &up = *(std::lower_bound(VelCurve.begin(), VelCurve.end(), R, cmp_Vel1D())+1);

      /* linearly interpolate between two bounds */
      const real V = lo.V + (R - lo.R) * (up.V - lo.V)/(up.R - lo.R);
#if 0
      fprintf(stderr, "R= %g  V= %g\n", R, V);
      fprintf(stderr, "lo: R= %g  V= %g\n", lo.R, lo.V);
      fprintf(stderr, "up: R= %g  V= %g\n", up.R, up.V);
#endif

      assert(V >= std::min(up.V, lo.V));
      assert(V <= std::max(up.V, lo.V));

      const real cosph = p.pos.x/R;
      const real sinph = p.pos.y/R;
      p = Particle(p.pos, vec3(-V*sinph, V*cosph, 0.0f));
    }
#endif
  }
  
  void TorusDust(const int nrScale, const int nzScale)
  {
    fprintf(stderr, " Generating ring of disk with TORUS particle distribution \n");

    while (ptcl.size() != ptcl.capacity())
    {
      const real M = 4.0*Ro;
      const real f = drand48() * M;

      /* sample point in R-Z section of the "stretched" torus */
      const real r     = nrScale * D * drand48();
      const real th    = 2*M_PI*drand48();
      const real costh = cos(th);
      const real sinth = sin(th);

      const real x   = r*costh;
      const real z   = r*sinth;
      const real eps = std::sqrt(SQR(x/D) + SQR(z/H));

      const real n   = GaussianDistribution(eps, 1.0);
      const real dV  = r*(Ro + x);
      const real dN  = n*dV;

      assert(dN < M);

      if (f < n)  /* use von Neuman rejection technique to test whether to accept the position */
      {
        const real Z   = z;
        const real R   = x + Ro;
        const real phi = drand48() * 2.0 * M_PI;
        ptcl.push_back(Particle(vec3(R*cos(phi), R*sin(phi), Z)));
      }
    }

#if 1
    for (Particle::Iterator it = ptcl.begin(); it != ptcl.end(); it++)
    {
      Particle &p = *it;

      const real R = std::sqrt(p.pos.x*p.pos.x + p.pos.y*p.pos.y);
      const Vel1D &lo = *(std::lower_bound(VelCurve.begin(), VelCurve.end(), R, cmp_Vel1D())-1);
      const Vel1D &up = *(std::lower_bound(VelCurve.begin(), VelCurve.end(), R, cmp_Vel1D())+1);

      /* linearly interpolate between two bounds */
      const real V = lo.V + (R - lo.R) * (up.V - lo.V)/(up.R - lo.R);
#if 0
      fprintf(stderr, "R= %g  V= %g\n", R, V);
      fprintf(stderr, "lo: R= %g  V= %g\n", lo.R, lo.V);
      fprintf(stderr, "up: R= %g  V= %g\n", up.R, up.V);
#endif

      assert(V >= std::min(up.V, lo.V));
      assert(V <= std::max(up.V, lo.V));

      const real cosph = p.pos.x/R;
      const real sinph = p.pos.y/R;
      p = Particle(p.pos, vec3(-V*sinph, V*cosph, 0.0f));
    }
#endif
  }

};

#endif /* _DUST_RING_H_ */
