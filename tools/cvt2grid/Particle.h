#ifndef __PARTICLE_H__
#define __PARTICLE_H__

#include <vector>
#include "vector3.h"
#include "morton_key.h"

class Particle
{
  public:
 
    typedef std::vector<Particle> Vector;
    typedef Vector::      iterator      Iterator;
    typedef Vector::const_iterator constIterator;

    int ID;                     // 1 1
    vec3 pos;                   // 3 4
    float mass;                 // 1 5
    vec3 vel;

  protected:
    float h;  /* range */       // 1 6

  public:
    float density;              // 1 7
    int   nnb;                  // 1 8
    morton_key<vec3, float> key; // 2 10

  protected:
    float hinv;                  // 1 11
    int iPad[5];

  public:

    /********* methods ******/

    Particle(const int _ID, const vec3 &_pos, const float _mass, const float _h = 0.0f) :
      ID(_ID), pos(_pos), mass(_mass), h(_h), density(0.0f), nnb(0) {  assert(sizeof(Particle) == 16*sizeof(float)); }
    Particle() {}

    void set_h(const float _h) 
    {
      h    = _h;
      hinv = 1.0f/h;
    }
    float get_h() const {return h;}
    float get_hinv() const {return hinv;}

    void compute_key(const vec3 &origin, const float size) 
    {
      key = morton_key<vec3, float>(pos - origin, size);
    };

    int octkey(const int rshift)
    {
      return 7 & (key.val >> rshift);
    }

#if 0 /* does not work w/ OpenMP, not sure why ... */
    const Particle operator = (const Particle &rhs)
    {
      typedef float v4sf __attribute__ ((vector_size(16)));
      v4sf *lp =(v4sf *)this;
      v4sf *rp =(v4sf *)(&rhs);
      lp[0] = rp[0];
      lp[1] = rp[1];
      lp[2] = rp[2];
      lp[3] = rp[3];
      
      return *this;
    }
#endif
};


#if 0
namespace std
{
	template <> 
	inline void iter_swap <Particle::Iterator, Particle::Iterator> (Particle::Iterator a, Particle::Iterator b)
  {
		typedef float v4sf __attribute__ ((vector_size(16)));
		v4sf *ap =(v4sf *)&(*a);
		v4sf *bp =(v4sf *)&(*b);
		v4sf tmpa0 = ap[0];
		v4sf tmpa1 = ap[1];
		v4sf tmpa2 = ap[2];
		v4sf tmpa3 = ap[3];
		v4sf tmpb0 = bp[0];
		v4sf tmpb1 = bp[1];
		v4sf tmpb2 = bp[2];
		v4sf tmpb3 = bp[3];
		ap[0] = tmpb0;
		ap[1] = tmpb1;
		ap[2] = tmpb2;
		ap[3] = tmpb3;
		bp[0] = tmpa0;
		bp[1] = tmpa1;
		bp[2] = tmpa2;
		bp[3] = tmpa3;
	}
}
#endif

#endif /* __PARTICLE_H__ */
