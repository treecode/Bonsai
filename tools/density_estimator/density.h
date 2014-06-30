#ifndef __DENSITY_H__
#define __DENSITY_H__

#include <algorithm>
#include "Node.h"
#include "wtime.h"


#if 0
#define SLOW
#endif

struct Density
{
  typedef boundary<float> Boundary;

  Particle::Vector density;
  Boundary BBox;  /* bounding box */

  struct cmp_particle_key 
  {
    bool operator () (const Particle &a, const Particle &b)
    {
      return a.key.val < b.key.val;
    }
  };

  struct cmp_particle_ID
  {
    bool operator () (const Particle &a, const Particle &b)
    {
      return a.ID < b.ID;
    }
  };

  Density(const Particle::Vector &ptcl_in, const int Nuse, const int Nngb = 32)
  {
    const double t0 = wtime();

    std::vector<Particle> &ptcl = Node::ptcl;
    ptcl.reserve(Nuse);
    const int Nin = ptcl_in.size();
    assert(Nuse <= Nin);

    const float fac = std::max(1.0f, (float)Nin/(float)Nuse);


    /* import particles and compute the Bounding Box */
    for (int i = 0; i < Nuse; i++)
    {
      ptcl.push_back(ptcl_in[(int)(i * fac)]);
      BBox.merge(Boundary(ptcl.back().pos));
    }
    std::cerr << BBox.min << std::endl;
    std::cerr << BBox.max << std::endl;
    const vec3  vsize = BBox.hlen();
    const float rsize = std::max(vsize.x, std::max(vsize.x, vsize.y)) * 2.0f;

    /* now build the tree */

    const int nbody = Nuse;

    for (int i = 0; i < nbody; i++)
      ptcl[i].compute_key(BBox.min, rsize);

    std::sort(ptcl.begin(), ptcl.end(), cmp_particle_key());

    Node::Node_heap.push_back(Node());
    Node &root = Node::Node_heap[0];
    for (int i = 0; i < nbody; i++)
      root.push_particle(i, 60);

#if 1  /* if h's are not know this set-up estimated range */
    const float volume = rsize*rsize*rsize;
    root.set_init_h(float(Nngb), volume);
#endif
    root.make_boundary();

#ifdef SLOW

#pragma omp parallel for
    for(int i=0; i<nbody; i++)
      ptcl[i] << root;
#else /* FAST */

#ifdef _OPENMP
    std::vector<Node *> group_list;
    root.find_group_Node(2000, group_list);
#pragma omp parallel for schedule(dynamic)
    for(int i=0; i<(int)group_list.size(); i++)
      *group_list[i] << root;
#else
    root << root;
#endif /* _OPENMP */

#endif /* SLOW */

    const double t1 = wtime();
    fprintf(stderr, " -- Density done in %g sec [ %g ptcl/sec ]\n",  t1 - t0, Nuse/(t1 - t0));

  };
};


#endif /* __DENSITY_H__ */
