#pragma once

#include <parallel/algorithm>
#include "Node.h"
#include "wtime.h"


#if 0
#define SLOW
#endif

struct Tree
{
  typedef boundary<float> Boundary;

  Particle::Vector tree;
  Boundary BBox;  /* bounding box */
  std::vector<Node *> leafArray;

  struct cmp_particle_key { bool operator() (const Particle &a, const Particle &b) {return a.key.val < b.key.val;} };


  Tree(const Particle::Vector &ptcl_in, const int Nngb = -1)
  {
    const double t0 = wtime();

    Node::clear();
    std::vector<Particle> &ptcl = Node::ptcl;
    ptcl = ptcl_in;
    const int nbody = ptcl_in.size();

    /* import particles and compute the Bounding Box */
    for (int i = 0; i < nbody; i++)
      BBox.merge(Boundary(ptcl[i].pos));

    


    std::cerr << BBox.min << std::endl;
    std::cerr << BBox.max << std::endl;

    const vec3  vsize = BBox.hlen();
    const float rsize = std::max(vsize.x, std::max(vsize.x, vsize.y)) * 2.0f;

    float rsize2 = 1.0;
    while (rsize2 > rsize) rsize2 *= 0.5;
    while (rsize2 < rsize) rsize2 *= 2.0;

    /* now build the tree */

    for (int i = 0; i < nbody; i++)
      ptcl[i].compute_key(BBox.min, rsize2);

    __gnu_parallel::sort(ptcl.begin(), ptcl.end(), cmp_particle_key());

    Node::Node_heap.push_back(Node());
    Node &root = Node::Node_heap[0];
    root.size = rsize2;
    for (int i = 0; i < nbody; i++)
      root.push_particle(i, 60);
    
    const float volume = rsize*rsize*rsize;
    root.set_init_h(float(Nngb), volume);
    
    root.find_group_Node(Node::NLEAF, leafArray);
    const double t1 = wtime();
    fprintf(stderr, " -- Tree build is done in %g sec [ %g ptcl/sec ]\n",  t1 - t0, nbody/(t1 - t0));

    const int niter = 10;
    if (Nngb > 0)
      for (int iter = 0; iter< niter; iter++)
      {
        root.make_boundary();
#pragma omp parallel for
        for (int i = 0; i < nbody; i++)
          ptcl[i].nnb = 0;

#if 0  /* SLOW */
#pragma omp parallel for
        for(int i=0; i<nbody; i++)
          ptcl[i] << root;
#else /* FAST */
        std::vector<Node *> group_list;
        root.find_group_Node(2000, group_list);
#pragma omp parallel for schedule(dynamic)
        for(int i=0; i<(int)group_list.size(); i++)
          *group_list[i] << root;
#endif
        using long_t = unsigned long long;
        long_t nbMean = 0;
        long_t nbMax  = 0;
        long_t nbMin  = 1<<30;
#pragma omp parallel for reduction(+:nbMean) reduction(max:nbMax) reduction(min:nbMin)
        for (int i = 0; i < nbody; i++)
        {
          const float f = 0.5f * (1.0f + cbrtf(Nngb / (float)ptcl[i].nnb));
          const float fScale = std::max(std::min(f,2.0f), 0.8f);
          ptcl[i].set_h(ptcl[i].get_h() * fScale);

          nbMean += ptcl[i].nnb;
          nbMax   = std::max(nbMax, (long_t)ptcl[i].nnb);
          nbMin   = std::min(nbMin, (long_t)ptcl[i].nnb);
        }
        fprintf(stderr, "iteration= %d : nbMin= %g  nbMean= %g  nbMax= %g\n", 
            iter, (float)nbMin, (float)nbMean/nbody, (float)nbMax);
      }

    const double t2 = wtime();
    fprintf(stderr, " -- Ngb find is done in %g sec [ %g ptcl/sec ]\n",  t2 - t1, nbody/(t2 - t1));

  };
};


