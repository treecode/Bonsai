#include "read_tipsy.h"
#include "density.h"


std::vector<Particle> Node::ptcl;
std::vector<Node>     Node::Node_heap;
std::vector<std::pair<Node*, Node*> > Node::pair_list;

#if 1
#define DENSDM
#endif

int main(int argc, char * argv[])
{
  ReadTipsy data;
  Particle::Vector ptcl_star, ptcl_dm;

  const int nbody = data.NTotal;

  ptcl_dm  .reserve(nbody);
  ptcl_star.reserve(nbody);

  for(int i=0; i<nbody; i++)
  {
    const vec3 pos = vec3(data.positions[i].x, data.positions[i].y, data.positions[i].z);
    if (data.IDs[i] < (int)200e6) 
      ptcl_star.push_back(Particle(i, pos, 1.0, 0.0));
    else
      ptcl_dm.push_back(Particle(i, pos, 1.0, 0.0));
  }
  fprintf(stderr, "nbody= %d : star= %d  DM= %d\n", nbody,
      (int)ptcl_star.size(),
      (int)ptcl_dm  .size());


  Node::allocate(nbody, nbody);
#ifdef DENSDM
  const int N = (int)ptcl_dm.size();
  Density density(ptcl_dm, ptcl_dm.size(), 64);
#else
  const int N =(int)ptcl_star.size();
  Density density(ptcl_star, N, 64);
#endif

  int ngb_min = nbody;
  int ngb_max = 0;
  double ngb_mean = 0;
  double ngb_mean2 = 0;

  int imax = 0;
  int nzero = 0;
  for (int i = 0; i < N; i++)
  {
    std::vector<Particle> &ptcl = Node::ptcl;
    const Particle &p = ptcl[i];

    if (p.nnb > 128)
    {
#if 0
      fprintf(stderr, "i= %d: nnb_exact= %d  nnb_compute= %d  h= %g pos= %g %g %g\n",
          i, nnb[ID], p.nnb, p.h, p.pos.x, p.pos.y, p.pos.z);
#endif
      imax++;
    }

    ngb_min = std::min(ngb_min, p.nnb);
    ngb_max = std::max(ngb_max, p.nnb);
    ngb_mean  += p.nnb;
    ngb_mean2 += p.nnb*p.nnb;
    if (p.nnb < 3)
    {
      nzero++;
      continue;
    }
    fprintf(stdout, " %d  %g %g %g   %g  %d %d\n", 
        p.ID, p.pos.x, p.pos.y, p.pos.z, p.density, 
#ifdef DENSDM
        0,
#else
        1,
#endif
        p.nnb);

  }
  ngb_mean  *= 1.0/(float)N;
  ngb_mean2 *= 1.0/(float)N;
  fprintf(stderr, " imax= %d nzero= %d\n", imax, nzero);
  fprintf(stderr, " nmin= %d  nmax= %d   nmean= %g  <sigma>= %g\n",
      ngb_min, ngb_max,
      ngb_mean,
      std::sqrt(ngb_mean2 - ngb_mean*ngb_mean));

  return 0;
}


