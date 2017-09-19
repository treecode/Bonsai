#include "density.h"

std::vector<Particle> Node::ptcl;
std::vector<Node>     Node::Node_heap;
std::vector<std::pair<Node*, Node*> > Node::pair_list;

int main(int argc, char * argv[])
{
  Particle::Vector ptcl;

  int idum, nbody;
  std::cin >> idum >> nbody;

  std::vector<int> nnb(nbody);
  ptcl.reserve(nbody);

  for(int i=0; i<nbody; i++){
    float h;
    vec3 pos;
    std::cin >> pos >> h >> nnb[i];
    ptcl.push_back(Particle(i, pos, 1.0, 2.0*h));
  }
  fprintf(stderr, "nbody= %d \n", nbody);

  Node::allocate(nbody, nbody);
  Density density(ptcl, nbody);

  int ngb_min = nbody;
  int ngb_max = 0;
  double ngb_mean = 0;
  double ngb_mean2 = 0;

  int imax = 0;
#if 0
  fprintf(stdout, "%d\n", nbody);
#endif
  for (int i = 0; i < nbody; i++)
  {
    std::vector<Particle> &ptcl = Node::ptcl;
    const Particle &p = ptcl[i];
#if 0
    if (!(p.nnb == nnb[p.ID]))
    {
      fprintf(stderr, "i= %d: nnb_exact= %d  nnb_compute= %d\n",
          i, nnb[p.ID], p.nnb);
    }
#endif

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
    fprintf(stdout, " %d  %g %g %g   %g \n", p.ID, p.pos.x, p.pos.y, p.pos.z, p.density);

  }
  ngb_mean  *= 1.0/(float)nbody;
  ngb_mean2 *= 1.0/(float)nbody;
  fprintf(stderr, " imax= %d \n", imax);
  fprintf(stderr, " nmin= %d  nmax= %d   nmean= %g  <sigma>= %g\n",
      ngb_min, ngb_max,
      ngb_mean,
      std::sqrt(ngb_mean2 - ngb_mean*ngb_mean));

  return 0;
}


