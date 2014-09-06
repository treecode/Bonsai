#include "BonsaiIO.h"
#include <cmath>

int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);


  if (argc < 3)
  {
    if (rank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  inputFileName outputFileName Lclip \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string inputFn(argv[1]);
  const std::string outputFn(argv[2]);

  const double Lclip = argc > 3 ? atof(argv[3]) : HUGE;
  fprintf(stderr, "Lclip= %g\n", Lclip);

  FILE *fin = fopen(inputFn.c_str(), "rb");
  assert(fin);

  struct __attribute__((__packed__)) header_t
  {
    int ntot;
    int nnopt;
    double hmin;
    double hmax;
    double sep0;
    double tf;
    double dtout;
    int nout;
    int nit;
    double t;
    int anv;
    double alpha;
    double beta;
    double tskip;
    int ngr;
    int nrelax;
    double trelax;
    double dt;
    double omega2;
  };

  int idum; 
  header_t h;
  fread(&idum,  sizeof(int), 1, fin); 
  assert(idum == (int)sizeof(header_t));

  fread(&h, sizeof(header_t), 1, fin);

  fread(&idum,  sizeof(int), 1, fin); 
  assert(idum == (int)sizeof(header_t));

  fprintf(stderr, "ntot= %d  t= %g \n", h.ntot, h.t);

  struct __attribute__((__packed__)) sph_t
  {
    double x,y,z;
    double am,hp,rho;
    double vx,vy,vz;
    double vxdot,vydot,vzdot;
    double u,udot;
    double grpot, mmu;
    int cc;
    double divv;
  };

  std::vector<sph_t> data;
  data.reserve(h.ntot);

  double min[3] = { HUGE};
  double max[3] = {- HUGE};
  for (int i = 0; i < h.ntot; i++)
  {
    fread(&idum,  sizeof(int), 1, fin); 
    assert(idum == (int)sizeof(sph_t));

    sph_t d;
    fread(&d, sizeof(sph_t), 1, fin);
    
    fread(&idum,  sizeof(int), 1, fin); 
    assert(idum == (int)sizeof(sph_t));

    if (!(
        d.x >= -Lclip && d.x < Lclip &&
        d.y >= -Lclip && d.y < Lclip &&
        d.z >= -Lclip && d.z < Lclip))
      continue;


    data.push_back(d);


    min[0] = std::min(min[0], data[i].x);
    min[1] = std::min(min[1], data[i].y);
    min[2] = std::min(min[2], data[i].z);
    
    max[0] = std::max(max[0], data[i].x);
    max[1] = std::max(max[1], data[i].y);
    max[2] = std::max(max[2], data[i].z);
  }

  fprintf(stderr, "min= %g %g %g \n", min[0], min[1], min[2]);
  fprintf(stderr, "max= %g %g %g \n", max[0], max[1], max[2]);

  fread(&idum,  sizeof(int), 1, fin); 
  assert(idum == (int)sizeof(int));

  fread(&idum, sizeof(int), 1, fin);
  assert(idum == h.ntot);

  fread(&idum,  sizeof(int), 1, fin); 
  assert(idum == (int)sizeof(int));

  fprintf(stderr, "n= %d\n", (int)data.size());
   
  BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputFn);

  /* write metadata */
  {
    BonsaiIO::DataType<header_t> el("SPH:header:jamieHeader_t", 1);
    el[0] = h;
    out.write(el);
  }

  /* write pos */
  {
    BonsaiIO::DataType<sph_t> el("SPH:data:jamieData_t", data.size());
    for (int i = 0; i < (int)data.size(); i++)
      el[i] = data[i];
    out.write(el);
  }

  out.close();


  MPI_Finalize();




  return 0;
}


