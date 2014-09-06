#include "BonsaiIO.h"
#include "IDType.h"
#include <cmath>
#include <array>

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
      fprintf(stderr, " %s  inputFileName outputFileName \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string inputFn(argv[1]);
  const std::string outputFn(argv[2]);

  FILE *fin = fopen(inputFn.c_str(), "rb");
  assert(fin);


  int io_ver = 0;
  int n = 0;
  int ngas = 0;

  float omega0;
  float omegab;
  float lambda0;
  float hubble0;
  float astart;
  float anow;
  float tnow;
  double lunit;
  double munit;
  double tunit;

  fread( &io_ver , sizeof(int),    1, fin);
  fread( &n  , sizeof(int),    1, fin);
  fread( &ngas   , sizeof(int),    1, fin);
  fread( &omega0 , sizeof(float),  1, fin);
  fread( &omegab , sizeof(float),  1, fin);
  fread( &lambda0, sizeof(float),  1, fin);
  fread( &hubble0 , sizeof(float),  1, fin);
  fread( &astart , sizeof(float),  1, fin);
  fread( &anow   , sizeof(float),  1, fin);
  fread( &tnow   , sizeof(float),  1, fin);
  fread( &lunit  , sizeof(double), 1, fin);
  fread( &munit  , sizeof(double), 1, fin);
  fread( &tunit  , sizeof(double), 1, fin);

  fprintf(stderr, "n= %d\n", n);

  using vec3 = std::array<float,3>;
  std::vector<vec3> posV(n), velV(n);
  fread(&posV[0], sizeof(vec3), n, fin);
  fread(&velV[0], sizeof(vec3), n, fin);

  
  const int pCount = n; 
  BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputFn);

  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      ID[i].setID(i);
      ID[i].setType(1);
    }
    out.write(ID);
  }
  
  /* write pos */
  {
    typedef float vec4[4];
    BonsaiIO::DataType<vec4> pos("Stars:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
    {
      pos[i][0] = posV[i][0] - 0.5f;
      pos[i][1] = posV[i][1] - 0.5f;
      pos[i][2] = posV[i][2] - 0.5f;
      pos[i][3] = 1.0f/(float)pCount;
    }
    out.write(pos);
  }
    
  /* write vel */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("Stars:VEL:float[3]",pCount);
    for (int i = 0; i< pCount; i++)
    {
      vel[i][0] = velV[i][0];
      vel[i][1] = velV[i][1];
      vel[i][2] = velV[i][2];
    }
    out.write(vel);
  }


  out.close();


  MPI_Finalize();




  return 0;
}


