#include <mpi.h>
#include "read_tipsy.h"


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
      fprintf(stderr, " %s  baseName nDomains reduceFactorFirst[1] reduceFactorSecond[1]\n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string baseName(argv[1]);
  const int nDomains = atoi(argv[2]);

  int reduceFactorFirst  = 1;
  int reduceFactorSecond = 1;

  if (argc > 3)
    reduceFactorFirst = atoi(argv[3]);
  if (argc > 4)
    reduceFactorSecond = atoi(argv[4]);

  if (rank == 0)
  {
    fprintf(stderr, " reduceFactorFirst=  %d\n", reduceFactorFirst);
    fprintf(stderr, " reduceFactorSecond= %d\n", reduceFactorSecond);
  }

#if 0
  reduceFactorFirst  = 10;
  reduceFactorSecond = 2;
#endif

  ReadTipsy data(
      baseName, 
      rank, nranks,
      nDomains, 
      reduceFactorFirst,
      reduceFactorSecond);

  long long nFirstLocal = data.firstID.size();
  long long nSecondLocal = data.secondID.size();

  long long nFirst, nSecond;
  MPI_Allreduce(&nFirstLocal, &nFirst, 1, MPI_LONG, MPI_SUM, comm);
  MPI_Allreduce(&nSecondLocal, &nSecond, 1, MPI_LONG, MPI_SUM, comm);

  if (rank == 0)
  {
    fprintf(stderr, " nFirst = %lld \n", nFirst);
    fprintf(stderr, " nSecond= %lld \n", nSecond);
    fprintf(stderr, " nTotal= %lld \n", nFirst + nSecond);
  }


  MPI_Finalize();




  return 0;
}


