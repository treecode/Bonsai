#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>
#include <sys/time.h>
static inline double rtc(void)
{
  struct timeval Tvalue;
  double etime;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  etime =  (double) Tvalue.tv_sec +
    1.e-6*((double) Tvalue.tv_usec);
  return etime;
}

struct real4
{
  float x,y,z,w;
};

#include "write_snapshot.h"
#include "sion_write_snapshot.h"

int main(int argc, char * argv [])
{
  const int n = argc > 1 ? atoi(argv[1]) : 1000000;
  fprintf(stderr,  " -- writing %d particles -- \n", n);
  assert(n > 16);

  MPI_Init(&argc, &argv);
  int rank, nrank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
  MPI_Comm_size (MPI_COMM_WORLD, &nrank);

  const MPI_Comm MPI_WORKING_WORLD = MPI_COMM_WORLD;

  std::vector<real4> pos(n), vel(n);
  std::vector<int> IDs(n);

  for (int i = 0; i < n ; i++)
  {
    const float fi = i;
    pos[i] = (real4){     fi,      fi+1.0f,      fi-1.0f,      -fi-1.0f};
    vel[i] = (real4){2.0f*fi, 2.0f*fi+1.0f, 2.0f*fi-1.0f, -2.0f*fi-1.0f};
    IDs[i] = 3*i-2;
  }

  const float time = 0.125;

  std::string fileName; fileName.resize(256);
  MPI_Barrier(MPI_WORKING_WORLD);
  const double t0 = rtc();

#ifndef _SION_
  sprintf(&fileName[0], "%s_%010.4f-%d", "naive_test", time, rank);
  const size_t nbytes = write_snapshot(
      &pos[0], &vel[0], &IDs[0], n, fileName, time,
      rank, nrank, MPI_WORKING_WORLD);
#else
  sprintf(&fileName[0], "%s_%010.4f-%d", "sion_test", time, nrank);
  const size_t nbytes = sion_write_snapshot(
      &pos[0], &vel[0], &IDs[0], n, fileName, time,
      rank, nrank, MPI_WORKING_WORLD);
#endif

  MPI_Barrier(MPI_WORKING_WORLD);
  const double t1 = rtc();

  if (rank == 0)
    fprintf(stderr, " -- writing took %g sec -- BW= %g MB/s\n",
        (t1-t0), nbytes/1e6/(t1-t0));


  MPI_Finalize();
}
