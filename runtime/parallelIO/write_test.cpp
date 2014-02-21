#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <vector>
#include <string>
#include <sys/time.h>

extern "C" {
int llapi_file_create(const char *name, unsigned long long stripe_size,
                             int stripe_offset, int stripe_count,
                             int stripe_pattern);
int llapi_file_open(const char *name, int flags, int mode,
                           unsigned long long stripe_size, int stripe_offset,
                           int stripe_count, int stripe_pattern);
};

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
  assert(n > 16);

  MPI_Init(&argc, &argv);
  int rank, nrank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank); 
  MPI_Comm_size (MPI_COMM_WORLD, &nrank);
  size_t nbytes;
  double t0,t1;
  const MPI_Comm MPI_WORKING_WORLD = MPI_COMM_WORLD;

  std::vector<real4> pos(n), vel(n);
  std::vector<int> IDs(n);

  if (rank == 0) fprintf(stderr,  " -- writing %d particles -- \n", n);

  for (int i = 0; i < n ; i++)
  {
    const float fi = i;
    pos[i] = (real4){     fi,      fi+1.0f,      fi-1.0f,      -fi-1.0f};
    vel[i] = (real4){2.0f*fi, 2.0f*fi+1.0f, 2.0f*fi-1.0f, -2.0f*fi-1.0f};
    IDs[i] = 3*i-2;
  }

  const float time = 0.0;

  std::string fileName; fileName.resize(256);

  if (rank == 0) printf("Naive test\n");
  MPI_Barrier(MPI_WORKING_WORLD);
  t0 = rtc();
  sprintf(&fileName[0], "%s_%010.4f-%d", "naive_test", time, rank);
  llapi_file_create(&fileName[0], 0, -1, 1, 0);
  nbytes = write_snapshot(
      &pos[0], &vel[0], &IDs[0], n, fileName, time,
      rank, nrank, MPI_WORKING_WORLD);
  t1 = rtc();

  if (rank == 0)
    fprintf(stderr, " -- Naive writing took %g sec -- BW= %g MB/s\n",
        (t1-t0), nrank*nbytes/1e6/(t1-t0));


#if defined(SION_MPI)
  if (rank == 0) printf("SION tests\n");
  MPI_Barrier(MPI_WORKING_WORLD);
  t0 = rtc();
  sprintf(&fileName[0], "%s", "sion_test");
  if (rank == 0) llapi_file_create(&fileName[0], 0, -1, -1, 0);
  nbytes = sion_write_snapshot(
      &pos[0], &vel[0], &IDs[0], n, fileName, time,
      rank, nrank, 1, MPI_WORKING_WORLD);
  t1 = rtc();

  if (rank == 0)
    fprintf(stderr, " -- SION writing 1 file took %g sec -- BW= %g MB/s\n",
        (t1-t0), nrank*nbytes/1e6/(t1-t0));
#endif

  MPI_Finalize();

}
