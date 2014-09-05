#include <mpi.h>
#include <cuda_runtime_api.h>
#include "SharedMemory.h"
#include "BonsaiSharedData.h"
#include "BonsaiIO.h"
#include "IDType.h"

using ShmQHeader = SharedMemoryClient<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryClient<BonsaiSharedQuickData>;
using ShmSHeader = SharedMemoryClient<BonsaiSharedSnapHeader>;
using ShmSData   = SharedMemoryClient<BonsaiSharedSnapData>;

static double write(
    const int rank, const MPI_Comm &comm,
    const std::vector<BonsaiIO::DataTypeBase*> &data,
    BonsaiIO::Core &out)
{
  double dtWrite = 0;
  for (const auto &type : data)
  {
    double t0 = MPI_Wtime();
    if (rank == 0)
      fprintf(stderr, " Writing %s ... \n", type->getName().c_str());
    long long int nLoc = type->getNumElements();
    long long int nGlb;
    MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (nGlb > 0)
    {
      if (rank == 0)
        fprintf(stderr, " Writing %lld of type %s\n",
            nGlb, type->getName().c_str());
      assert(out.write(*type));
      if (rank == 0)
        fprintf(stderr, " ---- \n");
    }
    else if (rank == 0)
    {
      fprintf(stderr, " %s is empty... not writing \n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}

template<typename ShmHeader, typename ShmData>
void writeLoop(ShmHeader &header, ShmData &data, const int rank, const int nrank, const MPI_Comm &comm)
{
  const float waittime = 1.0f; /* ms */
  static double tLast = -1;

  while (1)
  {
    while(header[0].tCurrent == tLast)
      usleep(static_cast<int>(waittime*1000));

    if (header[0].tCurrent == -1.0)
      break;

    header.acquireLock(waittime);
    const float tCurrent = header[0].tCurrent;
    const size_t nBodies = header[0].nBodies;
    char fn[1024];
    for (int i = 0; i < 1024; i++)
    {
      fn[i] = header[0].fileName[i];
      if (fn[i] == 0)
        break;
    }

    data.acquireLock(waittime);
    const size_t size = data.size();
    assert(size == nBodies);

    /* open file for writing */
  
    const double tOpen = MPI_Wtime(); 
    BonsaiIO::Core out(rank, nrank, comm, BonsaiIO::WRITE, fn);
    double dtOpen = MPI_Wtime() - tOpen;

    out.setTime(tCurrent);
    tLast = tCurrent;
     
    /* prepare data */ 

    size_t nDM = 0, nS = 0;
    for (size_t i = 0; i < size; i++)
    {
      switch(data[i].ID.getType())
      {
        case 0:
          nDM++;
          break;
        case 1:
          nS++;
          break;
        default:
          assert(0);
      }
    }

    typedef float float4[4];
    typedef float float3[3];
    typedef float float2[2];
    BonsaiIO::DataType<IDType> DM_id ("DM:IDType",       nDM);
    BonsaiIO::DataType<float4> DM_pos("DM:POS:real4",    nDM);
    BonsaiIO::DataType<float3> DM_vel("DM:VEL:float[3]", nDM);
   
    BonsaiIO::DataType<IDType> S_id ("Stars:IDType",       nS);
    BonsaiIO::DataType<float4> S_pos("Stars:POS:real4",    nS);
    BonsaiIO::DataType<float3> S_vel("Stars:VEL:float[3]", nS);

    std::vector<BonsaiIO::DataTypeBase*> 
      data2write = {&DM_id, &DM_pos, &DM_vel, &S_id, &S_pos, &S_vel};
        

    size_t iDM = 0, iS = 0;
    for (size_t i = 0; i < size; i++)
    {
      assert(iDM + iS == i);
      switch (data[i].ID.getType())
      {
        case 0:
          DM_id [iDM]    = data[i].ID;
          DM_pos[iDM][0] = data[i].x;
          DM_pos[iDM][1] = data[i].y;
          DM_pos[iDM][2] = data[i].z;
          DM_pos[iDM][3] = data[i].mass;
          DM_vel[iDM][0] = data[i].vx;
          DM_vel[iDM][1] = data[i].vy;
          DM_vel[iDM][2] = data[i].vz;
          iDM++;
          break;
        case 1:
          S_id [iS]    = data[i].ID;
          S_pos[iS][0] = data[i].x;
          S_pos[iS][1] = data[i].y;
          S_pos[iS][2] = data[i].z;
          S_pos[iS][3] = data[i].mass;
          S_vel[iS][0] = data[i].vx;
          S_vel[iS][1] = data[i].vy;
          S_vel[iS][2] = data[i].vz;
          iS++;
          break;
        default:
          assert(0);
      }
    }

    fprintf(stderr, "rank= %d : nDM= %d  nS= %d\n",
        rank, (int)nDM, (int)nS);
#if 0
    const double dtWrite = write(rank, comm, data2write, out);
#endif
    
    const double tClose = MPI_Wtime(); 
    out.close();
    double dtClose = MPI_Wtime() - tClose;
    
    const double writeBW = out.computeBandwidth();

    data.releaseLock();

    header.releaseLock();
  }

}

int main(int argc, char * argv[])
{
  
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int rank, nrank;
  MPI_Comm_size(comm, &nrank);
  MPI_Comm_rank(comm, &rank);

  bool snapDump = true;

  if (snapDump)
  {
    ShmSHeader shmSHeader(ShmSHeader::type::sharedFile());
    ShmSData   shmSData  (ShmSData  ::type::sharedFile());
    writeLoop(shmSHeader, shmSData, rank, nrank, comm);
  }
  else
  {
    ShmQHeader shmQHeader(ShmQHeader::type::sharedFile());
    ShmQData   shmQData  (ShmQData  ::type::sharedFile());
    writeLoop(shmQHeader, shmQData, rank, nrank, comm);
  }

  return 0;
}

