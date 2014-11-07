#undef NDEBUG
#include <mpi.h>
#include <cuda_runtime_api.h>
#include <sstream>
#include "anyoption.h"
#include "SharedMemory.h"
#include "BonsaiSharedData.h"
#include "BonsaiIO.h"
#include "IDType.h"
#include <array>

using ShmQHeader = SharedMemoryClient<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryClient<BonsaiSharedQuickData>;
using ShmSHeader = SharedMemoryClient<BonsaiSharedSnapHeader>;
using ShmSData   = SharedMemoryClient<BonsaiSharedSnapData>;

#if 0
#define _DEBUG
#else
#undef _DEBUG
#endif

static double write(
    const int rank, const MPI_Comm &comm,
    const std::vector<BonsaiIO::DataTypeBase*> &data,
    BonsaiIO::Core &out)
{
  double dtWrite = 0;
  for (const auto &type : data)
  {
    double t0 = MPI_Wtime();
    long long int nLoc = type->getNumElements();
    long long int nGlb;
    MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (nGlb > 0)
      assert(out.write(*type));
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}

template<typename ShmHeader, typename ShmData>
bool writeLoop(ShmHeader &header, ShmData &data, const int rank, const int nrank, const MPI_Comm &comm)
{
  double tLast = -1;


  /* handshake */

  header.acquireLock();
  header[0].handshake = true;
  header.releaseLock();

  while (header[0].handshake)
    usleep(1000);

  header.acquireLock();
  header[0].handshake = true;
  header.releaseLock();

  /* handshake complete */

  while (1)
  {
    while (header[0].done_writing);

    assert(tLast != header[0].tCurrent);
    tLast = header[0].tCurrent;

    if (header[0].tCurrent == -1.0)
      break;

    header.acquireLock();
    const float tCurrent = header[0].tCurrent;
    const size_t nBodies = header[0].nBodies;
    char fn[1024];
    for (int i = 0; i < 1024; i++)
    {
      fn[i] = header[0].fileName[i];
      if (fn[i] == 0)
        break;
    }

    data.acquireLock();
    const size_t size = data.size();
    assert(size == nBodies);

    /* write data */

    {
      const double tBeg  = MPI_Wtime();
      const double tOpen = MPI_Wtime(); 
      BonsaiIO::Core out(rank, nrank, comm, BonsaiIO::WRITE, fn);
      double dtOpen = MPI_Wtime() - tOpen;

      out.setTime(tCurrent);
      tLast = tCurrent;

      /* prepare data */ 

      size_t nDM = 0, nS = 0;
      constexpr int ntypecount = 10;
      std::array<size_t,ntypecount> ntypeloc, ntypeglb;
      std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
      for (size_t i = 0; i < size; i++)
      {
        const int type = data[i].ID.getType();
        if  (type < ntypecount)
          ntypeloc[type]++;
        switch(type)
        {
          case 0:
            nDM++;
            break;
          default:
            nS++;
        }
      }
    
      MPI_Reduce(&ntypeloc, &ntypeglb, ntypecount, MPI_LONG_LONG, MPI_SUM, 0, comm);
      if (rank == 0)
      {
        fprintf(stderr, "writing to %s \n", fn);
        for (int type = 0; type < ntypecount; type++)
          if (ntypeglb[type] > 0)
            fprintf(stderr, "bonsai_io:: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
      }

      typedef float float4[4];
      typedef float float3[3];
      typedef float float2[2];
      BonsaiIO::DataType<IDType> DM_id ("DM:IDType",        nDM);
      BonsaiIO::DataType<float4> DM_pos("DM:POS:real4",     nDM);
      BonsaiIO::DataType<float3> DM_vel("DM:VEL:float[3]",  nDM);
      BonsaiIO::DataType<float2> DM_rhoh("DM:RHOH:float[2]", nDM);

      BonsaiIO::DataType<IDType> S_id ("Stars:IDType",        nS);
      BonsaiIO::DataType<float4> S_pos("Stars:POS:real4",     nS);
      BonsaiIO::DataType<float3> S_vel("Stars:VEL:float[3]",  nS);
      BonsaiIO::DataType<float2> S_rhoh("Stars:RHOH:float[2]", nS);


      size_t iDM = 0, iS = 0;
      for (size_t i = 0; i < size; i++)
      {
        assert(iDM + iS == i);
        switch (data[i].ID.getType())
        {
          case 0:
            DM_id  [iDM]    = data[i].ID;
            DM_pos [iDM][0] = data[i].x;
            DM_pos [iDM][1] = data[i].y;
            DM_pos [iDM][2] = data[i].z;
            DM_pos [iDM][3] = data[i].mass;
            DM_vel [iDM][0] = data[i].vx;
            DM_vel [iDM][1] = data[i].vy;
            DM_vel [iDM][2] = data[i].vz;
            DM_rhoh[iDM][0] = data[i].rho;
            DM_rhoh[iDM][1] = data[i].h;
            iDM++;
            assert(iDM <= nDM);
            break;
          default:
            S_id  [iS]    = data[i].ID;
            S_pos [iS][0] = data[i].x;
            S_pos [iS][1] = data[i].y;
            S_pos [iS][2] = data[i].z;
            S_pos [iS][3] = data[i].mass;
            S_vel [iS][0] = data[i].vx;
            S_vel [iS][1] = data[i].vy;
            S_vel [iS][2] = data[i].vz;
            S_rhoh[iS][0] = data[i].rho;
            S_rhoh[iS][1] = data[i].h;
            iS++;
            assert(iS <= nS);
        }
      }
      
      for (int i = 0; i < nDM; i++)
        assert(DM_id[i].getType() == 0);
      for (int i = 0; i < nS; i++)
        assert(S_id[i].getType() > 0);

      const double dtWrite = write(rank, comm, 
          {
            &DM_id, &DM_pos, &DM_vel, &DM_rhoh,  
            &S_id, &S_pos, &S_vel, &S_rhoh
          }, out);

      const double tClose = MPI_Wtime(); 
      out.close();
      double dtClose = MPI_Wtime() - tClose;

      const double writeBW = out.computeBandwidth();
      const double tEnd = MPI_Wtime();

      long long nGlb[2], nLoc[2];
      nLoc[0] = nDM;
      nLoc[1] = nS;
      MPI_Reduce(nLoc, nGlb, 2, MPI_LONG_LONG, MPI_SUM, 0, comm);


      if (rank == 0)
        fprintf(stderr, " BonsaiIO:: total= %g sec nDM= %gM  nS= %gM [open= %g  write= %g close= %g] BW= %g MB/s \n",
            tEnd-tBeg, nGlb[0]/1e6, nGlb[1]/1e6, dtOpen, dtWrite, dtClose, writeBW/1e6);
    }
    header[0].done_writing = true;

    data.releaseLock();
    header.releaseLock();
  }

}

int main(int argc, char * argv[], MPI_Comm commWorld)
{
  
  MPI_Comm comm = MPI_COMM_WORLD;

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (!mpiInitialized)
  {
#ifdef _MPIMT
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    assert(MPI_THREAD_MULTIPLE == provided);
#else
    MPI_Init(&argc, &argv);
#endif
  }
  else
    comm = commWorld;
    
  int rank, nrank;
  MPI_Comm_size(comm, &nrank);
  MPI_Comm_rank(comm, &rank);
	
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  MPI_Get_processor_name(processor_name,&namelen);
  fprintf(stderr, "bonsai_io:: Rank: %d @ %s , total ranks: %d (mpiInit) \n", rank, processor_name, nrank);
  bool snapDump = true;
  {
		AnyOption opt;
    

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -q  --quick            Write a subsampled snapshot [default is a full restart-quality snapshots]");
		
    opt.setFlag( "help" ,   'h');
    opt.setFlag( "quick",  'q');
    
    opt.processCommandArgs( argc, argv );

    
    if(opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    if (opt.getFlag("quick")) snapDump = false;
  }


  const std::string mode(snapDump  ? "SNAPSHOT" : "QUICKDUMP");
  if (rank == 0)
    fprintf(stderr, "BonsaIO :: %s mode. Use '-h' for help. \n", mode.c_str());

  if (snapDump)
  {
    ShmSHeader shmSHeader(ShmSHeader::type::sharedFile(rank));
    ShmSData   shmSData  (ShmSData  ::type::sharedFile(rank));
    writeLoop(shmSHeader, shmSData, rank, nrank, comm);
  }
  else
  {
    ShmQHeader shmQHeader(ShmQHeader::type::sharedFile(rank));
    ShmQData   shmQData  (ShmQData  ::type::sharedFile(rank));
    writeLoop(shmQHeader, shmQData, rank, nrank, comm);
  }

  if (rank == 0)
    fprintf(stderr , "BonsaiIO ::  %s mode done writing .. \n", mode.c_str());
  if (!mpiInitialized)
    MPI_Finalize();

  return 0;
}

