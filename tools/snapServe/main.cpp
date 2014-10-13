#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <unistd.h>
#include <sstream>
#include <cmath>
#include "IDType.h"
#include "BonsaiSharedData.h"
#include "BonsaiIO.h"
#include "SharedMemory.h"
#ifndef BONSAI_CATALYST_CLANG
 #include <omp.h>
#endif
#include <functional>

#include "anyoption.h"
#include "RendererData.h"

using ShmQHeader = SharedMemoryServer<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryServer<BonsaiSharedQuickData>;
static ShmQHeader *shmQHeader = NULL;
static ShmQData   *shmQData   = NULL;

struct data_t
{
  float posx,posy,posz, mass;
  float velx,vely,velz;
  float rho,h;
  IDType ID;
};
using DataVec = std::vector<data_t>;

static void sendSharedData(
    const bool sync, 
    const double t_current,
    const DataVec &rdata,
    const char fileName[],
    const int rank, 
    const int nrank, 
    const MPI_Comm &comm)
{
  if (shmQHeader == NULL)
  {
    const size_t capacity  = rdata.size()*2;
    shmQHeader = new ShmQHeader(ShmQHeader::type::sharedFile(rank), 1);
    shmQData   = new ShmQData  (ShmQData  ::type::sharedFile(rank), capacity);
  }

  auto &header = *shmQHeader;
  auto &data   = *shmQData;

  static bool handShake = false;
  if (sync && handShake) 
  {
    /* handshake */

    header.acquireLock();
    header[0].handshake = false;
    header.releaseLock();

    while (!header[0].handshake)
      usleep(100);

    header.acquireLock();
    header[0].handshake = false;
    header.releaseLock();

    /* handshake complete */
    handShake = true;
  }

  if (sync)
    while (!header[0].done_writing);


  const size_t np = rdata.size();
  
  header.acquireLock();
  header[0].tCurrent = t_current;
  header[0].nBodies  = np;
  for (int i = 0; i < 1024; i++)
  {
    header[0].fileName[i] = fileName[i];
    if (fileName[i] == 0)
      break;
  }

  data.acquireLock();
  if (!data.resize(np))
  {
    std::cerr << "rank= " << rank << ": failed to resize. ";
    std::cerr << "Request " << np << " but capacity is  " << data.capacity() << "." << std::endl;
    data.releaseLock();
    header.releaseLock();
    MPI_Finalize();
    ::exit(0);
  }

  for (size_t i = 0; i < np; i++)
  {
    auto &p       = data[i];
    const auto &d = rdata[i];
    p.x    = d.posx;
    p.y    = d.posy;
    p.z    = d.posz;
    p.mass = d.mass;
    p.vx   = d.velx;
    p.vy   = d.vely;
    p.vz   = d.velz;
    p.vw   = 0.0;
    p.rho  = d.rho;
    p.h    = d.h;
    p.ID   = d.ID;
  }
  data.releaseLock();

  header[0].done_writing = false;
  header.releaseLock();
}

static std::tuple<double,DataVec> readBonsai(
    const int rank, const int nranks, const MPI_Comm &comm,
    const std::string &fileName,
    const int reduceDM,
    const int reduceS,
    const bool print_header = false) 
{
  const double t0 = MPI_Wtime();
  BonsaiIO::Core in(rank, nranks, comm, BonsaiIO::READ, fileName);
  if (rank == 0 && print_header)
  {
    fprintf(stderr, "---- Bonsai header info ----\n");
    in.getHeader().printFields();
    fprintf(stderr, "----------------------------\n");
  }
  typedef float float4[4];
  typedef float float3[3];
  typedef float float2[2];

  BonsaiIO::DataType<IDType> IDListS("Stars:IDType");
  BonsaiIO::DataType<float4> posS("Stars:POS:real4");
  BonsaiIO::DataType<float3> velS("Stars:VEL:float[3]");
  BonsaiIO::DataType<float2> rhohS("Stars:RHOH:float[2]");

  DataVec rdata;
  if (reduceS > 0)
  {
    if (!in.read(IDListS, true, reduceS)) return std::make_tuple(-1.0,rdata);
    if (rank  == 0)
      fprintf(stderr, " Reading star data \n");
    assert(in.read(posS,    true, reduceS));
    assert(in.read(velS,    true, reduceS));
    bool renderDensity = true;
    if (!in.read(rhohS,  true, reduceS))
    {
      if (rank == 0)
      {
        fprintf(stderr , " -- Stars RHOH data is found \n");
        fprintf(stderr , " -- rendering stars w/o density info \n");
      }
      renderDensity = false;
    }
    assert(IDListS.getNumElements() == posS.getNumElements());
    assert(IDListS.getNumElements() == velS.getNumElements());
    if (renderDensity)
      assert(IDListS.getNumElements() == posS.getNumElements());
  }

  BonsaiIO::DataType<IDType> IDListDM("DM:IDType");
  BonsaiIO::DataType<float4> posDM("DM:POS:real4");
  BonsaiIO::DataType<float3> velDM("DM:VEL:float[3]");
  BonsaiIO::DataType<float2> rhohDM("DM:RHOH:float[2]");
  if (reduceDM > 0)
  {
    if (rank  == 0)
      fprintf(stderr, " Reading DM data \n");
    if(!in.read(IDListDM, true, reduceDM)) return std::make_tuple(-1.0,rdata);
    assert(in.read(posDM,    true, reduceDM));
    assert(in.read(velDM,    true, reduceDM));
    bool renderDensity = true;
    if (!in.read(rhohDM,  true, reduceDM))
    {
      if (rank == 0)
      {
        fprintf(stderr , " -- DM RHOH data is found \n");
        fprintf(stderr , " -- rendering stars w/o density info \n");
      }
      renderDensity = false;
    }
    assert(IDListS.getNumElements() == posS.getNumElements());
    assert(IDListS.getNumElements() == velS.getNumElements());
    if (renderDensity)
      assert(IDListS.getNumElements() == posS.getNumElements());
  }

  const double t1 = MPI_Wtime();

  const int nS  = IDListS.getNumElements();
  const int nDM = IDListDM.getNumElements();
  long long int nSloc = nS, nSglb;
  long long int nDMloc = nDM, nDMglb;

  MPI_Allreduce(&nSloc, &nSglb, 1, MPI_LONG, MPI_SUM, comm);
  MPI_Allreduce(&nDMloc, &nDMglb, 1, MPI_LONG, MPI_SUM, comm);
  if (rank == 0)
  {
    fprintf(stderr, "nStars = %lld\n", nSglb);
    fprintf(stderr, "nDM    = %lld\n", nDMglb);
  }


  rdata.resize(nS+nDM);
  for (int i = 0; i < nS; i++)
  {
    auto &d = rdata[i];
    d.posx = posS[i][0];
    d.posy = posS[i][1];
    d.posz = posS[i][2];
    d.mass = posS[i][3];
    d.ID   = IDListS[i];
    assert(d.ID.getType() > 0); /* sanity check */
    d.velx = velS[i][0];
    d.vely = velS[i][1];
    d.velz = velS[i][2];
    if (rhohS.size() > 0)
    {
      d.rho = rhohS[i][0];
      d.h   = rhohS[i][1];
    }
    else
    {
      d.rho = 0.0;
      d.h   = 0.0;
    }
  }

  const double bw = in.computeBandwidth()/1e6;
  const double dtRead = t1-t0;
  if (rank == 0)
    fprintf(stderr, " :: dtRead= %g  sec readBW= %g MB/s \n", dtRead, bw);
  const double t = in.getTime();
  in.close();
  return std::make_tuple(t,rdata);
}

std::vector<std::string> lParseList(const std::string fileNameList)
{
  std::vector<std::string> fileList;

  return fileList;
}

#ifndef BONSAI_CATALYST_CLANG
int main(int argc, char * argv[], MPI_Comm commWorld)
{
#else
int main(int argc, char * argv[])
{
 MPI_Comm commWorld;
#endif

  std::string fileNameList;
  int nloop    = 1;
  bool quickSync = true;
  int reduceDM = 10;
  int reduceS  = 1;

  {
    AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

    ADDUSAGE(" ");
    ADDUSAGE("Usage:");
    ADDUSAGE(" ");
    ADDUSAGE(" -h  --help             Prints this help ");
    ADDUSAGE(" -i  --inlist #         Input list with snapshot filenames");
    ADDUSAGE(" -l  --loop   #         Loop count through file list [1]");
    ADDUSAGE("     --noquicksync      disable syncing with the client ");
    ADDUSAGE("     --reduceDM    #    cut down DM dataset by # factor [10]. 0-disable DM");
    ADDUSAGE("     --reduceS     #    cut down stars dataset by # factor [1]. 0-disable S");


    opt.setFlag  ( "help" ,        'h');
    opt.setOption( "inlist",       'i');
    opt.setFlag  ( "loop",         'l');
    opt.setFlag  ( "noquicksync");
    opt.setOption( "reduceDM");
    opt.setOption( "reduceS");

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    char *optarg = NULL;
    if ((optarg = opt.getValue("inlist")))    fileNameList       = std::string(optarg);
    if ((optarg = opt.getValue("loop")))      nloop             = atoi(optarg);
    if ((optarg = opt.getValue("reduceDM"))) reduceDM       = atoi(optarg);
    if ((optarg = opt.getValue("reduceS"))) reduceS       = atoi(optarg);
    if (opt.getValue("noquicksync")) quickSync = false;

    if (fileNameList.empty() ||
        reduceDM < 0 || reduceS < 0)
    {
      opt.printUsage();
      ::exit(0);
    }

#undef ADDUSAGE
  }

  MPI_Comm comm = MPI_COMM_WORLD;
  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  if (!mpiInitialized)
    MPI_Init(&argc, &argv);
  else
    comm = commWorld;

  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int namelen;
  MPI_Get_processor_name(processor_name,&namelen);
  fprintf(stderr, "bonsai_snapServe:: Proc id: %d @ %s , total processes: %d (mpiInit) \n", rank, processor_name, nranks);

  const auto &fileList = lParseList(fileNameList);

  for (int i = 0; i < nloop; i++)
    for (const auto &file : fileList)
    {
      fprintf(stderr, "loop= %3d: filename= %s \n", i, file.c_str());
      const auto &data = readBonsai(rank, nranks, comm,
          file, reduceDM, reduceS);
      sendSharedData(quickSync, std::get<0>(data), std::get<1>(data), file.c_str(), rank, nranks, comm);
    }

  MPI_Finalize();

  return 0;
}


