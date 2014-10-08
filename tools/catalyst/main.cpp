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
#include <omp.h>
#include <functional>

#include "anyoption.h"
#include "RendererData.h"

static void renderer(
    int argc, char** argv, 
    const int rank, const int nrank, const MPI_Comm &comm,
    RendererData &data,
    const char *fullScreenMode /* = "" */,
    const bool stereo /* = false */,
    std::function<void(int)> &callback)
{
  /* do rendering here */
  while (1)
  {
    sleep(1);
    callback(0);  /* fetch new data */
    if (data.isNewData())
    {
      fprintf(stderr , "rank= %d: --copying new data --\n", rank);
      /* copy new data into my buffer */
      data.unsetNewData();
    }
    fprintf(stderr ," rank= %d: rendering ... \n", rank);
  }
}

using ShmQHeader = SharedMemoryClient<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryClient<BonsaiSharedQuickData>;
static ShmQHeader *shmQHeader = NULL;
static ShmQData   *shmQData   = NULL;

static bool terminateRenderer = false;

bool fetchSharedData(const bool quickSync, RendererData &rData, const int rank, const int nrank, const MPI_Comm &comm,
    const int reduceDM = 1, const int reduceS = 1)
{
  if (shmQHeader == NULL)
  {
    shmQHeader = new ShmQHeader(ShmQHeader::type::sharedFile(rank));
    shmQData   = new ShmQData  (ShmQData  ::type::sharedFile(rank));
  }

  auto &header = *shmQHeader;
  auto &data   = *shmQData;

  static bool first = true;
  if (quickSync && first) 
  {
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
    first = false;
  }


  static float tLast = -1.0f;


  if (rData.isNewData())
    return false;


#if 0
  //  if (rank == 0)
  fprintf(stderr, " rank= %d: attempting to fetch data \n",rank);
#endif

  // header
  header.acquireLock();
  const float tCurrent = header[0].tCurrent;

  terminateRenderer = tCurrent == -1;

  int sumL = quickSync ? !header[0].done_writing : tCurrent != tLast;
  int sumG ;
  MPI_Allreduce(&sumL, &sumG, 1, MPI_INT, MPI_SUM, comm);


  bool completed = false;
  if (sumG == nrank) //tCurrent != tLast)
  {
    tLast = tCurrent;
    completed = true;

    // data
    const size_t nBodies = header[0].nBodies;
    data.acquireLock();

    const size_t size = data.size();
    assert(size == nBodies);

    /* skip particles that failed to get density, or with too big h */
    auto skipPtcl = [&](const int i)
    {
      return (data[i].rho == 0 || data[i].h == 0.0 || data[i].h > 100);
    };

    size_t nDM = 0, nS = 0;
    constexpr int ntypecount = 10;
    std::array<size_t,ntypecount> ntypeloc, ntypeglb;
    std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
    for (size_t i = 0; i < size; i++)
    {
      const int type = data[i].ID.getType();
      if  (type < ntypecount)
        ntypeloc[type]++;
      if (skipPtcl(i))
        continue;
      switch (type)
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
      for (int type = 0; type < ntypecount; type++)
        if (ntypeglb[type] > 0)
          fprintf(stderr, " ptype= %d:  np= %zu \n",type, ntypeglb[type]);
    }


    rData.resize(nS);
    size_t ip = 0;
    for (size_t i = 0; i < size; i++)
    {
      if (skipPtcl(i))
        continue;
      if (data[i].ID.getType() == 0 )  /* pick stars only */
        continue;

      rData.posx(ip) = data[i].x;
      rData.posy(ip) = data[i].y;
      rData.posz(ip) = data[i].z;
      rData.ID  (ip) = data[i].ID;
      rData.attribute(RendererData::MASS, ip) = data[i].mass;
      rData.attribute(RendererData::VEL,  ip) =
        std::sqrt(
            data[i].vx*data[i].vx+
            data[i].vy*data[i].vy+
            data[i].vz*data[i].vz);
      rData.attribute(RendererData::RHO, ip) = data[i].rho;
      rData.attribute(RendererData::H,   ip) = data[i].h;

      ip++;
      assert(ip <= nS);
    }
    rData.resize(ip);

    data.releaseLock();
  }

  header[0].done_writing = true;
  header.releaseLock();

#if 0
  //  if (rank == 0)
  fprintf(stderr, " rank= %d: done fetching data \n", rank);
#endif

  if (completed)
    rData.computeMinMax();


  return completed;
}


  template<typename T>
static T* readBonsai(
    const int rank, const int nranks, const MPI_Comm &comm,
    const std::string &fileName,
    const int reduceDM,
    const int reduceS,
    const bool print_header = false)
{
  BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::READ, fileName);
  if (rank == 0 && print_header)
  {
    fprintf(stderr, "---- Bonsai header info ----\n");
    out.getHeader().printFields();
    fprintf(stderr, "----------------------------\n");
  }
  typedef float float4[4];
  typedef float float3[3];
  typedef float float2[2];

  BonsaiIO::DataType<IDType> IDListS("Stars:IDType");
  BonsaiIO::DataType<float4> posS("Stars:POS:real4");
  BonsaiIO::DataType<float3> velS("Stars:VEL:float[3]");
  BonsaiIO::DataType<float2> rhohS("Stars:RHOH:float[2]");

  if (reduceS > 0)
  {
    if (!out.read(IDListS, true, reduceS)) return NULL;
    if (rank  == 0)
      fprintf(stderr, " Reading star data \n");
    assert(out.read(posS,    true, reduceS));
    assert(out.read(velS,    true, reduceS));
    bool renderDensity = true;
    if (!out.read(rhohS,  true, reduceS))
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
    if(!out.read(IDListDM, true, reduceDM)) return NULL;
    assert(out.read(posDM,    true, reduceDM));
    assert(out.read(velDM,    true, reduceDM));
    bool renderDensity = true;
    if (!out.read(rhohDM,  true, reduceDM))
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


  T *rDataPtr = new T(rank,nranks,comm);
  rDataPtr->resize(nS+nDM);
  auto &rData = *rDataPtr;
  for (int i = 0; i < nS; i++)
  {
    const int ip = i;
    rData.posx(ip) = posS[i][0];
    rData.posy(ip) = posS[i][1];
    rData.posz(ip) = posS[i][2];
    rData.ID  (ip) = IDListS[i];
    assert(rData.ID(ip).getType() > 0); /* sanity check */
    rData.attribute(RendererData::MASS, ip) = posS[i][3];
    rData.attribute(RendererData::VEL,  ip) =
      std::sqrt(
          velS[i][0]*velS[i][0] +
          velS[i][1]*velS[i][1] +
          velS[i][2]*velS[i][2]);
    if (rhohS.size() > 0)
    {
      rData.attribute(RendererData::RHO, ip) = rhohS[i][0];
      rData.attribute(RendererData::H,  ip)  = rhohS[i][1];
    }
    else
    {
      rData.attribute(RendererData::RHO, ip) = 0.0;
      rData.attribute(RendererData::H,   ip) = 0.0;
    }
  }
  for (int i = 0; i < nDM; i++)
  {
    const int ip = i + nS;
    rData.posx(ip) = posDM[i][0];
    rData.posy(ip) = posDM[i][1];
    rData.posz(ip) = posDM[i][2];
    rData.ID  (ip) = IDListDM[i];
    assert(rData.ID(ip).getType() == 0); /* sanity check */
    rData.attribute(RendererData::MASS, ip) = posDM[i][3];
    rData.attribute(RendererData::VEL,  ip) =
      std::sqrt(
          velDM[i][0]*velDM[i][0] +
          velDM[i][1]*velDM[i][1] +
          velDM[i][2]*velDM[i][2]);
    if (rhohDM.size() > 0)
    {
      rData.attribute(RendererData::RHO, ip) = rhohDM[i][0];
      rData.attribute(RendererData::H,   ip) = rhohDM[i][1];
    }
    else
    {
      rData.attribute(RendererData::RHO, ip) = 0.0;
      rData.attribute(RendererData::H,   ip) = 0.0;
    }
  }

  return rDataPtr;
}


int main(int argc, char * argv[], MPI_Comm commWorld)
{

  std::string fileName;
  int reduceDM    =  10;
  int reduceS=  1;
#ifndef PARTICLESRENDERER
  std::string fullScreenMode    = "";
  bool stereo     = false;
#endif
  int nmaxsample = 200000;
  std::string display;

  bool inSitu = false;
  bool quickSync = true;
  int sleeptime = 1;

  {
    AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

    ADDUSAGE(" ");
    ADDUSAGE("Usage:");
    ADDUSAGE(" ");
    ADDUSAGE(" -h  --help             Prints this help ");
    ADDUSAGE(" -i  --infile #         Input snapshot filename ");
    ADDUSAGE(" -I  --insitu          Enable in-situ rendering ");
    ADDUSAGE("     --sleep  #        start up sleep in sec [1]  ");
    ADDUSAGE("     --noquicksync      disable syncing with simulation [enabled] ");
    ADDUSAGE("     --reduceDM    #    cut down DM dataset by # factor [10]. 0-disable DM");
    ADDUSAGE("     --reduceS     #    cut down stars dataset by # factor [1]. 0-disable S");
#ifndef PARTICLESRENDERER
    ADDUSAGE("     --fullscreen  #    set fullscreen mode string");
    ADDUSAGE("     --stereo           enable stereo rendering");
#endif
    ADDUSAGE(" -s  --nmaxsample   #   set max number of samples for DD [" << nmaxsample << "]");
    ADDUSAGE(" -D  --display      #   set DISPLAY=display, otherwise inherited from environment");


    opt.setFlag  ( "help" ,        'h');
    opt.setOption( "infile",       'i');
    opt.setFlag  ( "insitu",       'I');
    opt.setOption( "reduceDM");
    opt.setOption( "sleep");
    opt.setOption( "reduceS");
    opt.setOption( "fullscreen");
    opt.setFlag("stereo");
    opt.setOption("nmaxsample", 's');
    opt.setOption("display", 'D');
    opt.setFlag  ( "noquicksync");

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    char *optarg = NULL;
    if (opt.getFlag("insitu"))  inSitu = true;
    if ((optarg = opt.getValue("infile")))       fileName           = std::string(optarg);
    if ((optarg = opt.getValue("reduceDM"))) reduceDM       = atoi(optarg);
    if ((optarg = opt.getValue("reduceS"))) reduceS       = atoi(optarg);
#ifndef PARTICLESRENDERER
    if ((optarg = opt.getValue("fullscreen")))	 fullScreenMode     = std::string(optarg);
    if (opt.getFlag("stereo"))  stereo = true;
#endif
    if ((optarg = opt.getValue("nmaxsample"))) nmaxsample = atoi(optarg);
    if ((optarg = opt.getValue("display"))) display = std::string(optarg);
    if ((optarg = opt.getValue("sleep"))) sleeptime = atoi(optarg);
    if (opt.getValue("noquicksync")) quickSync = false;

    if ((fileName.empty() && !inSitu) ||
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
  fprintf(stderr, "bonsai_renderer:: Proc id: %d @ %s , total processes: %d (mpiInit) \n", rank, processor_name, nranks);

  if (rank == 0)
  {
    char hostname[256];
    gethostname(hostname,256);
    char * display = getenv("DISPLAY");
    fprintf(stderr, "root: %s  display: %s \n", hostname, display);
  }

  if (!display.empty())
  {
    std::string var="DISPLAY="+display;
    putenv((char*)var.c_str());
  }

  if (rank == 0)
    fprintf(stderr, " Sleeping for %d seconds \n", sleeptime);
  sleep(sleeptime);



  using RendererDataT = RendererData;
  RendererDataT *rDataPtr;
  if (inSitu)
  {
    rDataPtr = new RendererDataT(rank,nranks,comm);
  }
  else
  {
    if ((rDataPtr = readBonsai<RendererDataT>(rank, nranks, comm, fileName, reduceDM, reduceS))) 
    {}
    else
    {
      if (rank == 0)
        fprintf(stderr, " I don't recognize the format ... please try again , or recompile to use with old tipsy if that is what you use ..\n");
      MPI_Finalize();
      ::exit(-1);
    }
    rDataPtr->computeMinMax();
    rDataPtr->setNewData();
  }

  assert(rDataPtr != 0);


  auto callbackFunc = [&](const int code) 
  {
    int quitL = (code == -1) || terminateRenderer;  /* exit code */
    int quitG;
    MPI_Allreduce(&quitL, &quitG, 1, MPI_INT, MPI_SUM, comm);
    if (quitG)
    {
      MPI_Finalize();
      ::exit(0);
    }

    if (inSitu )
      if (fetchSharedData(quickSync, *rDataPtr, rank, nranks, comm, reduceDM, reduceS))
      {
        rDataPtr->setNewData();
      }
  };

  std::function<void(int)> callback = callbackFunc;
  callback(0);  /* init data set */

  renderer(
      argc, argv, 
      rank, nranks, comm,
      *rDataPtr,
      fullScreenMode.c_str(), 
      stereo,
      callback);

  while(1) {}
  return 0;
}


