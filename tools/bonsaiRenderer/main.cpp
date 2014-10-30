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

#include "renderloop.h"
#include "anyoption.h"
#include "RendererData.h"

#if 0
#define USE_ICET
#endif

#ifdef USE_ICET
  #include <IceT.h>
  #include <IceTGL.h>
  #include <IceTMPI.h>
#endif

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


  int sumL, sumG ;
  if (quickSync)
  {
    while(1)
    {
      header.acquireLock();
      sumL = !header[0].done_writing;
      MPI_Allreduce(&sumL, &sumG, 1, MPI_INT, MPI_SUM, comm);
      if (sumG == nrank)
        break;
      header.releaseLock();
      usleep(1000);
    }
  }
  else
  {
    header.acquireLock();
    const float tCurrent = header[0].tCurrent;
    sumL = tCurrent != tLast;
    MPI_Allreduce(&sumL, &sumG, 1, MPI_INT, MPI_SUM, comm);
  }

  const float tCurrent = header[0].tCurrent;
  terminateRenderer = tCurrent == -1;
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
    rData.setTime(tCurrent);
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
    rData.setNbodySim(ip);

    data.releaseLock();
    header[0].done_writing = true;
  }

  header.releaseLock();

#if 0
  //  if (rank == 0)
  fprintf(stderr, " rank= %d: done fetching data \n", rank);
#endif

  if (completed)
    rData.computeMinMax();


  return completed;
}

void rescaleData(RendererData &rData, 
    const int rank,
    const int nrank,
    const MPI_Comm &comm,
    const bool doDD = false,
    const int  nmaxsample = 200000)
{
  if (doDD)
  {
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();
    rData.randomShuffle();
    rData.setNMAXSAMPLE(nmaxsample);
    fprintf(stderr, " rank= %d: pre n= %d\n", rank, rData.n());
    rData.distribute();
    //    rData.distribute();
    MPI_Barrier(comm);
    const double t1 = MPI_Wtime();
    fprintf(stderr, " rank= %d: post n= %d\n", rank, rData.n());
    if (rank == 0)
      fprintf(stderr, " DD= %g sec \n", t1-t0);
  }
 
  if (rank == 0) 
    fprintf(stderr, "vel: %g %g  rho= %g %g \n ",
        rData.attributeMin(RendererData::VEL),
        rData.attributeMax(RendererData::VEL),
        rData.attributeMin(RendererData::RHO),
        rData.attributeMax(RendererData::RHO));

#if 1
  static auto rhoMin = rData.attributeMin(RendererData::RHO)*10.0;
  static auto rhoMax = rData.attributeMax(RendererData::RHO)/10.0;
  static auto velMin = rData.attributeMin(RendererData::VEL)*2;
  static auto velMax = rData.attributeMax(RendererData::VEL)/2.0;

  rData.clampMinMax(RendererData::RHO, rhoMin, rhoMax);
  rData.clampMinMax(RendererData::VEL, velMin, velMax);
#endif


  rData.rescaleLinear(RendererData::RHO, 0, 60000.0);
  rData.scaleLog(RendererData::RHO);

  rData.rescaleLinear(RendererData::VEL, 0, 3000.0);
}


template<typename T>
static T* readBonsai(
    const int rank, const int nranks, const MPI_Comm &comm,
    const std::string &fileName,
    const int reduceDM,
    const int reduceS,
    const bool print_header = false)
{
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

  if (reduceS > 0)
  {
    if (!in.read(IDListS, true, reduceS)) return NULL;
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
    if(!in.read(IDListDM, true, reduceDM)) return NULL;
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
  rDataPtr->setTime(in.getTime());
  rDataPtr->setNbodySim(nS+nDM);
  in.close();
  auto &rData = *rDataPtr;

  constexpr int ntypecount = 10;
  std::array<size_t,ntypecount> ntypeloc, ntypeglb;
  std::fill(ntypeloc.begin(), ntypeloc.end(), 0);

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
    if (rData.ID(ip).getType() < ntypecount)
      ntypeloc[rData.ID(ip).getType()]++;
  }
  for (int i = 0; i < nDM; i++)
  {
    ntypeloc[0]++;
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
  
  MPI_Reduce(&ntypeloc, &ntypeglb, ntypecount, MPI_LONG_LONG, MPI_SUM, 0, comm);
  if (rank == 0)
  {
    size_t nsum = 0;
    for (int type = 0; type < ntypecount; type++)
    {
      nsum += ntypeglb[type];
      if (ntypeglb[type] > 0)
        fprintf(stderr, "bonsai-read: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
    }
    assert(nsum > 0);
  }

  return rDataPtr;
}

template<typename T>
static T* readJamieSPH(
    const int rank, const int nranks, const MPI_Comm &comm,
    const std::string &fileName,
    const int reduceS,
    const bool print_header = false)
{
  BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::READ, fileName);
  if (rank == 0 && print_header)
  {
    out.getHeader().printFields();
  }
  
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

  assert(reduceS > 0);

  BonsaiIO::DataType<header_t> h("SPH:header:jamieHeader_t");
  BonsaiIO::DataType<sph_t> sph("SPH:data:jamieData_t");

  if (!out.read(h)) 
    return NULL;
  if (rank  == 0)
    fprintf(stderr, " Reading SPH data \n");
  assert(out.read(sph, true, reduceS));

  fprintf(stderr, "rank= %d  ntot= %d\n", rank, (int)sph.size());



  T *rDataPtr = new T(rank,nranks,comm);
  rDataPtr->resize(sph.size());

  auto &rData = *rDataPtr;
  for (int i = 0; i < (int)sph.size(); i++)
  {
    const int ip = i;
    rData.posx(ip) = sph[i].x;
    rData.posy(ip) = sph[i].y;
    rData.posz(ip) = sph[i].z;
    rData.ID  (ip).setID(i);
    rData.ID  (ip).setType(1);
#if 1
    rData.attribute(RendererData::VEL,  ip) =
      std::sqrt(
          sph[i].vx*sph[i].vx +
          sph[i].vy*sph[i].vy +
          sph[i].vz*sph[i].vz);
#else
    rData.attribute(RendererData::VEL,  ip) = sph[i].udot;
#endif
    rData.attribute(RendererData::RHO, ip) = sph[i].rho;
    rData.attribute(RendererData::H,   ip)  = sph[i].hp;
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
  bool doDD = false;
  std::string display;

  bool inSitu = false;
  bool quickSync = true;
  int sleeptime = 1;

  std::string imageFileName;
  std::string cameraFileName;
  int nCameraFrame = 0;


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
		ADDUSAGE(" -d  --doDD             enable domain decomposition  [disabled]");
    ADDUSAGE(" -s  --nmaxsample   #   set max number of samples for DD [" << nmaxsample << "]");
    ADDUSAGE(" -D  --display      #   set DISPLAY=display, otherwise inherited from environment");
    ADDUSAGE("     --camera       #   camera path file");
    ADDUSAGE("     --cameraframe  #   Reframe original camera path to # frames. [ignore]");
    ADDUSAGE("     --image        #   image base filename");


		opt.setFlag  ( "help" ,        'h');
		opt.setOption( "infile",       'i');
		opt.setFlag  ( "insitu",       'I');
		opt.setOption( "reduceDM");
		opt.setOption( "sleep");
		opt.setOption( "reduceS");
    opt.setOption( "fullscreen");
    opt.setOption( "camera");
    opt.setOption( "cameraframe");
    opt.setOption( "image");
    opt.setFlag("stereo");
    opt.setFlag("doDD", 'd');
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
    if (opt.getFlag("doDD"))  doDD = true;
    if ((optarg = opt.getValue("display"))) display = std::string(optarg);
    if ((optarg = opt.getValue("sleep"))) sleeptime = atoi(optarg);
    if (opt.getFlag("noquicksync")) quickSync = false;
    if ((optarg = opt.getValue("image"))) imageFileName = std::string(optarg);
    if ((optarg = opt.getValue("camera"))) cameraFileName = std::string(optarg);
    if ((optarg = opt.getValue("cameraframe"))) nCameraFrame = std::atoi(optarg);

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



  using RendererDataT = RendererDataDistribute;
  RendererDataT *rDataPtr;
  if (inSitu)
  {
    rDataPtr = new RendererDataT(rank,nranks,comm);
  }
  else
  {
    if ((rDataPtr = readBonsai<RendererDataT>(rank, nranks, comm, fileName, reduceDM, reduceS))) {}
    else if ((rDataPtr = readJamieSPH<RendererDataT>(rank, nranks, comm, fileName, reduceS,true))) {}
    else
    {
      if (rank == 0)
        fprintf(stderr, " I don't recognize the format ... please try again , or recompile to use with old tipsy if that is what you use ..\n");
      MPI_Finalize();
      ::exit(-1);
    }
    rDataPtr->computeMinMax();
    rescaleData(*rDataPtr, rank,nranks,comm, doDD,nmaxsample);
    rDataPtr->setNewData();
  }

  assert(rDataPtr != 0);
 

  CameraPath *camera = nullptr;
  if (!cameraFileName.empty())
  {
    camera = new CameraPath(cameraFileName);
    rDataPtr->setCameraPath(camera); 
    if (nCameraFrame > 0)
    {
       if (rank == 0)
         fprintf(stderr, " Reframe camera from %d -> %d \n",
             camera->nFrames(), nCameraFrame);
       camera->reframe(nCameraFrame);

    }
  }


  auto dataSetFunc = [&](const int code) -> void 
  {
    int quitL = (code == -1) || terminateRenderer;  /* exit code */
    int quitG;
    MPI_Allreduce(&quitL, &quitG, 1, MPI_INT, MPI_SUM, comm);
    if (quitG)
    {
      delete camera;
      MPI_Finalize();
      ::exit(0);
    }

    if (inSitu )
      if (fetchSharedData(quickSync, *rDataPtr, rank, nranks, comm, reduceDM, reduceS))
      {
        int nTotal, nLocal = rDataPtr->size();
	MPI_Allreduce(&nLocal, &nTotal, 1, MPI_INT, MPI_SUM, comm);

        if (nTotal > 0)
        {
          rescaleData(*rDataPtr, rank,nranks,comm, doDD,nmaxsample);
          rDataPtr->setNewData();
        }
      }
  };
  std::function<void(int)> updateFunc = dataSetFunc;



  dataSetFunc(0);

#ifdef USE_ICET
  //Setup the IceT context and communicators
  IceTCommunicator icetComm =   icetCreateMPICommunicator(comm);
/*IceTContext   icetContext =*/ icetCreateContext(icetComm);
  icetDestroyMPICommunicator(icetComm); //Save since the comm is copied to the icetContext
  icetDiagnostics(ICET_DIAG_FULL);
#endif

  initAppRenderer(argc, argv, 
      rank, nranks, comm,
      *rDataPtr,
      fullScreenMode.c_str(), 
      stereo,
      updateFunc,
      imageFileName);

  while(1) 
  return 0;
}


