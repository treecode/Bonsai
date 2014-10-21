/*

Bonsai V2: A parallel GPU N-body gravitational Tree-code

(c) 2010-2012:
Jeroen Bedorf
Evghenii Gaburov
Simon Portegies Zwart

Leiden Observatory, Leiden University

http://castle.strw.leidenuniv.nl
http://github.com/treecode/Bonsai

*/

#ifdef WIN32
  #define WIN32_LEAN_AND_MEAN
  #define NOMINMAX
  #include <Windows.h>
  #include <process.h>
  #define M_PI        3.14159265358979323846264338328

  #include <stdlib.h>
  #include <time.h>
  void srand48(const long seed)
  {
    srand(seed);
  }
  //JB This is not a proper work around but just to get things compiled...
  double drand48()
  {
    return double(rand())/RAND_MAX;
  }
#endif


#ifdef USE_MPI
  #include <omp.h>
  #include <mpi.h>
#endif

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include "log.h"
#include "anyoption.h"
#include "renderloop.h"
#include "plummer.h"
#include "disk_shuffle.h"
#ifdef GALACTICS
#include "galactics.h"
#endif
#include "IDType.h"
#include "BonsaiIO.h"
#include <array>


#if ENABLE_LOG
  bool ENABLE_RUNTIME_LOG;
  bool PREPEND_RANK;
  int  PREPEND_RANK_PROCID;
  int  PREPEND_RANK_NPROCS;
#endif

using namespace std;

#include "../profiling/bonsai_timing.h"

int devID;
int renderDevID;

extern void initTimers()
{
#ifndef CUXTIMER_DISABLE
  // Set up the profiling timing info
  build_tree_init();
  compute_propertiesD_init();
  dev_approximate_gravity_init();
  parallel_init();
  sortKernels_init();
  timestep_init();
#endif
}

extern void displayTimers()
{
#ifndef CUXTIMER_DISABLE
  // Display all timing info on the way out
  build_tree_display();
  compute_propertiesD_display();
  //dev_approximate_gravity_display();
  //parallel_display();
  //sortKernels_display();
  //timestep_display();
#endif
}

#include "octree.h"

#ifdef USE_OPENGL
#include "renderloop.h"
#include <cuda_gl_interop.h>
#endif

static double lReadBonsaiFields(
    const int rank, const MPI_Comm &comm,
    const std::vector<BonsaiIO::DataTypeBase*> &data, 
    BonsaiIO::Core &in, 
    const int reduce, 
    const bool restartFlag = true)
{
  double dtRead = 0;
  for (auto &type : data)
  {
    double t0 = MPI_Wtime();
    if (rank == 0)
      fprintf(stderr, " Reading %s ...\n", type->getName().c_str());
    if (in.read(*type, restartFlag, reduce))
    {
      long long int nLoc = type->getNumElements();
      long long int nGlb;
      MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
      if (rank == 0)
      {
        fprintf(stderr, " Read %lld of type %s\n",
            nGlb, type->getName().c_str());
        fprintf(stderr, " ---- \n");
      }
    } 
    else 
    {
      if (rank == 0)
      {
        fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
        fprintf(stderr, " ---- \n");
      }
    }
      
    dtRead += MPI_Wtime() - t0;
  }

  return dtRead;
}

template<typename T>
static inline T& lBonsaiSafeCast(BonsaiIO::DataTypeBase* ptrBase)
{
  T* ptr = dynamic_cast<T*>(ptrBase);
  assert(ptr != NULL);
  return *ptr;
}

static void lReadBonsaiFile(
    std::vector<real4 > &bodyPositions,
    std::vector<real4 > &bodyVelocities,
    std::vector<ullong> &bodyIDs,
    int &NFirst, int &NSecond, int &NThird,
    octree *tree,
    const std::string &fileName,
    const int rank, const int nrank, const MPI_Comm &comm,
    const bool restart = true,
    const int reduceFactor = 1)
{
  if (rank == 0)
    std::cerr << " >>> Reading Bonsai file format : " << fileName <<  std::endl;

  BonsaiIO::Core *in;
  try
  {
    in = new BonsaiIO::Core(rank, nrank, comm, BonsaiIO::READ, fileName);
  }
  catch (const std::exception &e)
  {
    if (rank == 0)
      fprintf(stderr, "Something went wrong: %s \n", e.what());
    MPI_Finalize();
    ::exit(0);
  }

  if (rank == 0)
    in->getHeader().printFields();

  std::vector<BonsaiIO::DataTypeBase*> data;
  typedef float float3[3];
  typedef float float2[2];

  using IDType = BonsaiIO::DataType<IDType>;
  using Pos    = BonsaiIO::DataType<real4>;
  using Vel    = BonsaiIO::DataType<float3>;
  using RhoH   = BonsaiIO::DataType<float2>;
  data.push_back(new IDType("DM:IDType"));
  data.push_back(new Pos   ("DM:POS:real4"));
  data.push_back(new Vel   ("DM:VEL:float[3]"));
  data.push_back(new IDType("Stars:IDType"));
  data.push_back(new Pos   ("Stars:POS:real4"));
  data.push_back(new Vel   ("Stars:VEL:float[3]"));

  const double dtRead = lReadBonsaiFields(rank, comm, data, *in, reduceFactor, restart);

  const auto &DM_IDType = lBonsaiSafeCast<IDType>(data[0]);
  const auto &DM_Pos    = lBonsaiSafeCast<Pos   >(data[1]);
  const auto &DM_Vel    = lBonsaiSafeCast<Vel   >(data[2]);
  const auto &S_IDType  = lBonsaiSafeCast<IDType>(data[3]);
  const auto &S_Pos     = lBonsaiSafeCast<Pos   >(data[4]);
  const auto &S_Vel     = lBonsaiSafeCast<Vel   >(data[5]);

  const size_t nDM = DM_IDType.size();
  assert(nDM == DM_Pos.size());
  assert(nDM == DM_Vel.size());
  
  const size_t nS = S_IDType.size();
  assert(nS == S_Pos.size());
  assert(nS == S_Vel.size());


  NFirst  = static_cast<std::remove_reference<decltype(NFirst )>::type>(nDM);
  NSecond = static_cast<std::remove_reference<decltype(NSecond)>::type>(nS);
  NThird  = 0;

  bodyPositions.resize(nDM+nS);
  bodyVelocities.resize(nDM+nS);
  bodyIDs.resize(nDM+nS);

  /* store DM */

  constexpr int ntypecount = 10;
  std::array<size_t,ntypecount> ntypeloc, ntypeglb;
  std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
  for (int i = 0; i < nDM; i++)
  {
    ntypeloc[0]++;
    auto &pos = bodyPositions[i];
    auto &vel = bodyVelocities[i];
    auto &ID  = bodyIDs[i];
    pos = DM_Pos[i];
    pos.w *= reduceFactor;
    vel = make_float4(DM_Vel[i][0], DM_Vel[i][1], DM_Vel[i][2],0.0f);
    ID  = DM_IDType[i].getID() + DARKMATTERID;
  }
  
  for (int i = 0; i < nS; i++)
  {
    auto &pos = bodyPositions[nDM+i];
    auto &vel = bodyVelocities[nDM+i];
    auto &ID  = bodyIDs[nDM+i];
    pos = S_Pos[i];
    pos.w *= reduceFactor;
    vel = make_float4(S_Vel[i][0], S_Vel[i][1], S_Vel[i][2],0.0f);
    ID  = S_IDType[i].getID();
    switch (S_IDType[i].getType())
    {
      case 1:  /*  Bulge */
        ID += BULGEID;
        break;
      case 2:  /*  Disk */
        ID += DISKID;
        break;
    }
    if (S_IDType[i].getType() < ntypecount)
      ntypeloc[S_IDType[i].getType()]++;
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


  LOGF(stderr,"Read time from snapshot: %f \n", in->getTime());

  if(static_cast<float>(in->getTime()) > 10e10 ||
     static_cast<float>(in->getTime()) < -10e10)
  {
	tree->set_t_current(0);	  
  }
  else
  {
  	tree->set_t_current(static_cast<float>(in->getTime()));
  }

  in->close();
  const double bw = in->computeBandwidth()/1e6;
  for (auto d : data)
    delete d;
  delete in;
  if (rank == 0)
    fprintf(stderr, " :: dtRead= %g  sec readBW= %g MB/s \n", dtRead, bw);
}


void read_tipsy_file_parallel(const MPI_Comm &mpiCommWorld,
    vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<ullong> &bodiesIDs,  float eps2, string fileName,
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<ullong> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor,
                              const bool restart)
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /* 
     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the 
     location where previously the potential was stored.
  */
  
  
  char fullFileName[256];
  if(restart)
    sprintf(fullFileName, "%s%d", fileName.c_str(), rank);
  else
    sprintf(fullFileName, "%s", fileName.c_str());

  LOG("Trying to read file: %s \n", fullFileName);
  
  
  
  ifstream inputFile(fullFileName, ios::in | ios::binary);
  if(!inputFile.is_open())
  {
    LOG("Can't open input file \n");
    ::exit(0);
  }
  
  dumpV2  h;
  inputFile.read((char*)&h, sizeof(h));  

  int NTotal;
  ullong idummy;
  real4 positions;
  real4 velocity;

     
  //Read Tipsy header
  NTotal        = h.nbodies;
  NFirst        = h.ndark;
  NSecond       = h.nstar;
  NThird        = h.nsph;

  printf("File version: %d \n", h.version);

  int fileFormatVersion = 0;

  if(h.version == 2) fileFormatVersion = 2;



  tree->set_t_current((float) h.time);
  
  //Rough divide
  uint perProc = (NTotal / procs) /reduce_bodies_factor;
  if(restart) perProc = NTotal /reduce_bodies_factor; //don't subdivide when using restart
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;
  
  dark_particleV2 d;
  star_particleV2 s;

  int globalParticleCount = 0;
  int bodyCount = 0;
  int dustCount = 0;
  
  constexpr int ntypecount = 10;
  std::array<size_t,ntypecount> ntypeloc, ntypeglb;
  std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
  for(int i=0; i < NTotal; i++)
  {
    if(i < NFirst)
    {
      inputFile.read((char*)&d, sizeof(d));
      //velocity.w        = d.eps;
      velocity.w        = 0;
      positions.w       = d.mass;
      positions.x       = d.pos[0];
      positions.y       = d.pos[1];
      positions.z       = d.pos[2];
      velocity.x        = d.vel[0];
      velocity.y        = d.vel[1];
      velocity.z        = d.vel[2];
      idummy            = d.getID();

      //printf("%d\t%f\t%f\t%f\n", i, positions.x, positions.y, positions.z);

      //Force compatibility with older 32bit ID files by mapping the particle IDs
      if(fileFormatVersion == 0)
      {
        idummy    = s.getID_V1() + DARKMATTERID;
      }
      //end mapping
    }
    else
    {
      inputFile.read((char*)&s, sizeof(s));
      //velocity.w        = s.eps;
      velocity.w        = 0;
      positions.w       = s.mass;
      positions.x       = s.pos[0];
      positions.y       = s.pos[1];
      positions.z       = s.pos[2];
      velocity.x        = s.vel[0];
      velocity.y        = s.vel[1];
      velocity.z        = s.vel[2];
      idummy            = s.getID();

      //Force compatibility with older 32bit ID files by mapping the particle IDs
      if(fileFormatVersion == 0)
      {
        if(s.getID_V1() >= 100000000)
          idummy    = s.getID_V1() + BULGEID; //Bulge particles
        else
          idummy    = s.getID_V1();
      }
      //end mapping
    }


    if(positions.z < -10e10)
    {
       fprintf(stderr," Removing particle %d because of Z is: %f \n", globalParticleCount, positions.z);
       continue;
    }

    const auto id = idummy;
    if(id >= DISKID  && id < BULGEID)       
    {
      ntypeloc[2]++;
    }
    else if(id >= BULGEID && id < DARKMATTERID)  
    {
      ntypeloc[1]++;
    }
    else if (id >= DARKMATTERID)
    {
      ntypeloc[0]++;
    }

    globalParticleCount++;
   
    #ifdef USE_DUST
      if(idummy >= 50000000 && idummy < 100000000)
      {
        dustCount++;
        if( dustCount % reduce_dust_factor == 0 ) 
          positions.w *= reduce_dust_factor;

        if( dustCount % reduce_dust_factor != 0 )
          continue;
        dustPositions.push_back(positions);
        dustVelocities.push_back(velocity);
        dustIDs.push_back(idummy);      
      }
      else
      {
        bodyCount++;
        if( bodyCount % reduce_bodies_factor == 0 ) 
		      positions.w *= reduce_bodies_factor;

	      if( bodyCount % reduce_bodies_factor != 0 )
		      continue;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);  
      }

    
    #else
      if( globalParticleCount % reduce_bodies_factor == 0 ) 
        positions.w *= reduce_bodies_factor;

      if( globalParticleCount % reduce_bodies_factor != 0 )
        continue;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(idummy);  

    #endif

    particleCount++;


  
  
    if(!restart)
    {
      if(bodyPositions.size() > perProc && procCntr != procs)
      {
        tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
        procCntr++;

        bodyPositions.clear();
        bodyVelocities.clear();
        bodiesIDs.clear();
      }
    }
  }//end while
  
  inputFile.close();
  
  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);  
//   NTotal2 = particleCount-1;
  NTotal2 = particleCount;
  LOGF(stderr,"NTotal: %d\tper proc: %d\tFor ourself: %d \tNDust: %d \n",
               NTotal, perProc, (int)bodiesIDs.size(), (int)dustPositions.size());

  /* this check was added to test whether particle type was failed to identified.
   * sometimed DM particles are treated as nonDM on 
   * MW+M31 4.6M particle snapshot 
   */
  if (restart)
  {
    MPI_Reduce(&ntypeloc, &ntypeglb, 10, MPI_LONG_LONG, MPI_SUM, 0, mpiCommWorld);
  }
  else
  {
    std::copy(ntypeloc.begin(), ntypeloc.end(), ntypeglb.begin());
  }

  if (rank == 0)
  {
    for (int type = 0; type < ntypecount; type++)
      if (ntypeglb[type] > 0)
        fprintf(stderr, "tispy-read: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
  }
}



#ifdef GALACTICS
  void generateGalacticsModel(const int      procId,
                              const int      nProcs,
                              const int      nMilkyWay,
                              const int      nMWfork,
                              const bool     scaleMass,
                              vector<real4>  &bodyPositions,
                              vector<real4>  &bodyVelocities,
                              vector<ullong> &bodyIDs)
  {
    if (procId == 0) printf("Using MilkyWay model with n= %d per proc, forked %d times \n", nMilkyWay, nMWfork);
    assert(nMilkyWay > 0);
    assert(nMWfork > 0);

    //Verify that all required files are available
    const char* fileList[] = {"cordbh.dat", "dbh.dat", "freqdbh.dat", "mr.dat",
                              "denspsibulge.dat", "denspsihalo.dat", "component_numbers.txt"};
    const int nFiles       = sizeof(fileList) / sizeof(fileList[0]);

    for(int i=0; i < nFiles; i++)
    {
      ifstream ifile(fileList[i]);
      if (!ifile) {
        fprintf(stderr,"Can not find the required input file: %s \n", fileList[i]);
        ::exit(-1);
      }
    }

    //Read in the particle ratios
    int nHalo, nBulge,nDisk;
    ifstream ifile("component_numbers.txt");
    std::string line;
    std::getline(ifile, line);
    sscanf(line.c_str(),"%d %d %d\n", &nHalo, &nBulge, &nDisk);

    fprintf(stderr,"Particle numbers from config file: %d %d %d \n", nHalo, nBulge, nDisk);
    ifile.close();

//    #if 1 /* in this setup all particles will be of equal mass (exact number are galactic-depednant)  */
//      const float fdisk  = 15.1;
//      const float fbulge = 5.1;
//      const float fhalo  = 242.31;
//    #else  /* here, bulge & mw particles have the same mass, but halo particles is 32x heavier */
//      const float fdisk  = 15.1;
//      const float fbulge = 5.1;
//      const float fhalo  = 7.5;
//    #endif
//    const float fsum = fdisk + fhalo + fbulge;

    const float fsum = (float)(nHalo + nBulge + nDisk);
    const int ndisk  = (int)  (nMilkyWay * nDisk /fsum);
    const int nbulge = (int)  (nMilkyWay * nBulge/fsum);
    const int nhalo  = (int)  (nMilkyWay * nHalo /fsum);

    assert(ndisk  > 0);
    assert(nbulge > 0);
    assert(nhalo  > 0);

    if (procId == 0)
      fprintf(stderr,"Requested numbers: ndisk= %d  nbulge= %d  nhalo= %d :: ntotal= %d\n",
                      ndisk, nbulge, nhalo, ndisk+nbulge+nhalo);

    const Galactics g(procId, nProcs, ndisk, nbulge, nhalo, nMWfork);
    if (procId == 0)
     printf("Generated numbers:  ndisk= %d  nbulge= %d  nhalo= %d :: ntotal= %d\n",
             g.get_ndisk(), g.get_nbulge(), g.get_nhalo(), g.get_ntot());

    const int ntot = g.get_ntot();
    bodyPositions.resize(ntot);
    bodyVelocities.resize(ntot);
    bodyIDs.resize(ntot);

    //Generate unique 64bit IDs, counter starts at individual boundaries
    //Note that we get 32bit IDs back from the Galactics code
    unsigned long long diskID  = ((unsigned long long) ndisk *procId) + DISKID;
    unsigned long long bulgeID = ((unsigned long long) nbulge*procId) + BULGEID;
    unsigned long long haloID  = ((unsigned long long) nhalo *procId) + DARKMATTERID;

    for (int i= 0; i < ntot; i++)
    {
      assert(!std::isnan(g[i].x));
      assert(!std::isnan(g[i].y));
      assert(!std::isnan(g[i].z));
      assert(g[i].mass > 0.0);

      //Generate unique IDS for each particle in the full model
      if( g[i].id >= 200000000)                             //Dark matter
        bodyIDs[i] = haloID++;
      else if( g[i].id >= 100000000 && g[i].id < 200000000) //Bulge
        bodyIDs[i] = bulgeID++;
      else                                                  //Disk
        bodyIDs[i] = diskID++;

      bodyPositions[i].x = g[i].x;
      bodyPositions[i].y = g[i].y;
      bodyPositions[i].z = g[i].z;
      if(scaleMass)
       bodyPositions[i].w = g[i].mass * 1.0/(double)nProcs;
      else
       bodyPositions[i].w = g[i].mass; // * 1.0/(double)nProcs ,scaled later ..

      assert(!std::isnan(g[i].vx));
      assert(!std::isnan(g[i].vy));
      assert(!std::isnan(g[i].vz));

      bodyVelocities[i].x = g[i].vx;
      bodyVelocities[i].y = g[i].vy;
      bodyVelocities[i].z = g[i].vz;
      bodyVelocities[i].w = 0.0;
    }
  } //generateGalacticsModel
#endif


double get_time_main()
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}


//Buffers and flags used for the IO thread


volatile IOSharedData_t ioSharedData;

// std::vector<real4>   ioThreadPos;
// std::vector<real4>   ioThreadVel;
// std::vector<ullong>  ioThreadIDs;
//volatile sharedIOThreadStruct ioThreadProps;



long long my_dev::base_mem::currentMemUsage;
long long my_dev::base_mem::maxMemUsage;

int main(int argc, char** argv, MPI_Comm comm)
{
  my_dev::base_mem::currentMemUsage = 0;
  my_dev::base_mem::maxMemUsage     = 0;

  vector<real4>   bodyPositions;
  vector<real4>   bodyVelocities;
  vector<ullong>  bodyIDs;

  vector<real4>   dustPositions;
  vector<real4>   dustVelocities;
  vector<ullong>  dustIDs;
 
  float eps      = 0.05f;
  float theta    = 0.75f;
  float timeStep = 1.0f / 16.0f;
  float tEnd     = 1;
  int   iterEnd  = (1 << 30);
  devID          = 0;
  renderDevID    = 0;

  string fileName          =  "";
  string logFileName       = "gpuLog.log";
  string snapshotFile      = "snapshot_";
  std::string bonsaiFileName;
  float snapshotIter       = -1;
  float  remoDistance      = -1.0;
  int rebuild_tree_rate    = 1;
  int reduce_bodies_factor = 1;
  int reduce_dust_factor   = 1;
  string fullScreenMode    = "";
  bool direct     = false;
  bool fullscreen = false;
  bool displayFPS = false;
  bool diskmode   = false;
  bool stereo     = false;
  bool restartSim = false;

  float quickDump  = 0.0;
  float quickRatio = 0.1;
  bool  quickSync  = true;
  bool  useMPIIO = false;

#if ENABLE_LOG
  ENABLE_RUNTIME_LOG = false;
  PREPEND_RANK       = false;
#endif

#ifdef USE_OPENGL
	TstartGlow = 0.0;
	dTstartGlow = 1.0;
#endif
        
  double tStartupStart = get_time_main();       
  double tStartModel   = 0;
  double tEndModel     = 0;
  
  

  int nPlummer  = -1;
  int nSphere   = -1;
  int nCube     = -1;
  int nMilkyWay = -1;
  int nMWfork   =  4;
  std::string taskVar;
//#define TITAN_G
//#define SLURM_G
#ifdef TITAN_G
  //Works for both Titan and Piz Daint
  taskVar = std::string("PMI_FORK_RANK");
#elif defined SLURM_G
  taskVar = std::string("SLURM_PROCID");
#endif
	/************** beg - command line arguments ********/
#if 1
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename in Tipsy format");
		ADDUSAGE(" -f  --bonsaifile #     Input snapshot filename in Bonsai format [muse be used with --usempiio]");
		ADDUSAGE("     --restart          Let each process restart from a snapshot as specified by 'infile'");
		ADDUSAGE("     --logfile #        Log filename [" << logFileName << "]");
		ADDUSAGE("     --dev #            Device ID [" << devID << "]");
		ADDUSAGE("     --renderdev #      Rendering Device ID [" << renderDevID << "]");
		ADDUSAGE(" -t  --dt #             time step [" << timeStep << "]");
		ADDUSAGE(" -T  --tend #           N-body end time [" << tEnd << "]");
		ADDUSAGE(" -I  --iend #           N-body end iteration [" << iterEnd << "]");
		ADDUSAGE(" -e  --eps #            softening (will be squared) [" << eps << "]");
		ADDUSAGE(" -o  --theta #          opening angle (theta) [" <<theta << "]");
		ADDUSAGE("     --snapname #       snapshot base name (N-body time is appended in 000000 format) [" << snapshotFile << "]");
		ADDUSAGE("     --snapiter #       snapshot iteration (N-body time) [" << snapshotIter << "]");
		ADDUSAGE("     --quickdump  #     how ofter to dump quick output (N-body time) [" << quickDump << "]");
		ADDUSAGE("     --quickratio #     which fraction of data to dump (fraction) [" << quickRatio << "]");
    ADDUSAGE("     --noquicksync      disable syncing for quick dumping ");
    ADDUSAGE("     --usempiio         use MPI-IO [disabled]");
		ADDUSAGE("     --rmdist #         Particle removal distance (-1 to disable) [" << remoDistance << "]");
		ADDUSAGE(" -r  --rebuild #        rebuild tree every # steps [" << rebuild_tree_rate << "]");
		ADDUSAGE("     --reducebodies #   cut down bodies dataset by # factor ");
#ifdef USE_DUST
    ADDUSAGE("     --reducedust #     cut down dust dataset by # factor ");
#endif
#if ENABLE_LOG
    ADDUSAGE("     --log              enable logging ");
    ADDUSAGE("     --prepend-rank     prepend the MPI rank in front of the log-lines ");
#endif
    ADDUSAGE("     --direct           enable N^2 direct gravitation [" << (direct ? "on" : "off") << "]");
#ifdef USE_OPENGL
		ADDUSAGE("     --fullscreen #     set fullscreen mode string");
    ADDUSAGE("     --displayfps       enable on-screen FPS display");
		ADDUSAGE("     --Tglow  #         enable glow @ # Myr [" << TstartGlow << "]");
		ADDUSAGE("     --dTglow  #        reach full brightness in @ # Myr [" << dTstartGlow << "]");
		ADDUSAGE("     --stereo           enable stereo rendering");
#endif
#ifdef GALACTICS
		ADDUSAGE("     --milkyway #       use Milky Way model with # particles per proc");
		ADDUSAGE("     --mwfork   #       fork Milky Way generator into # processes [" << nMWfork << "]");
    ADDUSAGE("     --taskvar  #       variable name to obtain task id [for randoms seed] before MPI_Init. \n");
#endif
    ADDUSAGE("     --plummer  #       use Plummer model with # particles per proc");
		ADDUSAGE("     --sphere   #       use spherical model with # particles per proc");
		ADDUSAGE("     --cube     #       use cube model with # particles per proc");
    ADDUSAGE("     --diskmode         use diskmode to read same input file all MPI taks and randomly shuffle its positions");
		ADDUSAGE(" ");


		opt.setFlag( "help" ,   'h');
		opt.setFlag( "diskmode");
		opt.setOption( "infile",  'i');
		opt.setOption( "bonsaifile",  'f');
		opt.setFlag  ( "restart");
		opt.setOption( "dt",      't' );
		opt.setOption( "tend",    'T' );
		opt.setOption( "iend",    'I' );
		opt.setOption( "eps",     'e' );
		opt.setOption( "theta",   'o' );
		opt.setOption( "rebuild", 'r' );
    opt.setOption( "plummer");
#ifdef GALACTICS
    opt.setOption( "milkyway");
    opt.setOption( "mwfork");
    opt.setOption( "taskvar");
#endif
    opt.setOption( "sphere");
    opt.setOption( "cube");
    opt.setOption( "dev" );
    opt.setOption( "renderdev" );
    opt.setOption( "logfile" );
    opt.setOption( "snapname");
    opt.setOption( "snapiter");
    opt.setOption( "quickdump");
    opt.setOption( "quickratio");
    opt.setFlag  ( "usempiio");
    opt.setFlag  ( "noquicksync");
    opt.setOption( "rmdist");
    opt.setOption( "valueadd");
    opt.setOption( "reducebodies");
#ifdef USE_DUST
    opt.setOption( "reducedust");
#endif /* USE_DUST */
#if ENABLE_LOG
    opt.setFlag("log");
    opt.setFlag("prepend-rank");
#endif
    opt.setFlag("direct");
#ifdef USE_OPENGL
    opt.setOption( "fullscreen");
    opt.setOption( "Tglow");
    opt.setOption( "dTglow");
    opt.setFlag("displayfps");
    opt.setFlag("stereo");
#endif

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    if (opt.getFlag("direct")) direct = true;
    if (opt.getFlag("restart")) restartSim = true;
    if (opt.getFlag("displayfps")) displayFPS = true;
    if (opt.getFlag("diskmode")) diskmode = true;
    if(opt.getFlag("stereo"))   stereo = true;

#if ENABLE_LOG
    if (opt.getFlag("log"))           ENABLE_RUNTIME_LOG = true;
    if (opt.getFlag("prepend-rank"))  PREPEND_RANK       = true;
#endif    
    char *optarg = NULL;
    if ((optarg = opt.getValue("infile")))       fileName           = string(optarg);
    if ((optarg = opt.getValue("bonsaifile")))   bonsaiFileName     = std::string(optarg);
    if ((optarg = opt.getValue("plummer")))      nPlummer           = atoi(optarg);
    if ((optarg = opt.getValue("milkyway")))     nMilkyWay          = atoi(optarg);
    if ((optarg = opt.getValue("mwfork")))       nMWfork            = atoi(optarg);
    if ((optarg = opt.getValue("taskvar")))      taskVar            = std::string(optarg);
    if ((optarg = opt.getValue("sphere")))       nSphere            = atoi(optarg);
    if ((optarg = opt.getValue("cube")))         nCube              = atoi(optarg);
    if ((optarg = opt.getValue("logfile")))      logFileName        = string(optarg);
    if ((optarg = opt.getValue("dev")))          devID              = atoi  (optarg);
    renderDevID = devID;
    if ((optarg = opt.getValue("renderdev")))    renderDevID        = atoi  (optarg);
    if ((optarg = opt.getValue("dt")))           timeStep           = (float) atof  (optarg);
    if ((optarg = opt.getValue("tend")))         tEnd               = (float) atof  (optarg);
    if ((optarg = opt.getValue("iend")))         iterEnd            = atoi  (optarg);
    if ((optarg = opt.getValue("eps")))          eps                = (float) atof  (optarg);
    if ((optarg = opt.getValue("theta")))        theta              = (float) atof  (optarg);
    if ((optarg = opt.getValue("snapname")))     snapshotFile       = string(optarg);
    if ((optarg = opt.getValue("snapiter")))     snapshotIter       = (float) atof  (optarg);
    if ((optarg = opt.getValue("quickdump")))    quickDump          = (float) atof  (optarg);
    if ((optarg = opt.getValue("quickratio")))   quickRatio         = (float) atof  (optarg);
    if (opt.getValue("usempiio")) useMPIIO = true;
    if (opt.getValue("noquicksync")) quickSync = false;
    if ((optarg = opt.getValue("rmdist")))       remoDistance       = (float) atof  (optarg);
    if ((optarg = opt.getValue("rebuild")))      rebuild_tree_rate  = atoi  (optarg);
    if ((optarg = opt.getValue("reducebodies"))) reduce_bodies_factor = atoi  (optarg);
    if ((optarg = opt.getValue("reducedust")))	 reduce_dust_factor = atoi  (optarg);
#if USE_OPENGL
    if ((optarg = opt.getValue("fullscreen")))	 fullScreenMode     = string(optarg);
    if ((optarg = opt.getValue("Tglow")))	 TstartGlow  = (float)atof(optarg);
    if ((optarg = opt.getValue("dTglow")))	 dTstartGlow  = (float)atof(optarg);
    dTstartGlow = std::max(dTstartGlow, 1.0f);
#endif
    if (bonsaiFileName.empty() && fileName.empty() && nPlummer == -1 && nSphere == -1 && nMilkyWay == -1 && nCube == -1)
    {
      opt.printUsage();
      ::exit(0);
    }
    if (!bonsaiFileName.empty() && !useMPIIO)
    {
      opt.printUsage();
      ::exit(0);
    }

#undef ADDUSAGE
  }
#endif



  /********** init galaxy before MPI initialization to prevent problems with forking **********/
  const char * argVal = getenv(taskVar.c_str());
  if (argVal == NULL)
  {
    fprintf(stderr, " Unknown ENV_VARIABLE: %s  -- Falling to basic forking method after MPI_Init, unsafe!\n", taskVar.c_str());
    taskVar = std::string();
  }
  if (nMilkyWay >= 0 && !taskVar.empty())
  {
    assert(argVal != NULL);
    const int procId = atoi(argVal);
    //    fprintf(stderr, " taskVar= %s , value= %d\n", taskVar.c_str(), procId);
    #ifdef GALACTICS
        tStartModel = get_time_main();
        //Use 32768*7 for nProcs to create independent seeds for all processes we use
        //do not scale untill we know the number of processors
        generateGalacticsModel(procId, 32768*7, nMilkyWay, nMWfork,
                               false, bodyPositions, bodyVelocities,
                               bodyIDs);
        tEndModel   = get_time_main();
    #else
        assert(0);
    #endif
  }

  /*********************************/

  /************** end - command line arguments ********/

  /* Overrule settings for the device */
  //  const char * tempRankStr = getenv("OMPI_COMM_WORLD_RANK");
  //  devID = renderDevID = atoi(tempRankStr);
  //  fprintf(stderr,"Overruled ids: %d ", devID);
  /* End overrule */


  int NTotal, NFirst, NSecond, NThird;
  NTotal = NFirst = NSecond = NThird = 0;

#ifdef USE_OPENGL
  // create OpenGL context first, and register for interop
  initGL(argc, argv, fullScreenMode.c_str(), stereo);
  cudaGLSetGLDevice(devID);
#endif

  initTimers();

  int pid = -1;
#ifdef WIN32
  pid = _getpid();
#else
  pid = (int)getpid();
#endif
  //Used for profiler, note this has to be done before initing to
  //octree otherwise it has no effect...Therefore use pid instead of mpi procId
  char *gpu_prof_log;
  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
  if(gpu_prof_log){
    char tmp[50];
    sprintf(tmp,"process_%d_%s",pid,gpu_prof_log);
#ifdef WIN32
    //        SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
#else
    //        setenv("CUDA_PROFILE_LOG",tmp,1);
    LOGF(stderr, "TESTING log on proc: %d val: %s \n", pid, tmp);
#endif
  }

  int mpiInitialized = 0;
  MPI_Initialized(&mpiInitialized);
  MPI_Comm mpiCommWorld = MPI_COMM_WORLD;
  if (!mpiInitialized)
    MPI_Init(&argc, &argv);
  else
    mpiCommWorld = comm;

  //Creat the octree class and set the properties
  octree *tree = new octree(
      mpiCommWorld,
      argv, devID, theta, eps, 
      snapshotFile, snapshotIter,  
      quickDump, quickRatio, quickSync,
      useMPIIO,
      timeStep,
      tEnd, iterEnd, (int)remoDistance, rebuild_tree_rate, direct);

  double tStartup = tree->get_time();

  //Get parallel processing information  
  int procId = tree->mpiGetRank();
  int nProcs = tree->mpiGetNProcs();

  if (procId == 0)
  {
    //Note can't use LOGF here since MPI isn't initialized yet
    cerr << "[INIT]\tUsed settings: \n";
    cerr << "[INIT]\tInput  filename " << fileName << endl;
    cerr << "[INIT]\tBonsai filename " << bonsaiFileName << endl;
    cerr << "[INIT]\tLog filename " << logFileName << endl;
    cerr << "[INIT]\tTheta: \t\t"             << theta        << "\t\teps: \t\t"          << eps << endl;
    cerr << "[INIT]\tTimestep: \t"          << timeStep     << "\t\ttEnd: \t\t"         << tEnd << endl;
    cerr << "[INIT]\titerEnd: \t" << iterEnd << endl;
    cerr << "[INIT]\tUse MPI-IO: \t" << (useMPIIO ? "YES" : "NO") << endl;
    cerr << "[INIT]\tsnapshotFile: \t"      << snapshotFile << "\tsnapshotIter: \t" << snapshotIter << endl;
    if (useMPIIO)
    {
      cerr << "[INIT]\t  quickDump: \t"      << quickDump << "\t\tquickRatio: \t" << quickRatio << endl;
    }
    cerr << "[INIT]\tInput file: \t"        << fileName     << "\t\tdevID: \t\t"        << devID << endl;
    cerr << "[INIT]\tRemove dist: \t"   << remoDistance << endl;
    cerr << "[INIT]\tRebuild tree every " << rebuild_tree_rate << " timestep\n";


    if( reduce_bodies_factor > 1 )
      cerr << "[INIT]\tReduce number of non-dust bodies by " << reduce_bodies_factor << " \n";
    if( reduce_dust_factor > 1 )
      cerr << "[INIT]\tReduce number of dust bodies by " << reduce_dust_factor << " \n";

#if ENABLE_LOG
    if (ENABLE_RUNTIME_LOG)
      cerr << "[INIT]\tRuntime logging is ENABLED \n";
    else
      cerr << "[INIT]\tRuntime logging is DISABLED \n";
#endif
    cerr << "[INIT]\tDirect gravitation is " << (direct ? "ENABLED" : "DISABLED") << endl;
#if USE_OPENGL
    cerr << "[INIT]\tTglow = " << TstartGlow << endl;
    cerr << "[INIT]\tdTglow = " << dTstartGlow << endl;
    cerr << "[INIT]\tstereo = " << stereo << endl;
#endif
#ifdef USE_MPI                
    cerr << "[INIT]\tCode is built WITH MPI Support \n";
#else
    cerr << "[INIT]\tCode is built WITHOUT MPI Support \n";
#endif
  }
  assert(quickRatio > 0 && quickRatio <= 1);

#ifdef USE_MPI

  //Used on Titan and Piz Daint
  #if 1
    omp_set_num_threads(16);
  #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );

      int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

      int i, set=-1;
      for (i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, &cpuset))
          set = i;
      //    fprintf(stderr,"[Proc: %d ] Thread %d bound to: %d Total cores: %d\n",
      //        procId, tid,  set, num_cores);
    }
  #endif


  #if 0
    omp_set_num_threads(4);
    //default
    // int cpulist[] = {0,1,2,3,8,9,10,11};
    int cpulist[] = {0,1,2,3, 8,9,10,11, 4,5,6,7, 12,13,14,15}; //HA-PACS
    //int cpulist[] = {0,1,2,3,4,5,6,7};
    //int cpulist[] = {0,2,4,6, 8,10,12,14};
    //int cpulist[] = {1,3,5,7, 9,11,13,15};
    //int cpulist[] = {1,9,5,11, 3,7,13,15};
    //int cpulist[] = {1,15,3,13, 2,4,6,8};
    //int cpulist[] = {1,1,1,1, 1,1,1,1};


  #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      //int core_id = procId*4+tid;
      int core_id = (procId%4)*4+tid;
      core_id     = cpulist[core_id];

      int num_cores = sysconf(_SC_NPROCESSORS_ONLN);

      cpu_set_t cpuset;
      CPU_ZERO(&cpuset);
      CPU_SET(core_id, &cpuset);
      pthread_t current_thread = pthread_self();
      int return_val = pthread_setaffinity_np(current_thread, sizeof(cpu_set_t), &cpuset);

      CPU_ZERO(&cpuset);
      pthread_getaffinity_np(pthread_self()  , sizeof( cpu_set_t ), &cpuset );

      int i, set=-1;
      for (i = 0; i < CPU_SETSIZE; i++)
        if (CPU_ISSET(i, &cpuset))
          set = i;
      //printf("CPU2: CPU %d\n", i);

      fprintf(stderr,"Binding thread: %d of rank: %d to cpu: %d CHECK: %d Total cores: %d\n",
          tid, procId, core_id, set, num_cores);
    }
  #endif
#endif


  #if ENABLE_LOG
    #ifdef USE_MPI
      PREPEND_RANK_PROCID = procId;
      PREPEND_RANK_NPROCS = nProcs;
    #endif
  #endif


  if(nProcs > 1)
  {
    logFileName.append("-");

    char buff[16];
    sprintf(buff,"%d-%d", nProcs, procId);
    logFileName.append(buff);
  }

  //ofstream logFile(logFileName.c_str());
  //Use a string stream buffer, only write at end of the run
  std::stringstream logStream;
  ostream &logFile = logStream;

  tree->set_context(logFile, false); //Do logging to file and enable timing (false = enabled)
  
  char logPretext[64];
  sprintf(logPretext, "PROC-%05d ", procId);
  tree->set_logPreamble(logPretext);

  double tStartup2 = tree->get_time();  

  if (!bonsaiFileName.empty() && useMPIIO)
  {
    const MPI_Comm &comm = mpiCommWorld;
    lReadBonsaiFile(
        bodyPositions, 
        bodyVelocities,
        bodyIDs,
        NFirst,NSecond,NThird,
        tree,
        bonsaiFileName,
        procId, nProcs, comm,
        restartSim,
        reduce_bodies_factor);
  }
  else if(restartSim)
  {
    //The input snapshot file are many files with each process reading its own particles
    read_tipsy_file_parallel(mpiCommWorld, bodyPositions, bodyVelocities, bodyIDs, eps, fileName,
                             procId, nProcs, NTotal, NFirst, NSecond, NThird, tree,
                             dustPositions, dustVelocities, dustIDs,
                             reduce_bodies_factor, reduce_dust_factor, true);
  }
  else if (nPlummer == -1 && nSphere == -1  && nCube == -1 && !diskmode && nMilkyWay == -1)
  {
    if(procId == 0)
    {
      #ifdef TIPSYOUTPUT
            read_tipsy_file_parallel(mpiCommWorld, bodyPositions, bodyVelocities, bodyIDs, eps, fileName,
                procId, nProcs, NTotal, NFirst, NSecond, NThird, tree,
                dustPositions, dustVelocities, dustIDs, reduce_bodies_factor, reduce_dust_factor, false);
      #else
            assert(0); //This file format is removed
            //read_dumbp_file_parallel(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, procId, nProcs, NTotal, NFirst, NSecond, NThird, tree, reduce_bodies_factor);
      #endif
    }
    else
    {
      tree->ICRecv(0, bodyPositions, bodyVelocities,  bodyIDs);
    }
    #if USE_MPI
        float tCurrent = tree->get_t_current();
        MPI_Bcast(&tCurrent, 1, MPI_FLOAT, 0,mpiCommWorld);
        tree->set_t_current(tCurrent);
    #endif
  }
  else if(nMilkyWay >= 0)
  {
    #ifdef GALACTICS
        if (taskVar.empty())
        {
          tStartModel   = get_time_main();
          generateGalacticsModel(procId, nProcs, nMilkyWay, nMWfork,
                                 true, bodyPositions, bodyVelocities,
                                 bodyIDs);
          tEndModel   = get_time_main();
        }
        else
        {
          //Scale mass of previously generated model
          const int ntot = bodyPositions.size();
          for (int i= 0; i < ntot; i++)
            bodyPositions[i].w *= 1.0/(double)nProcs;
        }
    #else
          assert(0);
    #endif
  }
  else if(nPlummer >= 0)
  {
    if (procId == 0) printf("Using Plummer model with n= %d per process \n", nPlummer);
    assert(nPlummer > 0);
    const int seed = 19810614 + procId;
    const Plummer m(nPlummer, procId, seed);
    bodyPositions.resize(nPlummer);
    bodyVelocities.resize(nPlummer);
    bodyIDs.resize(nPlummer);
    for (int i= 0; i < nPlummer; i++)
    {
      assert(!std::isnan(m.pos[i].x));
      assert(!std::isnan(m.pos[i].y));
      assert(!std::isnan(m.pos[i].z));
      assert(m.mass[i] > 0.0);
      bodyIDs[i]   = ((unsigned long long) nPlummer)*procId + i;

      bodyPositions[i].x = m.pos[i].x;
      bodyPositions[i].y = m.pos[i].y;
      bodyPositions[i].z = m.pos[i].z;
      bodyPositions[i].w = m.mass[i] * 1.0/nProcs;

      bodyVelocities[i].x = m.vel[i].x;
      bodyVelocities[i].y = m.vel[i].y;
      bodyVelocities[i].z = m.vel[i].z;
      bodyVelocities[i].w = 0;
    }
  }
  else if (nSphere >= 0)
  {
    //Sphere
    if (procId == 0) printf("Using Spherical model with n= %d per process \n", nSphere);
    assert(nSphere >= 0);
    bodyPositions.resize(nSphere);
    bodyVelocities.resize(nSphere);
    bodyIDs.resize(nSphere);

    srand48(procId+19840501);

    /* generate uniform sphere */
    int np = 0;
    while (np < nSphere)
    {
      const double x = 2.0*drand48()-1.0;
      const double y = 2.0*drand48()-1.0;
      const double z = 2.0*drand48()-1.0;
      const double r2 = x*x+y*y+z*z;
      if (r2 < 1)
      {
        bodyIDs[np]   = ((unsigned long long) nSphere)*procId + np;

        bodyPositions[np].x = x;
        bodyPositions[np].y = y;
        bodyPositions[np].z = z;
        bodyPositions[np].w = (1.0/nSphere) * 1.0/nProcs;

        bodyVelocities[np].x = 0;
        bodyVelocities[np].y = 0;
        bodyVelocities[np].z = 0;
        bodyVelocities[np].w = 0;
        np++;
      }//if
    }//while
  }//else
  else if (nCube >= 0)
  {
    //CUBE
    if (procId == 0) printf("Using Cube model with n= %d per process \n", nSphere);
    assert(nCube >= 0);
    bodyPositions.resize(nCube);
    bodyVelocities.resize(nCube);
    bodyIDs.resize(nCube);

    srand48(procId+19840501);

    /* generate uniform sphere */
    for (int i= 0; i < nCube; i++)
    {
      const double x = 2*drand48()-1.0;
      const double y = 2*drand48()-1.0;
      const double z = 2*drand48()-1.0;

      bodyIDs[i]   =  ((unsigned long long) nCube)*procId + i;

      bodyPositions[i].x = x;
      bodyPositions[i].y = y;
      bodyPositions[i].z = z;
      bodyPositions[i].w = (1.0/nCube) * 1.0/nCube;

      bodyVelocities[i].x = 0;
      bodyVelocities[i].y = 0;
      bodyVelocities[i].z = 0;
      bodyVelocities[i].w = 0;
    }//
  }//else
  else if (diskmode)
  {
    if (procId == 0) printf("Using diskmode with filename %s\n", fileName.c_str());
    const int seed = procId+19840501;
    srand48(seed);
    const DiskShuffle disk(fileName);
    const int np = disk.get_ntot();
    bodyPositions.resize(np);
    bodyVelocities.resize(np);
    bodyIDs.resize(np);
    for (int i= 0; i < np; i++)
    {
      bodyIDs[i]   =  ((unsigned long long) np)*procId + i;

      bodyPositions[i].x = disk.pos(i).x;
      bodyPositions[i].y = disk.pos (i).y;
      bodyPositions[i].z = disk.pos (i).z;
      bodyPositions[i].w = disk.mass(i) * 1.0/nProcs;

      bodyVelocities[i].x = disk.vel(i).x;
      bodyVelocities[i].y = disk.vel(i).y;
      bodyVelocities[i].z = disk.vel(i).z;
      bodyVelocities[i].w = 0;
    }
  }
  else
    assert(0);

  tree->mpiSync();


#ifdef TIPSYOUTPUT
  LOGF(stderr, " t_current = %g\n", tree->get_t_current());
#endif


  //Set the properties of the data set, it only is really used by process 0, which does the 
  //actual file I/O  
  tree->setDataSetProperties(NTotal, NFirst, NSecond, NThird);

  if(procId == 0)  
    fprintf(stderr, "Dataset particle information: Ntotal: %d\tNFirst: %d\tNSecond: %d\tNThird: %d \n",
        NTotal, NFirst, NSecond, NThird);


  //Sanity check
  double mass = 0, totalMass;
  for(unsigned int i=0; i < bodyPositions.size(); i++)
  {
    mass += bodyPositions[i].w;
  }

  tree->load_kernels();

#ifdef USE_MPI
  MPI_Reduce(&mass,&totalMass,1, MPI_DOUBLE, MPI_SUM,0, mpiCommWorld);
#else
  totalMass = mass;
#endif

  if(procId == 0)   LOGF(stderr, "Combined Mass: %f \tNTotal: %d \n", totalMass, NTotal);


  fprintf(stderr,"Proc: %d Bootup times: Tree/MPI: %lg Threads/log: %lg IC-model: %lg \n",
                procId, tStartup-tStartupStart, tStartup2-tStartup, tEndModel - tStartModel);


  double t0 = tree->get_time();

  tree->localTree.setN((int)bodyPositions.size());
  tree->allocateParticleMemory(tree->localTree);

  //Load data onto the device
  for(uint i=0; i < bodyPositions.size(); i++)
  {
    tree->localTree.bodies_pos[i] = bodyPositions[i];
    tree->localTree.bodies_vel[i] = bodyVelocities[i];
    tree->localTree.bodies_ids[i] = bodyIDs[i];

    tree->localTree.bodies_Ppos[i] = bodyPositions[i];
    tree->localTree.bodies_Pvel[i] = bodyVelocities[i];
    tree->localTree.bodies_time[i] = make_float2(tree->get_t_current(), tree->get_t_current());


  }

  tree->localTree.bodies_time.h2d();
  tree->localTree.bodies_pos.h2d();
  tree->localTree.bodies_vel.h2d();
  tree->localTree.bodies_Ppos.h2d();
  tree->localTree.bodies_Pvel.h2d();
  tree->localTree.bodies_ids.h2d();

  //If required set the dust particles
  #ifdef USE_DUST
    if( (int)dustPositions.size() > 0)
    {
      LOGF(stderr, "Allocating dust properties for %d dust particles \n",
          (int)dustPositions.size());
      tree->localTree.setNDust((int)dustPositions.size());
      tree->allocateDustMemory(tree->localTree);

      //Load dust data onto the device
      for(uint i=0; i < dustPositions.size(); i++)
      {
        tree->localTree.dust_pos[i] = dustPositions[i];
        tree->localTree.dust_vel[i] = dustVelocities[i];
        tree->localTree.dust_ids[i] = dustIDs[i];
      }

      tree->localTree.dust_pos.h2d();
      tree->localTree.dust_vel.h2d();
      tree->localTree.dust_ids.h2d();
    }
  #endif //ifdef USE_DUST


  #ifdef USE_MPI
    //Sum all the particles to get total number of particles in the system
    tree->mpiSumParticleCount(tree->localTree.n);

    //Startup the OMP threads
    omp_set_num_threads(4);
  #endif


  //Start the integration
#ifdef USE_OPENGL
  octree::IterationData idata;
  initAppRenderer(argc, argv, tree, idata, displayFPS, stereo);
  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);
#else
  tree->mpiSync();
  if (procId==0) fprintf(stderr, " Starting iterating\n");


  bool simulationFinished = false;
  ioSharedData.writingFinished       = true;

  /* w/o MPI-IO use async fwrite, so use 2 threads
   * otherwise, use 1 threads
   */
#pragma omp parallel num_threads(1+ (!useMPIIO))
  {
    const int tid = omp_get_thread_num();
    if (tid == 0)
    {
      //Catch exceptions to add some extra print info
      try
      {
        tree->iterate();
      }
      catch(const std::exception &exc)
      {
        std::cerr << "Process: "  << procId << "\t" << exc.what() <<std::endl;
        if(nProcs > 1) ::abort();
      }
      catch(...)
      {
        std::cerr << "Unknown exception on process: " << procId << std::endl;
        if(nProcs > 1) ::abort();
      }
      simulationFinished = true;
    }
    else
    {
      assert(!useMPIIO);
      /* IO */
      sleep(1);
      while(!simulationFinished)
      {
        if(ioSharedData.writingFinished == false)
        {
          const int n           = ioSharedData.nBodies;
          const float t_current = ioSharedData.t_current;

          string fileName; fileName.resize(256);
          sprintf(&fileName[0], "%s_%010.4f", snapshotFile.c_str(), t_current);

          if(nProcs <= 16)
          {
            tree->write_dumbp_snapshot_parallel(ioSharedData.Pos, ioSharedData.Vel,
                ioSharedData.IDs, n, fileName.c_str(), t_current) ;

          }
          else
          {
            sprintf(&fileName[0], "%s_%010.4f-%d", snapshotFile.c_str(), t_current, procId);
            tree->write_snapshot_per_process(ioSharedData.Pos, ioSharedData.Vel,
                ioSharedData.IDs, n,
                fileName.c_str(), t_current) ;
          }
          ioSharedData.free();
          assert(ioSharedData.writingFinished == false);
          ioSharedData.writingFinished = true;
        }
        else
        {
          usleep(100);
        }
      }
    }
  }

  if (useMPIIO)
    tree->terminateIO();

  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);


  std::stringstream sstemp;
  sstemp << "Finished total took: " << tree->get_time()-t0 << std::endl;
  std::string stemp = sstemp.str();
  tree->writeLogData(stemp);
  tree->writeLogToFile();//Final write incase anything is left in the buffers

  if(tree->procId == 0)
  {
    LOGF(stderr, "TOTAL:   Time spent between the start of 'iterate' and the final time-step (very first step is not accounted)\n",0);
    LOGF(stderr, "Grav:    Time spent to compute gravity, including communication (wall-clock time)\n",0);
    LOGF(stderr, "GPUgrav: Time spent ON the GPU to compute local and LET gravity\n",0);
    LOGF(stderr, "LET Com: Time spent in exchanging and building LET data\n",0);
    LOGF(stderr, "Build:   Time spent in constructing the tree (incl sorting, making groups, etc.)\n",0);
    LOGF(stderr, "Domain:  Time spent in computing new domain decomposition and exchanging particles between nodes.\n",0);
    LOGF(stderr, "Wait:    Time spent in waiting on other processes after the gravity part.\n",0);
  }


  delete tree;
  tree = NULL;
  

#endif

  displayTimers();

#ifdef USE_MPI
  if (!mpiInitialized)
    MPI_Finalize();
#endif
  return 0;
}
