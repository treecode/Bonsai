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

/*
 *
 * TODO
 * Close BonsaiIO on destruction to properly close open File handles
 * Fix the block time stepping
 * Add block time-stepping to the multi-GPU code
 *
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
#include <thread>
#include <chrono>
#include <sys/time.h>
#include <omp.h>
#include "log.h"
#include "anyoption.h"
#include "renderloop.h"

#include "read_tipsy_file_parallel.h"

#include <array>

#include <FileIO.h>
#include <ICGenerators.h>


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

#ifdef WAR_OF_GALAXIES
#include <stdexcept>
void throw_if_flag_is_used(AnyOption const& opt, std::vector<std::string> arguments)
{
  for (auto const& arg : arguments)
    // Error in AnyOption: getter function not const
    if (const_cast<AnyOption&>(opt).getFlag(arg.c_str())) throw std::runtime_error(arg + " is not valid in war-of-galaxy mode.");
}

void throw_if_option_is_used(AnyOption const& opt, std::vector<std::string> arguments)
{
  for (auto const& arg : arguments)
    // Error in AnyOption: getter function not const
    if (const_cast<AnyOption&>(opt).getValue(arg.c_str()) != nullptr) throw std::runtime_error(arg + " is not valid in war-of-galaxy mode.");
}
#endif




double get_time_main()
{
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
}


/*
 * This thread watches if the simulation progresses or if it 'hangs'
 * this sometime happens (for unknown) reasons on large runs with file writing.
 * To prevent usage of precious compute cycles this thread kills the program 
 * after 360 seconds of no progress.
 */
void watchThread(octree *bonsai)
{
	double tCurrent = 0;
	do 
	{
	  std::this_thread::sleep_for(std::chrono::seconds(360));
	  double tNew = bonsai->getTime();
	  if(tNew != tCurrent)
	  {
		tCurrent = tNew;
	  }
	  else
	  {
		fprintf(stderr,"No progress detected for 360 seconds, the program will now exit!\n");
		::exit(0);
	  }
	}
	while(true);
}



//Buffers and flags used for the IO thread
volatile IOSharedData_t ioSharedData;

long long my_dev::base_mem::currentMemUsage;
long long my_dev::base_mem::maxMemUsage;

int main(int argc, char** argv, MPI_Comm comm, int shrMemPID)
{
  my_dev::base_mem::currentMemUsage = 0;
  my_dev::base_mem::maxMemUsage     = 0;

  vector<real4>   bodyPositions;
  vector<real4>   bodyVelocities;
  vector<ullong>  bodyIDs;

 
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

  bool mpiRenderMode = false;
  
  

  int nPlummer  = -1;
  int nSphere   = -1;
  int nCube     = -1;
  int nMilkyWay = -1;
  int nMWfork   =  4;

  std::string wogPath;
  int wogPort = 50007;
  real wogCameraDistance = 500.0;
  real wogDeletionRadiusFactor = 1.0;

  int galSeed   =  0;
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


 		ADDUSAGE("Bonsai command line usage:");
 		ADDUSAGE("                  ");
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
		ADDUSAGE("     --seed     #       seed to use for the Milky Way  [" << galSeed  << "]");
    ADDUSAGE("     --taskvar  #       variable name to obtain task id [for randoms seed] before MPI_Init. \n");
#endif
    ADDUSAGE("     --plummer  #       use Plummer model with # particles per proc");
		ADDUSAGE("     --sphere   #       use spherical model with # particles per proc");
		ADDUSAGE("     --cube     #       use cube model with # particles per proc");
    ADDUSAGE("     --diskmode         use diskmode to read same input file all MPI taks and randomly shuffle its positions");
    ADDUSAGE("     --mpirendermode    use MPI to communicate with the renderer. Must only be used with bonsai_driver. [disabled]");
#ifdef WAR_OF_GALAXIES
    ADDUSAGE("     --war-of-galaxies #    input path for WarOfGalaxies");
    ADDUSAGE("     --port #               Port for WarOfGalaxies");
    ADDUSAGE("     --camera-distance #    OpenGL camera distance for WarOfGalaxies");
    ADDUSAGE("     --del-radius-factor #  Scaling factor of deletion sphere for WarOfGalaxies");
#endif
		ADDUSAGE(" ");


		opt.setFlag( "help" ,   'h');
		opt.setFlag( "diskmode");
		opt.setFlag( "mpirendermode");
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
    opt.setOption( "seed");
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
	opt.setOption("war-of-galaxies");
	opt.setOption("port");
	opt.setOption("camera-distance");
	opt.setOption("del-radius-factor");

    opt.processCommandArgs( argc, argv );


    if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
    {
      /* print usage if no options or requested help */
      opt.printUsage();
      ::exit(0);
    }

    if (opt.getFlag("direct"))          direct        = true;
    if (opt.getFlag("restart"))         restartSim    = true;
    if (opt.getFlag("displayfps"))      displayFPS    = true;
    if (opt.getFlag("diskmode"))        diskmode      = true;
    if (opt.getFlag("mpirendermode"))   mpiRenderMode = true;
    if(opt.getFlag("stereo"))           stereo        = true;

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
    if ((optarg = opt.getValue("seed")))         galSeed            = atoi(optarg);
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
    if ((optarg = opt.getValue("war-of-galaxies")))   wogPath                 = string(optarg);
    if ((optarg = opt.getValue("port")))              wogPort                 = atoi(optarg);
    if ((optarg = opt.getValue("camera-distance")))   wogCameraDistance       = atof(optarg);
    if ((optarg = opt.getValue("del-radius-factor"))) wogDeletionRadiusFactor = atof(optarg);
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

#ifdef WAR_OF_GALAXIES
    /// WarOfGalaxies: Deactivate unneeded flags using WarOfGalaxies
    throw_if_flag_is_used(opt, {{"direct", "restart", "diskmode", "stereo", "prepend-rank"}});
    throw_if_option_is_used(opt, {{"plummer", "milkyway", "mwfork", "sphere", "tend", "iend",
      "snapname", "snapiter", "rmdist", "valueadd", "rebuild", "reducedust", "gameMode"}});
#endif

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
        //do not scale until we know the number of processors
        generateGalacticsModel(procId, 32768*7, galSeed, nMilkyWay, nMWfork,
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


#ifdef USE_OPENGL
  // create OpenGL context first, and register for interop
  initGL(argc, argv, fullScreenMode.c_str(), stereo);
  cudaGLSetGLDevice(devID); //TODO should this not be renderDev?
#endif

  initTimers();

#ifdef WIN32
  int pid = _getpid();
#else
  int pid = (int)getpid();
#endif

#if 0
  //Used for profiler, note this has to be done before initing to
  //octree otherwise it has no effect...Therefore use pid instead of mpi procId
  char *gpu_prof_log;
  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
  if(gpu_prof_log){
    char tmp[50];
    sprintf(tmp,"process_%d_%s",pid,gpu_prof_log);
    #ifdef WIN32
                SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
    #else
                setenv("CUDA_PROFILE_LOG",tmp,1);
        LOGF(stderr, "TESTING log on proc: %d val: %s \n", pid, tmp);
    #endif
  }
#endif

  int mpiInitialized =  0;
  int procId         = -1;
  int nProcs         = -1;
  
#ifdef USE_MPI  
      MPI_Initialized(&mpiInitialized);
      MPI_Comm mpiCommWorld = MPI_COMM_WORLD;
      if (!mpiInitialized)
      {
        #ifdef _MPIMT
            int provided;
            MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
            assert(MPI_THREAD_MULTIPLE == provided);
        #else
            MPI_Init(&argc, &argv);
        #endif
        shrMemPID = 0;
      }
      else
      {
        //MPI environment initialized by the driver program
        mpiCommWorld = comm;
      }

      MPI_Comm_size(mpiCommWorld, &nProcs);
      MPI_Comm_rank(mpiCommWorld, &procId);
#else
    MPI_Comm mpiCommWorld = 0;
    procId                = 0;
    nProcs                = 1;
#endif
#if ENABLE_LOG
  PREPEND_RANK_PROCID = procId;
  PREPEND_RANK_NPROCS = nProcs;
#endif

    if (mpiRenderMode) assert(mpiInitialized); //The external renderer requires a working MPI environment


    if(nProcs > 1)
    {
      logFileName.append("-");
      char buff[16];
      sprintf(buff,"%d-%d", nProcs, procId);
      logFileName.append(buff);
    }

    //Use a string stream buffer, only write data at the end of the run
    std::stringstream logStream;
    ostream &logFile = logStream;

    my_dev::context cudaContext;

    if(nProcs > 1) devID = procId % getNumberOfCUDADevices();

    cudaContext.create(logFile, false); //Do logging to file and enable timing (false = enabled)
    cudaContext.createQueue(devID);

    char logPretext[64];
    sprintf(logPretext, "PROC-%05d ", procId);
    cudaContext.setLogPreamble(logPretext);




    //Create the octree class and set the properties
    octree *tree = new octree(  mpiCommWorld,
                                &cudaContext,
                                argv, devID, theta, eps,
                                snapshotFile, snapshotIter,
                                quickDump, quickRatio, quickSync,
                                useMPIIO,mpiRenderMode,
                                timeStep,
                                tEnd, iterEnd,
                                rebuild_tree_rate, direct, shrMemPID);




  double tStartup = tree->get_time();


  if (procId == 0)
  {
    //Note can't use LOGF here since MPI isn't initialized yet
    cerr << "[INIT]\tUsed settings: \n";
    cerr << "[INIT]\tInput  filename "      << fileName                                                      << endl;
    cerr << "[INIT]\tBonsai filename "      << bonsaiFileName                                                << endl;
    cerr << "[INIT]\tLog filename "         << logFileName                                                   << endl;
    cerr << "[INIT]\tTheta: \t\t"           << theta                << "\t\teps: \t\t"      << eps           << endl;
    cerr << "[INIT]\tTimestep: \t"          << timeStep             << "\t\ttEnd: \t\t"     << tEnd          << endl;
    cerr << "[INIT]\titerEnd: \t"           << iterEnd                                                       << endl;
    cerr << "[INIT]\tUse MPI-IO: \t"        << (useMPIIO ? "YES" : "NO")                                     << endl;
    cerr << "[INIT]\tsnapshotFile: \t"      << snapshotFile          << "\tsnapshotIter: \t" << snapshotIter << endl;
    if (useMPIIO)
    {
      cerr << "[INIT]\t  quickDump: \t"      << quickDump << "\t\tquickRatio: \t" << quickRatio << endl;
    }
    cerr << "[INIT]\tInput file: \t"        << fileName     << "\t\tdevID: \t\t"        << devID << endl;
    cerr << "[INIT]\tRemove dist: \t"   << remoDistance << endl;
    cerr << "[INIT]\tRebuild tree every " << rebuild_tree_rate << " timestep\n";


    if( reduce_bodies_factor > 1 ) cerr << "[INIT]\tReduce number of non-dust bodies by " << reduce_bodies_factor << " \n";
    if( reduce_dust_factor   > 1 ) cerr << "[INIT]\tReduce number of dust bodies by " << reduce_dust_factor << " \n";

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



  double tStartup2 = tree->get_time();  

  if (!bonsaiFileName.empty() && useMPIIO)
  {
#ifdef USE_MPI        
    //Read a BonsaiIO file
    const MPI_Comm &comm = mpiCommWorld;
    float tCurrent = 0;
    tree->lReadBonsaiFile(
        bodyPositions, 
        bodyVelocities,
        bodyIDs,
        tCurrent,
        bonsaiFileName,
        procId, nProcs, comm,
        restartSim,
        reduce_bodies_factor);
    tree->set_t_current(tCurrent);
#else
    fprintf(stderr,"Usage of these options requires to code to be built with MPI support!\n"); exit(0);
#endif      
  }
  else if ((nPlummer == -1 && nSphere == -1  && nCube == -1 && !diskmode && nMilkyWay == -1) || restartSim)
  {
#ifdef WAR_OF_GALAXIES
      std::cout << "WarOfGalaxies: Input file is used as dummy particles." << std::endl;
      for (auto & id : bodyIDs) id = id - id % 10 + 9;

      // get center of mass
      real mass;
      real4 center_of_mass = make_real4(0.0, 0.0, 0.0, 0.0);

      for (auto const& p : bodyPositions)
      {
        mass = p.w;
        center_of_mass.x += mass * p.x;
        center_of_mass.y += mass * p.y;
        center_of_mass.z += mass * p.z;
        center_of_mass.w += mass;
      }

      center_of_mass.x /= center_of_mass.w;
      center_of_mass.y /= center_of_mass.w;
      center_of_mass.z /= center_of_mass.w;

      // get center-of-mass velocity
      real4 center_of_mass_velocity = make_real4(0.0, 0.0, 0.0, 0.0);

      for (size_t i = 0; i < bodyVelocities.size(); i++)
      {
        mass = bodyPositions[i].w;
        center_of_mass_velocity.x += mass * bodyVelocities[i].x;
        center_of_mass_velocity.y += mass * bodyVelocities[i].y;
        center_of_mass_velocity.z += mass * bodyVelocities[i].z;
        center_of_mass_velocity.w += mass;
      }

      center_of_mass_velocity.x /= center_of_mass_velocity.w;
      center_of_mass_velocity.y /= center_of_mass_velocity.w;
      center_of_mass_velocity.z /= center_of_mass_velocity.w;

      // shift center of mass to position (0,0,10000)
      for (auto &p : bodyPositions)
      {
        p.x -= center_of_mass.x;
        p.y -= center_of_mass.y;
        p.z -= center_of_mass.z + 10000;
      }

      // steady
      for (auto &v : bodyVelocities)
      {
        v.x -= center_of_mass_velocity.x;
        v.y -= center_of_mass_velocity.y;
        v.z -= center_of_mass_velocity.z;
      }
#endif
    float sTime = 0;
    tree->fileIO->readFile(mpiCommWorld, bodyPositions, bodyVelocities, bodyIDs, fileName,
                           procId, nProcs, sTime, reduce_bodies_factor, restartSim);
    tree->set_t_current((float) sTime);
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

          generateGalacticsModel(procId, nProcs, galSeed, nMilkyWay, nMWfork,
                                 true, bodyPositions, bodyVelocities, bodyIDs);
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
    generatePlummerModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nPlummer);
  }
  else if (nSphere >= 0)
  {
    generateSphereModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nSphere);
  }//else
  else if (nCube >= 0)
  {
    generateCubeModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, nCube);
  }//else
  else if (diskmode)
  {
    generateShuffledDiskModel(bodyPositions, bodyVelocities, bodyIDs, procId, nProcs, fileName);
  }
  else
    assert(0);

  tree->mpiSync();

  //Sanity check
  double mass = 0;
  for(unsigned int i=0; i < bodyPositions.size(); i++) { mass += bodyPositions[i].w; }
  double totalMass = tree->SumOnRootRank(mass);

  tree->mpiSumParticleCount((int)bodyPositions.size());

  LOGF(stderr, "t_current = %g nLocal %d massLocal: %f Combined Mass: %f nGlobal: %llu \n",
                tree->get_t_current(), (int)bodyPositions.size(),
                mass, totalMass, tree->nTotalFreq_ull);
  fprintf(stderr,"Proc: %d Bootup times: Tree/MPI: %lg Threads/log: %lg IC-model: %lg \n",
                 procId, tStartup-tStartupStart, tStartup2-tStartup, tEndModel - tStartModel);
  tree->load_kernels();

  double t0 = tree->get_time();

  tree->localTree.setN((int)bodyPositions.size());
  tree->allocateParticleMemory(tree->localTree);

  //Load data onto the device
  for(uint i=0; i < bodyPositions.size(); i++)
  {
    tree->localTree.bodies_pos[i]  = bodyPositions[i];
    tree->localTree.bodies_Ppos[i] = bodyPositions[i];
    tree->localTree.bodies_vel[i]  = bodyVelocities[i];
    tree->localTree.bodies_Pvel[i] = bodyVelocities[i];
    tree->localTree.bodies_ids[i]  = bodyIDs[i];
    tree->localTree.bodies_time[i] = make_float2(tree->get_t_current(), tree->get_t_current());
  }

  tree->localTree.bodies_time.h2d();
  tree->localTree.bodies_pos. h2d();
  tree->localTree.bodies_vel. h2d();
  tree->localTree.bodies_Ppos.h2d();
  tree->localTree.bodies_Pvel.h2d();
  tree->localTree.bodies_ids. h2d();


  #ifdef USE_MPI
    omp_set_num_threads(4); //Startup the OMP threads to be used during LET phase
  #endif


  //Start the integration
#ifdef USE_OPENGL
  octree::IterationData idata;
  initAppRenderer(argc, argv, tree, idata, displayFPS, stereo,
	wogPath, wogPort, wogCameraDistance, wogDeletionRadiusFactor);
  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);
#else
  tree->mpiSync();
  if (procId==0) fprintf(stderr, " Start iterating\n");


  bool simulationFinished = false;
  ioSharedData.writingFinished       = true;

  //Uncomment this to have a monitor thread that can kill the 
  //program when no activity is detected
  //std::thread watcher(watchThread, tree);
  //watcher.detach();



  /* w/o MPI-IO use async fwrite, so use 2 threads otherwise, use 1 threads
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
          const float t_current = ioSharedData.t_current;

          bool distributed = true;
          string fileName; fileName.resize(256);
          sprintf(&fileName[0], "%s_%010.4f-%d", snapshotFile.c_str(), t_current, procId);


          if(nProcs <= 16)
          {
//              distributed = false;
//              sprintf(&fileName[0], "%s_%010.4f", snapshotFile.c_str(), t_current);
          }

          tree->fileIO->writeFile(ioSharedData.Pos, ioSharedData.Vel,
                                  ioSharedData.IDs, ioSharedData.nBodies,
                                  fileName.c_str(), t_current,
                                  procId, nProcs, mpiCommWorld, distributed) ;

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

  if (useMPIIO) tree->terminateIO();

  LOG("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);


  std::stringstream sstemp;
  sstemp << "Finished total took: " << tree->get_time()-t0 << std::endl;
  std::string stemp = sstemp.str();
  tree->writeLogData(stemp);
  tree->writeLogToFile();//Final write in case anything is left in the buffers

  if(tree->procId == 0)
  {
    LOGF(stderr, "TOTAL:   Time spent between the start of 'iterate' and the final time-step (very first step is not accounted)\n");
    LOGF(stderr, "Grav:    Time spent to compute gravity, including communication (wall-clock time)\n");
    LOGF(stderr, "GPUgrav: Time spent ON the GPU to compute local and LET gravity\n");
    LOGF(stderr, "LET Com: Time spent in exchanging and building LET data\n");
    LOGF(stderr, "Build:   Time spent in constructing the tree (incl sorting, making groups, etc.)\n");
    LOGF(stderr, "Domain:  Time spent in computing new domain decomposition and exchanging particles between nodes.\n");
    LOGF(stderr, "Wait:    Time spent in waiting on other processes after the gravity part.\n");
  }


  delete tree;
  tree = NULL;

#endif

  displayTimers();

#ifdef USE_MPI
  //Finalize MPI if we initialized it ourselves, otherwise the driver will do it.
  if (!mpiInitialized) MPI_Finalize();
#endif
  return 0;
}
