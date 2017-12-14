#undef NDEBUG
#include "octree.h"
#include "postProcessModules.h"
#include "thrust_war_of_galaxies.h"

#include <iostream>
#include <algorithm>

using namespace std;

static double de_max  = 0;
static double dde_max = 0;  


cudaEvent_t startLocalGrav;
cudaEvent_t startRemoteGrav;
cudaEvent_t endLocalGrav;
cudaEvent_t endRemoteGrav;

float runningLETTimeSum, lastTotal, lastLocal;


void octree::makeLET()
{
#ifdef USE_MPI
   //LET code test
  double t00 = get_time();

  //Start copies, while grpTree info is exchanged
  localTree.boxSizeInfo.d2h  (  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxCenterInfo.d2h(  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.multipole.d2h    (3*localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxSizeInfo.waitForCopyEvent();
  localTree.boxCenterInfo.waitForCopyEvent();
  
  double t10 = get_time();
  //Exchange domain grpTrees, while memory copies take place
  this->sendCurrentInfoGrpTree();

  double t20 = get_time();


  localTree.multipole.waitForCopyEvent();
  double t40 = get_time();
  LOGF(stderr,"MakeLET Preparing data-copy: %lg  sendGroups: %lg Total: %lg \n",
               t10-t00, t20-t10, t40-t00);

  std::vector<real4> topLevelsBuffer;
  std::vector<uint2> treeSizeAndOffset;
  int copyTreeUpToLevel = 0;
  //Start LET kernels
  essential_tree_exchangeV2(localTree,
                            remoteTree,
                            topLevelsBuffer,
                            treeSizeAndOffset,
                            copyTreeUpToLevel);

  letRunning = false;
#endif  
}

extern void read_tipsy_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<int> &bodiesIDs,  float eps2, string fileName, 
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<int> &dustIDs, int reduce_bodies_factor, int reduce_dust_factor, const bool restart) ;
extern int setupMergerModel(vector<real4> &bodyPositions1,      vector<real4> &bodyVelocities1,
                            vector<int>   &bodyIDs1,            vector<real4> &bodyPositions2,
                            vector<real4> &bodyVelocities2,     vector<int>   &bodyIDs2);

bool octree::addGalaxy(int galaxyID)
{
    //To add an galaxy we need to have read it in from the host
    #ifdef USE_DUST
      //We move the dust data into the position data (on the device :) )
      localTree.bodies_pos.copy_devonly(localTree.dust_pos, localTree.n_dust, localTree.n);
      localTree.bodies_vel.copy_devonly(localTree.dust_vel, localTree.n_dust, localTree.n);
      localTree.bodies_ids.copy_devonly(localTree.dust_ids, localTree.n_dust, localTree.n);
    #endif
  
    this->localTree.bodies_pos.d2h();
    this->localTree.bodies_vel.d2h();
    this->localTree.bodies_ids.d2h();
    
    vector<real4> newGalaxy_pos;
    vector<real4> newGalaxy_vel;
    vector<int> newGalaxy_ids;
    vector<real4> currentGalaxy_pos;
    vector<real4> currentGalaxy_vel;
    vector<int>   currentGalaxy_ids;    

    int n_particles = this->localTree.n + this->localTree.n_dust;
    currentGalaxy_pos.insert(currentGalaxy_pos.begin(), &this->localTree.bodies_pos[0],
                          &this->localTree.bodies_pos[0]+n_particles);
    currentGalaxy_vel.insert(currentGalaxy_vel.begin(), &this->localTree.bodies_vel[0],
                          &this->localTree.bodies_vel[0]+n_particles);
    currentGalaxy_ids.insert(currentGalaxy_ids.begin(), &this->localTree.bodies_ids[0],
                          &this->localTree.bodies_ids[0]+n_particles);    
    
    vector<real4> newGalaxy_pos_dust;
    vector<real4> newGalaxy_vel_dust;
    vector<int> newGalaxy_ids_dust;    
    
    //string fileName = "model3_child_compact.tipsy";
    //string fileName = "modelC30kDust.bin";
    string fileName = "/local/doserbd/projects/ESO/war-of-galaxies/testSolar_5M_galaxy2.tipsy";
    int rank = 0;
    int procs = 1;
    int NTotal, NFirst, NSecond, Nthird = 0;
    int reduce_bodies = 50;
    read_tipsy_file_parallel(newGalaxy_pos, newGalaxy_vel, newGalaxy_ids, 0, fileName, 
                             rank, procs, NTotal, NFirst, NSecond, NThird, this,
                             newGalaxy_pos_dust, newGalaxy_vel_dust, newGalaxy_ids_dust,
                             reduce_bodies, 1, false);
    
void octree::iterate_setup() {

  if(execStream == NULL)
  {
      if(execStream == NULL)          execStream          = new my_dev::dev_stream(0);
      if(gravStream == NULL)          gravStream          = new my_dev::dev_stream(0);
      if(copyStream == NULL)          copyStream          = new my_dev::dev_stream(0);
      if(LETDataToHostStream == NULL) LETDataToHostStream = new my_dev::dev_stream(0);

      CU_SAFE_CALL(cudaEventCreate(&startLocalGrav));
      CU_SAFE_CALL(cudaEventCreate(&endLocalGrav));
      CU_SAFE_CALL(cudaEventCreate(&startRemoteGrav));
      CU_SAFE_CALL(cudaEventCreate(&endRemoteGrav));

      devContext->writeLogEvent("Start execution\n");
  }

  //Setup of the multi-process particle distribution, initially it should be equal
  #ifdef USE_MPI
    if(nProcs > 1)
    {
      for(int i=0; i < 5; i++)
      {
        double notUsed     = 0;
        int maxN = 0, minN = 0;
        sort_bodies(localTree, true, true); //Initial sort to get global boundaries to compute keys
        parallelDataSummary(localTree, 30, 30, notUsed, notUsed, true); //1 for all process, equal part distribution

        //Check if the min/max are within certain percentage
        MPI_Allreduce(&localTree.n, &maxN, 1, MPI_INT, MPI_MAX, mpiCommWorld);
        MPI_Allreduce(&localTree.n, &minN, 1, MPI_INT, MPI_MIN, mpiCommWorld);

        //Compute difference in percent
        int perc = (int)(100*(maxN-minN)/(double)minN);

        if(procId == 0)
        {
          LOGF(stderr, "Particle setup iteration: %d Min: %d  Max: %d Diff: %d %%\n", i, minN, maxN, perc);
        }
        if(perc < 10) break; //We're happy if difference is less than 10%
      }
    }
  #endif

  sort_bodies(localTree, true, true); //Initial sort to get global boundaries to compute keys
  letRunning      = false;
}

void octree::releaseGalaxy(Galaxy const& galaxy)
{
  // Get particle data back to the host so we can add our new data
  this->localTree.bodies_pos.d2h();
  this->localTree.bodies_acc0.d2h();
  this->localTree.bodies_vel.d2h();
  this->localTree.bodies_time.d2h();
  this->localTree.bodies_ids.d2h();
  this->localTree.bodies_Ppos.d2h();
  this->localTree.bodies_Pvel.d2h();

  vector<real4> new_pos;
  vector<real4> new_vel;
  vector<int> new_ids;
  int old_nb_particles = this->localTree.n;
  int new_nb_particles = old_nb_particles + galaxy.pos.size();

  for (int i(0); i != old_nb_particles; ++i)
  {
    new_pos.push_back(this->localTree.bodies_pos[i]);
    new_vel.push_back(this->localTree.bodies_vel[i]);
    new_ids.push_back(this->localTree.bodies_ids[i]);
  }

  for (int i(0); i != galaxy.pos.size(); ++i)
  {
    new_pos.push_back(galaxy.pos[i]);
    new_vel.push_back(galaxy.vel[i]);
    new_ids.push_back(galaxy.ids[i]);
  }

  // Set new size of the buffers
  this->localTree.setN(new_nb_particles);

  // Resize preserves original data
  this->reallocateParticleMemory(this->localTree);

  // Copy back to host storage
  memcpy(&this->localTree.bodies_pos[0], &new_pos[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_vel[0], &new_vel[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_ids[0], &new_ids[0], sizeof(int) * new_nb_particles);

  float2 curTime = this->localTree.bodies_time[0];
  for (int i(0); i != new_nb_particles; ++i)
    this->localTree.bodies_time[i] = curTime;
  for (int i(old_nb_particles); i != new_nb_particles; ++i)
    this->localTree.bodies_acc0[i] = make_float4(0.0, 0.0, 0.0, 0.0);
  this->localTree.bodies_acc1.zeroMem();

  this->localTree.bodies_pos.h2d();
  this->localTree.bodies_acc0.h2d();
  this->localTree.bodies_vel.h2d();
  this->localTree.bodies_time.h2d();
  this->localTree.bodies_ids.h2d();

  // Fill the predicted arrays
  this->localTree.bodies_Ppos.copy(this->localTree.bodies_pos, localTree.n);
  this->localTree.bodies_Pvel.copy(this->localTree.bodies_pos, localTree.n);

  resetEnergy();
}

void octree::removeGalaxy(int user_id)
{
  // Get particle data back to the host so we can add our new data
  this->localTree.bodies_pos.d2h();
  this->localTree.bodies_vel.d2h();
  this->localTree.bodies_ids.d2h();
  this->localTree.bodies_acc0.d2h();
  this->localTree.bodies_time.d2h();
  this->localTree.bodies_Ppos.d2h();
  this->localTree.bodies_Pvel.d2h();

  vector<real4> new_pos;
  vector<real4> new_vel;
  vector<int> new_ids;
  int old_nb_particles = this->localTree.n;
  int new_nb_particles = 0;

  for (int i(0); i != old_nb_particles; ++i)
  {
	if (this->localTree.bodies_ids[i] % 10 == user_id) continue;
    new_pos.push_back(this->localTree.bodies_pos[i]);
    new_vel.push_back(this->localTree.bodies_vel[i]);
    new_ids.push_back(this->localTree.bodies_ids[i]);
    ++new_nb_particles;
  }

  // Set new size of the buffers
  this->localTree.setN(new_nb_particles);

  // Resize preserves original data
  this->reallocateParticleMemory(this->localTree);

  // Copy back to host storage
  memcpy(&this->localTree.bodies_pos[0], &new_pos[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_vel[0], &new_vel[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_ids[0], &new_ids[0], sizeof(int) * new_nb_particles);

  float2 curTime = this->localTree.bodies_time[0];
  for(int i=0; i < this->localTree.n; i++)
    this->localTree.bodies_time[i] = curTime;
  this->localTree.bodies_acc1.zeroMem();

  this->localTree.bodies_pos.h2d();
  this->localTree.bodies_acc0.h2d();
  this->localTree.bodies_vel.h2d();
  this->localTree.bodies_time.h2d();
  this->localTree.bodies_ids.h2d();

  // Fill the predicted arrays
  this->localTree.bodies_Ppos.copy(this->localTree.bodies_pos, localTree.n);
  this->localTree.bodies_Pvel.copy(this->localTree.bodies_pos, localTree.n);

  resetEnergy();
}

void octree::removeParticles(real deletion_radius_square, my_dev::dev_mem<uint> &user_particles, int number_of_users)
{
#ifdef USE_THRUST

  user_particles.h2d();
  remove_particles(this->localTree, deletion_radius_square, user_particles, number_of_users);
  user_particles.d2h();

  // Resize preserves original data
  this->reallocateParticleMemory(this->localTree);

#else

  // Get particle data back to the host so we can add our new data
  this->localTree.bodies_pos.d2h();
  this->localTree.bodies_vel.d2h();
  this->localTree.bodies_ids.d2h();
  this->localTree.bodies_acc0.d2h();
  this->localTree.bodies_time.d2h();
  this->localTree.bodies_Ppos.d2h();
  this->localTree.bodies_Pvel.d2h();

  vector<real4> new_pos;
  vector<real4> new_vel;
  vector<int> new_ids;
  int old_nb_particles = this->localTree.n;
  int new_nb_particles = 0;

  for (int i(0); i != old_nb_particles; ++i)
  {
    real4 position = this->localTree.bodies_pos[i];
	if (position.x * position.x + position.y * position.y + position.z * position.z > deletion_radius_square
	    and this->localTree.bodies_ids[i] % 10 != 9)
	{
	  --user_particles[this->localTree.bodies_ids[i] % 10];
      continue;
	}
    new_pos.push_back(this->localTree.bodies_pos[i]);
    new_vel.push_back(this->localTree.bodies_vel[i]);
    new_ids.push_back(this->localTree.bodies_ids[i]);
    ++new_nb_particles;
  }

  // Set new size of the buffers
  this->localTree.setN(new_nb_particles);

  // Resize preserves original data
  this->reallocateParticleMemory(this->localTree);

  // Copy back to host storage
  memcpy(&this->localTree.bodies_pos[0], &new_pos[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_vel[0], &new_vel[0], sizeof(real4) * new_nb_particles);
  memcpy(&this->localTree.bodies_ids[0], &new_ids[0], sizeof(int) * new_nb_particles);

  float2 curTime = this->localTree.bodies_time[0];
  for(int i=0; i < this->localTree.n; i++)
    this->localTree.bodies_time[i] = curTime;
  this->localTree.bodies_acc1.zeroMem();

  this->localTree.bodies_pos.h2d();
  this->localTree.bodies_acc0.h2d();
  this->localTree.bodies_vel.h2d();
  this->localTree.bodies_time.h2d();
  this->localTree.bodies_ids.h2d();

  // Fill the predicted arrays
  this->localTree.bodies_Ppos.copy(this->localTree.bodies_pos, localTree.n);
  this->localTree.bodies_Pvel.copy(this->localTree.bodies_pos, localTree.n);

#endif

  resetEnergy();
}

// returns true if this iteration is the last (t_current >= t_end), false otherwise
bool octree::iterate_once(IterationData &idata) {
    double t1 = 0;

    //if(t_current < 1) //Clear startup timings
    //if(0)
    if(iter < 32)
    {
      idata.totalGPUGravTimeLocal = 0;
      idata.totalGPUGravTimeLET   = 0;
      idata.totalLETCommTime      = 0;
      idata.totalBuildTime        = 0;
      idata.totalDomTime          = 0;
      idata.lastWaitTime          = 0;
      idata.startTime             = get_time();
      idata.totalGravTime         = 0;
      idata.totalDomUp            = 0;
      idata.totalDomEx            = 0;
      idata.totalDomWait          = 0;
      idata.totalPredCor          = 0;
    }


    LOG("At the start of iterate:\n");
    
    bool forceTreeRebuild = false;
    bool needDomainUpdate = true;

    double tTempTime = get_time();

    //predict local tree
    devContext->startTiming(execStream->s());
    predict(this->localTree);
    devContext->stopTiming("Predict", 9, execStream->s());

    idata.totalPredCor += get_time() - tTempTime;

    if(nProcs > 1)
    {
      //if(1) //Always update domain boundaries/particles
      if((iter % rebuild_tree_rate) == 0)
      {
        double domUp =0, domEx = 0;
        double tZ = get_time();
        devContext->startTiming(execStream->s());
        parallelDataSummary(localTree, lastTotal, lastLocal, domUp, domEx, false);
        devContext->stopTiming("UpdateDomain", 6, execStream->s());
        double tZZ = get_time();
        idata.lastDomTime   = tZZ-tZ;
        idata.totalDomTime += idata.lastDomTime;

        idata.totalDomUp += domUp;
        idata.totalDomEx += domEx;

        devContext->startTiming(execStream->s());
        mpiSync();
        devContext->stopTiming("DomainUnbalance", 12, execStream->s());

        idata.totalDomWait += get_time()-tZZ;

        needDomainUpdate    = false; //We did a boundary sync in the parallel decomposition part
        needDomainUpdate    = true; //TODO if I set it to false results degrade. Check why, for now just updte
      }
    }

    if (useDirectGravity)
    {
      devContext->startTiming(gravStream->s());
      direct_gravity(this->localTree);
      devContext->stopTiming("Direct_gravity", 4);
    }
    else
    {
      //Build the tree using the predicted positions
      // bool rebuild_tree = Nact_since_last_tree_rebuild > 4*this->localTree.n;   
      bool rebuild_tree = true;

      rebuild_tree = ((iter % rebuild_tree_rate) == 0);
      if(rebuild_tree)
      {
        //Rebuild the tree
        t1 = get_time();
        this->sort_bodies(this->localTree, needDomainUpdate);
        this->build(this->localTree);
        LOGF(stderr, " done in %g sec : %g Mptcl/sec\n", get_time()-t1, this->localTree.n/1e6/(get_time()-t1));

        this->allocateTreePropMemory(this->localTree);
        this->compute_properties(this->localTree);


        #ifdef DO_BLOCK_TIMESTEP
                devContext->startTiming(execStream->s());
                setActiveGrpsFunc(this->localTree);
                devContext->stopTiming("setActiveGrpsFunc", 10, execStream->s());
                idata.Nact_since_last_tree_rebuild = 0;
        #endif

        idata.lastBuildTime   = get_time() - t1;
        idata.totalBuildTime += idata.lastBuildTime;  
      }
      else
      {
        #ifdef DO_BLOCK_TIMESTEP
          devContext->startTiming(execStream->s());
          setActiveGrpsFunc(this->localTree);
          devContext->stopTiming("setActiveGrpsFunc", 10, execStream->s());
          idata.Nact_since_last_tree_rebuild = 0;
        #endif        
        //Don't rebuild only update the current boxes
        this->compute_properties(this->localTree);

      }//end rebuild tree

      //Approximate gravity
      t1 = get_time();
      //devContext.startTiming(gravStream->s());
      approximate_gravity(this->localTree);
//      devContext.stopTiming("Approximation", 4, gravStream->s());

      runningLETTimeSum = 0;

      if(nProcs > 1) makeLET();
    }//else if useDirectGravity

    gravStream->sync(); //Syncs the gravity stream, including any gravity computations due to LET actions

    idata.lastGravTime      = get_time() - t1;
    idata.totalGravTime    += idata.lastGravTime;
    idata.lastLETCommTime   = thisPartLETExTime;
    idata.totalLETCommTime += thisPartLETExTime;


    //Compute the total number of interactions that we executed
    tTempTime = get_time();
#if 1
   localTree.interactions.d2h();

   long long directSum = 0;
   long long apprSum = 0;

   for(int i=0; i < localTree.n; i++)
   {
     apprSum     += localTree.interactions[i].x;
     directSum   += localTree.interactions[i].y;
   }
   char buff2[512];
   sprintf(buff2, "INT Interaction at (rank= %d ) iter: %d\tdirect: %llu\tappr: %llu\tavg dir: %f\tavg appr: %f\n",
                   procId,iter, directSum ,apprSum, directSum / (float)localTree.n, apprSum / (float)localTree.n);
   devContext->writeLogEvent(buff2);
#endif
   LOGF(stderr,"Stats calculation took: %lg \n", get_time()-tTempTime);


    float ms=0, msLET=0;
#if 1 //enable when load-balancing, gets the accurate GPU time from events
    CU_SAFE_CALL(cudaEventElapsedTime(&ms, startLocalGrav, endLocalGrav));
    if(nProcs > 1)  CU_SAFE_CALL(cudaEventElapsedTime(&msLET,startRemoteGrav, endRemoteGrav));

    msLET += runningLETTimeSum;

    char buff[512];
    sprintf(buff,  "APPTIME [%d]: Iter: %d\t%g \tn: %d EventTime: %f  and %f\tSum: %f\n",
        procId, iter, idata.lastGravTime, this->localTree.n, ms, msLET, ms+msLET);
    LOGF(stderr,"%s", buff);
    devContext->writeLogEvent(buff);
#else
    ms    = 1;
    msLET = 1;
#endif

    idata.lastGPUGravTimeLocal   = ms;
    idata.lastGPUGravTimeLET     = msLET;
    idata.totalGPUGravTimeLocal += ms;
    idata.totalGPUGravTimeLET   += msLET;

    //Different options for basing the load balance on
    lastLocal = ms;
    lastTotal = ms + msLET;    

    //Corrector
    tTempTime = get_time();
    devContext->startTiming(execStream->s());
    correct(this->localTree);
    devContext->stopTiming("Correct", 8, execStream->s());
    idata.totalPredCor += get_time() - tTempTime;


    
    if(nProcs > 1)
    {
      #ifdef USE_MPI
      //Wait on all processes and time how long the waiting took
      t1 = get_time();
      devContext->startTiming(execStream->s());
      //Gather info about the load-balance, used to decide if we need to refine the domains
      MPI_Allreduce(&lastTotal, &maxExecTimePrevStep, 1, MPI_FLOAT, MPI_MAX, mpiCommWorld);
      MPI_Allreduce(&lastTotal, &avgExecTimePrevStep, 1, MPI_FLOAT, MPI_SUM, mpiCommWorld);
      avgExecTimePrevStep /= nProcs;

      devContext->stopTiming("Unbalance", 12, execStream->s());
      idata.lastWaitTime  += get_time() - t1;
      idata.totalWaitTime += idata.lastWaitTime;
      #endif
    }
    
    idata.Nact_since_last_tree_rebuild += this->localTree.n_active_particles;

    //Compute energies
    tTempTime = get_time();
    devContext->startTiming(execStream->s());
    double de = compute_energies(this->localTree);
    devContext->stopTiming("Energy", 7, execStream->s());
    idata.totalPredCor += get_time() - tTempTime;

    if(statisticsIter > 0)
    {
      if(t_current >= nextStatsTime)
      {
        nextStatsTime += statisticsIter;
        double tDens0 = get_time();
        localTree.bodies_pos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        double tDens1 = get_time();
        const DENSITY dens(mpiCommWorld, procId, nProcs, localTree.n,
                           &localTree.bodies_pos[0],
                           &localTree.bodies_vel[0],
                           &localTree.bodies_ids[0],
                           1, 2.33e9, 20, "density", t_current);

        double tDens2 = get_time();
        if(procId == 0) LOGF(stderr,"Density took: Copy: %lg Create: %lg \n", tDens1-tDens0, tDens2-tDens1);

        double tDisk1 = get_time();
        const DISKSTATS diskstats(mpiCommWorld, procId, nProcs, localTree.n,
                           &localTree.bodies_pos[0],
                           &localTree.bodies_vel[0],
                           &localTree.bodies_ids[0],
                           1, 2.33e9, "diskstats", t_current);

        double tDisk2 = get_time();
        if(procId == 0) LOGF(stderr,"Diskstats took: Create: %lg \n", tDisk2-tDisk1);
      }
    }//Statistics dumping


    if (useMPIIO)
    {
#ifdef USE_MPI
      if (mpiRenderMode) dumpDataMPI(); //To renderer process
      else               dumpData();    //To disk
#endif      
    }
    else if (snapshotIter > 0)
    {
      if((t_current >= nextSnapTime))
      {
        nextSnapTime += snapshotIter;

        while(!ioSharedData.writingFinished)
        {
          fprintf(stderr,"Waiting till previous snapshot has been written\n");
          usleep(100); //Wait till previous snapshot is written
        }

        ioSharedData.t_current  = t_current;

        //TODO JB, why do we do malloc here?
        assert(ioSharedData.nBodies == 0);
        ioSharedData.malloc(localTree.n);


        localTree.bodies_pos.d2h(localTree.n, ioSharedData.Pos);
        localTree.bodies_vel.d2h(localTree.n, ioSharedData.Vel);
        localTree.bodies_ids.d2h(localTree.n, ioSharedData.IDs);
        ioSharedData.writingFinished = false;
        if(nProcs <= 16) while (!ioSharedData.writingFinished);
      }
    }

#ifndef WAR_OF_GALAXIES
    std::cout << "hey" << std::endl;
    if (iter >= iterEnd) return true;

    if(t_current >= tEnd)
    {
      compute_energies(this->localTree);
      double totalTime = get_time() - idata.startTime;
      LOG("Finished: %f > %f \tLoop alone took: %f\n", t_current, tEnd, totalTime);
      my_dev::base_mem::printMemUsage();
      return true;
    }
    iter++; 
#endif

    return false;
}



void octree::iterate_teardown(IterationData &idata) {
  if(execStream != NULL) {
    delete execStream;
    execStream = NULL;
  }

  if(gravStream != NULL) {
    delete gravStream;
    gravStream = NULL;
  }

  if(copyStream != NULL) {
    delete copyStream;
    copyStream = NULL;
  }

  if(LETDataToHostStream != NULL)  {
    delete LETDataToHostStream;
    LETDataToHostStream = NULL;
  }
}

void octree::iterate(bool amuse) {
  IterationData idata;
  if(!amuse) iterate_setup();
  idata.startTime = get_time();


  while(true)
  {
    bool stopRun = iterate_once(idata);

    double totalTime = get_time() - idata.startTime;

    static char textBuff[16384];
    sprintf(textBuff,"TIME [%02d] TOTAL: %g\t Grav: %g (GPUgrav %g , LET Com: %g)\tBuild: %g\tDomain: %g\t Wait: %g\tdomUp: %g\tdomEx: %g\tdomWait: %g\ttPredCor: %g\n",
                      procId, totalTime, idata.totalGravTime,
                      (idata.totalGPUGravTimeLocal+idata.totalGPUGravTimeLET) / 1000,
                      idata.totalLETCommTime,
                      idata.totalBuildTime, idata.totalDomTime, idata.lastWaitTime,
                      idata.totalDomUp, idata.totalDomEx, idata.totalDomWait, idata.totalPredCor);

    if (procId == 0)
    {
      LOGF(stderr,"%s", textBuff);
      LOGF(stdout,"%s", textBuff);
    }

    devContext->writeLogEvent(textBuff);
    this->writeLogToFile();     //Write the logdata to file

    if(stopRun) break;
  } //end while
  
  if(!amuse) iterate_teardown(idata);
} //end iterate


void octree::predict(tree_structure &tree)
{
  //Functions that predicts the particles to the next timestep

  //tend is time per particle
  //tnext is reduce result

  //First we get the minimum time, which is the next integration time
  #ifdef DO_BLOCK_TIMESTEP
    getTNext.set_args(sizeof(float)*128, &tree.n, tree.bodies_time.p(), tnext.p());
    getTNext.setWork(-1, 128, NBLOCK_REDUCE);
    getTNext.execute2(execStream->s());

    //TODO
    //This will not work in block-step! Only shared- time step
    //in block step we need syncs and global communication
    if(tree.n == 0)
    {
      t_previous  =  t_current;
      t_current  += timeStep;
    }
    else
    {
      //Reduce the last parts on the host
      tnext.d2h();
      t_previous = t_current;
      t_current  = tnext[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
      {
          t_current = std::min(t_current, tnext[i]);
      }
    }
  #else
    static int temp = 0;
    t_previous =  t_current;
    if(temp > 0) t_current  += timeStep;
    else	      temp 		 = 1;
  #endif


    //Set valid list to zero, TODO should we act on this comment?

    predictParticles.set_args(0, &tree.n, &t_current, &t_previous, tree.bodies_pos.p(), tree.bodies_vel.p(),
                    tree.bodies_acc0.p(), tree.bodies_time.p(), tree.bodies_Ppos.p(), tree.bodies_Pvel.p());
    predictParticles.setWork(tree.n, 128);
    predictParticles.execute2(execStream->s());

} //End predict


void octree::setActiveGrpsFunc(tree_structure &tree)
{
  //Moved to compute_properties
}

void octree::direct_gravity(tree_structure &tree)
{
    std::vector<size_t> localWork  = {256, 1};
    std::vector<size_t> globalWork = {static_cast<size_t>(256 * ((tree.n + 255) / 256)), 1};

    directGrav.set_args(sizeof(float4)*256, tree.bodies_acc0.p(), tree.bodies_Ppos.p(),
                        tree.bodies_Ppos.p(), &tree.n, &tree.n, &(this->eps2));
    directGrav.setWork(globalWork, localWork);
    directGrav.execute2(gravStream->s());
}

void octree::approximate_gravity(tree_structure &tree)
{ 

  uint2 node_begend;
  int level_start = tree.startLevelMin;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  tree.activePartlist.zeroMemGPUAsync(gravStream->s());
  LOG("node begend: %d %d iter-> %d\n", node_begend.x, node_begend.y, iter);

  //Set the kernel parameters, many!
  approxGrav.set_args(0, &tree.n_active_groups,
                         &tree.n,
                         &(this->eps2),
                         &node_begend,
                         tree.active_group_list.p(),
                         tree.bodies_Ppos.p(),
                         tree.multipole.p(),
                         tree.bodies_acc1.p(),
                         tree.bodies_Ppos.p(),
                         tree.ngb.p(),
                         tree.activePartlist.p(),
                         tree.interactions.p(),
                         tree.boxSizeInfo.p(),
                         tree.groupSizeInfo.p(),
                         tree.boxCenterInfo.p(),
                         tree.groupCenterInfo.p(),
                         tree.bodies_Pvel.p(),
                         tree.generalBuffer1.p(),  //The buffer to store the tree walks
                         tree.bodies_h.p(),        //Per particle search radius
                         tree.bodies_dens.p());    //Per particle density (x) and nnb (y)

  approxGrav.set_texture<real4>(0,  tree.boxSizeInfo,    "texNodeSize");
  approxGrav.set_texture<real4>(1,  tree.boxCenterInfo,  "texNodeCenter");
  approxGrav.set_texture<real4>(2,  tree.multipole,      "texMultipole");
  approxGrav.set_texture<real4>(3,  tree.bodies_Ppos,   "texBody");

  approxGrav.setWork(-1, NTHREAD, nBlocksForTreeWalk);
  //approxGrav.setWork(-1, 32, 1);

  cudaEventRecord(startLocalGrav, gravStream->s());
  approxGrav.execute2(gravStream->s());  //First half
  cudaEventRecord(endLocalGrav, gravStream->s());


#if 0
	//Print density information
	tree.bodies_dens.d2h();
	tree.bodies_pos.d2h();
	tree.bodies_h.d2h();

	int nnbMin = 10e7;
	int nnbMax = -10e7;
	int nnbSum = 0;

	static bool firstIter0 = true;
	for(int i=0; i < tree.n; i++)
	{
		float r = sqrt(pow(tree.bodies_pos[i].x,2) + pow(tree.bodies_pos[i].y, 2) + pow(tree.bodies_pos[i].z,2));

		nnbMin =  std::min(nnbMin, (int)tree.bodies_dens[i].y);
		nnbMax =  std::max(nnbMax, (int)tree.bodies_dens[i].y);
		nnbSum += (int)tree.bodies_dens[i].y;
if(firstIter0 == true || iter == 40){
		fprintf(stderr, "DENS Iter: %d\t%d\t%f\t%f\t%f\tr: %f\th: %f\td: %f\tnnb: %f\t logs: %f %f  \n",
			iter,
			i, tree.bodies_pos[i].x, tree.bodies_pos[i].y, tree.bodies_pos[i].z,
			r, 
			tree.bodies_h[i],
			tree.bodies_dens[i].x, tree.bodies_dens[i].y,
			log10(tree.bodies_dens[i].x), log2(tree.bodies_dens[i].x)
			);
}

	}
		firstIter0 = false;
		fprintf(stderr,"STATD Iter: %d\tMin: %d\tMax: %d\tAvg: %f\n", iter, nnbMin, nnbMax, nnbSum / (float)tree.n);
//	exit(0);
#endif  



  //Print interaction statistics
  #if 0
  tree.body2group_list.d2h();
  tree.interactions.d2h();
    long long directSum = 0;
    long long apprSum = 0;
    long long directSum2 = 0;
    long long apprSum2 = 0;
    
    
    int maxDir = -1;
    int maxAppr = -1;

    for(int i=0; i < tree.n; i++)
    {
      apprSum     += tree.interactions[i].x;
      directSum   += tree.interactions[i].y;
      
      maxAppr = max(maxAppr,tree.interactions[i].x);
      maxDir  = max(maxDir,tree.interactions[i].y);
      
      apprSum2     += tree.interactions[i].x*tree.interactions[i].x;
      directSum2   += tree.interactions[i].y*tree.interactions[i].y;   
      
//      if(i < 35)
//      fprintf(stderr, "%d\t Direct: %d\tApprox: %d\t Group: %d \n",
//              i, tree.interactions[i].y, tree.interactions[i].x,
//              tree.body2group_list[i]);
    }
    cout << "Interaction at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;    

  #endif
  
  
  if(mpiGetNProcs() == 1) //Only do it here if there is only one process
  {
   //#ifdef DO_BLOCK_TIMESTEP
  #if 0 //Demo mode
      //Reduce the number of valid particles    
      getNActive.set_arg<int>(0,    &tree.n);
      getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
      getNActive.set_arg<cl_mem>(2, this->nactive.p());
      getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
      getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
      
      //JB Need a sync here This is required otherwise the gravity overlaps the reduction
      //and we get incorrect numbers. 
      //Note Disabled this whole function for demo!
      gravStream->sync(); 
      getNActive.execute(execStream->s());
      
      

      //Reduce the last parts on the host
      this->nactive.d2h();
      tree.n_active_particles = this->nactive[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
          tree.n_active_particles += this->nactive[i];

      LOG("Active particles: %d \n", tree.n_active_particles);
    #else
      tree.n_active_particles = tree.n;
      LOG("Active particles: %d \n", tree.n_active_particles);
    #endif
  }
}
//end approximate


void octree::approximate_gravity_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool doActiveParticles)
{
  //Start and end node of the remote tree structure
  uint2 node_begend;  
  node_begend.x =  0;
  node_begend.y =  remoteTree.remoteTreeStruct.w;
  
  //The texture offset used:
  int nodeTexOffset     = remoteTree.remoteTreeStruct.z ;
  
  //The start and end of the top nodes:
  node_begend.x = (remoteTree.remoteTreeStruct.w >> 16);
  node_begend.y = (remoteTree.remoteTreeStruct.w & 0xFFFF);  
 
  //Number of particles and number of nodes in the remote tree
  int remoteP = remoteTree.remoteTreeStruct.x;
  int remoteN = remoteTree.remoteTreeStruct.y;

  LOG("LET node begend [%d]: %d %d iter-> %d\n", procId, node_begend.x, node_begend.y, iter);

  void *multiLoc = remoteTree.fullRemoteTree.a(1*(remoteP) + 2*(remoteN+nodeTexOffset));
  void *boxSILoc = remoteTree.fullRemoteTree.a(1*(remoteP));
  void *boxCILoc = remoteTree.fullRemoteTree.a(1*(remoteP) + remoteN + nodeTexOffset);

  approxGravLET.set_args(0,
                         &tree.n_active_groups,
                         &tree.n,
                         &(this->eps2),
                         &node_begend,
                         tree.active_group_list.p(),
                         remoteTree.fullRemoteTree.p(),
                         &multiLoc,
                         tree.bodies_acc1.p(),
                         tree.bodies_Ppos.p(),
                         tree.ngb.p(),
                         tree.activePartlist.p(),
                         tree.interactions.p(),
                         &boxSILoc,
                         tree.groupSizeInfo.p(),
                         &boxCILoc,
                         tree.groupCenterInfo.p(),
                         tree.bodies_Pvel.p(),      //<- Predicted local body velocity
                         tree.generalBuffer1.p(),  //The buffer to store the tree walks
                         tree.bodies_h.p(),        //Per particle search radius
                         tree.bodies_dens.p());    //Per particle density (x) and nnb (y)
  approxGravLET.set_texture<real4>(0,  remoteTree.fullRemoteTree, "texNodeSize",  1*(remoteP), remoteN);
  approxGravLET.set_texture<real4>(1,  remoteTree.fullRemoteTree, "texNodeCenter",1*(remoteP) + (remoteN + nodeTexOffset),     remoteN);
  approxGravLET.set_texture<real4>(2,  remoteTree.fullRemoteTree, "texMultipole" ,1*(remoteP) + 2*(remoteN + nodeTexOffset), 3*remoteN);
  approxGravLET.set_texture<real4>(3,  remoteTree.fullRemoteTree, "texBody"      ,0,                                           remoteP);

  approxGravLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);
  
  if(letRunning)
  {
    //don't want to overwrite the data of previous LET tree
    gravStream->sync();

    //Add the time to the time sum for the let
    float msLET;
    CU_SAFE_CALL(cudaEventElapsedTime(&msLET,startRemoteGrav, endRemoteGrav));
    runningLETTimeSum += msLET;
  }
  
  remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
  tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Resets atomics

  CU_SAFE_CALL(cudaEventRecord(startRemoteGrav, gravStream->s()));
  approxGravLET.execute2(gravStream->s());
  CU_SAFE_CALL(cudaEventRecord(endRemoteGrav, gravStream->s()));
  letRunning = true;


 //Print interaction statistics
  #if 0
    tree.interactions.d2h();
//     tree.body2group_list.d2h();
    
    long long directSum = 0;
    long long apprSum = 0;
    
    int maxDir = -1;
    int maxAppr = -1;
    
    long long directSum2 = 0;
    long long apprSum2 = 0;
    
    
    for(int i=0; i < tree.n; i++)
    {
      apprSum     += tree.interactions[i].x;
      directSum   += tree.interactions[i].y;
      
      maxAppr = max(maxAppr,tree.interactions[i].x);
      maxDir  = max(maxDir, tree.interactions[i].y);
      
      apprSum2     += (tree.interactions[i].x*tree.interactions[i].x);
      directSum2   += (tree.interactions[i].y*tree.interactions[i].y);    
    }

    cout << "Interaction (LET) at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n  << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;
  #endif
    
  if(doActiveParticles) //Only do it here if there is only one process
  {
   //#ifdef DO_BLOCK_TIMESTEP  
  #if 0 //Demo mode
      //Reduce the number of valid particles    
      getNActive.set_arg<int>(0,    &tree.n);
      getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
      getNActive.set_arg<cl_mem>(2, this->nactive.p());
      getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
      getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
      
      //JB Need a sync here This is required otherwise the gravity overlaps the reduction
      //and we get incorrect numbers. 
      //Note Disabled this whole function for demo!
      gravStream->sync(); 
      getNActive.execute(execStream->s());
      
      

      //Reduce the last parts on the host
      this->nactive.d2h();
      tree.n_active_particles = this->nactive[0];
      for (int i = 1; i < NBLOCK_REDUCE ; i++)
          tree.n_active_particles += this->nactive[i];

      LOG("Active particles: %d \n", tree.n_active_particles);
    #else
      tree.n_active_particles = tree.n;
      LOG("Active particles: %d \n", tree.n_active_particles);
    #endif
  }    
}
//end approximate



void octree::correct(tree_structure &tree)
{ 
  //TODO this might be moved to the gravity call where we have that info anyway?
  tree.n_active_particles = tree.n;
  #ifdef DO_BLOCK_TIMESTEP
    //Reduce the number of valid particles
    gravStream->sync(); //Sync to make sure that the gravity phase is finished
//    getNActive.set_arg<int>(0,    &tree.n);
//    getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
//    getNActive.set_arg<cl_mem>(2, this->nactive.p());
//    getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
    getNActive.set_args(sizeof(int)*128, &tree.n, tree.activePartlist.p(), this->nactive.p());
    getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
    getNActive.execute2(execStream->s());

    //Reduce the last parts on the host
    this->nactive.d2h();
    tree.n_active_particles = this->nactive[0];
    for (int i = 1; i < NBLOCK_REDUCE ; i++)
        tree.n_active_particles += this->nactive[i];
  #endif
  LOG("Active particles: %d \n", tree.n_active_particles);


  my_dev::dev_mem<float2>  float2Buffer;
  my_dev::dev_mem<real4>   real4Buffer1;

  int memOffset = float2Buffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
      memOffset = real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, memOffset);


  correctParticles.set_args(0, &tree.n, &t_current, tree.bodies_time.p(), tree.activePartlist.p(),
                            tree.bodies_vel.p(), tree.bodies_acc0.p(), tree.bodies_acc1.p(),
                            tree.bodies_h.p(), tree.bodies_dens.p(), tree.bodies_pos.p(),
                            tree.bodies_Ppos.p(), tree.bodies_Pvel.p(), tree.oriParticleOrder.p(),
                            real4Buffer1.p(), float2Buffer.p());
  correctParticles.setWork(tree.n, 128);
  correctParticles.execute2(execStream->s());
 
  //Copy the shuffled items back to their original buffers
  tree.bodies_acc0.copy_devonly(real4Buffer1, tree.n);
  tree.bodies_time.copy_devonly(float2Buffer, float2Buffer.get_size());


  #ifdef DO_BLOCK_TIMESTEP
    computeDt.set_args(0, &tree.n, &t_current, &(this->eta), &(this->dt_limit), &(this->eps2),
                          tree.bodies_time.p(), tree.bodies_vel.p(), tree.ngb.p(), tree.bodies_pos.p(),
                          tree.bodies_acc0.p(), tree.activePartlist.p(), &timeStep);
    computeDt.setWork(tree.n, 128);
    computeDt.execute2(execStream->s());
  #endif
}



 //Double precision
double octree::compute_energies(tree_structure &tree)
{
  Ekin = 0.0; Epot = 0.0;

  #if 0
    double hEkin = 0.0;
    double hEpot = 0.0;

    tree.bodies_pos.d2h();
    tree.bodies_vel.d2h();
    tree.bodies_acc0.d2h();
    for (int i = 0; i < tree.n; i++) {
      float4 vel = tree.bodies_vel[i];
      hEkin += tree.bodies_pos[i].w*0.5*(vel.x*vel.x +
                                 vel.y*vel.y +
                                 vel.z*vel.z);
      hEpot += tree.bodies_pos[i].w*0.5*tree.bodies_acc0[i].w;
      //if(i < 128)
      if(i < 0)
      {
    	  LOGF(stderr,"%d\tAcc: %f %f %f %f\tPx: %f\tVx: %f\tkin: %f\tpot: %f\n", i,
    			  tree.bodies_acc0[i].x, tree.bodies_acc0[i].y, tree.bodies_acc0[i].z,
    			  tree.bodies_acc0[i].w, tree.bodies_pos[i].x, tree.bodies_vel[i].x,
    			  hEkin, hEpot);
      }
    }
    MPI_Barrier(mpiCommWorld);
    double hEtot = hEpot + hEkin;
    LOG("Energy (on host): Etot = %.10lg Ekin = %.10lg Epot = %.10lg \n", hEtot, hEkin, hEpot);
  #endif

  //float2 energy: x is kinetic energy, y is potential energy
  int blockSize = NBLOCK_REDUCE ;
  my_dev::dev_mem<double2>  energy;
  energy.cmalloc_copy(tree.generalBuffer1, blockSize, 0);
  
  computeEnergy.set_args(sizeof(double)*128*2, &tree.n, tree.bodies_pos.p(), tree.bodies_vel.p(), tree.bodies_acc0.p(), energy.p());
  computeEnergy.setWork(-1, 128, blockSize);
  computeEnergy.execute2(execStream->s());

  //Reduce the last parts on the host
  energy.d2h();
  Ekin = energy[0].x;
  Epot = energy[0].y;
  for (int i = 1; i < blockSize ; i++)
  {
      Ekin += energy[i].x;
      Epot += energy[i].y;
  }
  
  //Sum the values / energies of the system using MPI
  AllSum(Epot); AllSum(Ekin);
  
  Etot = Epot + Ekin;

  if (store_energy_flag) {
    Ekin0 = Ekin;
    Epot0 = Epot;
    Etot0 = Etot;
    Ekin1 = Ekin;
    Epot1 = Epot;
    Etot1 = Etot;
    tinit = get_time();
    store_energy_flag = false;
  }

  
  double de  = (Etot - Etot0)/Etot0;
  double dde = (Etot - Etot1)/Etot1;

  if(tree.n_active_particles == tree.n)
  {
    de_max  = std::max( de_max, std::abs( de));
    dde_max = std::max(dde_max, std::abs(dde));
  }  
  
  Ekin1 = Ekin;
  Epot1 = Epot;
  Etot1 = Etot;
  
  if(mpiGetRank() == 0)
  {
#if 1
  LOG("iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n",
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);  
  LOGF(stderr, "iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n", 
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);          
#else
  printf("iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n",
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);  
  fprintf(stderr, "iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n", 
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);          
#endif
  }

  return de;
}

