#undef NDEBUG
#include "octree.h"
#include  "postProcessModules.h"

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

inline int host_float_as_int(float val)
{
  union{float f; int i;} u; //__float_as_int
  u.f           = val;
  return u.i;
}


void octree::distributeBoundaries(bool doOnlyUpdate)
{
    devContext->pushNVTX("distributeBoundaries");

    //TODO we only need this copy once right? Some values do not change
    //when we do an intermediate update.

    localTree.boxSizeInfo.d2h  (  localTree.n_nodes, false, LETDataToHostStream->s());
    localTree.boxCenterInfo.d2h(  localTree.n_nodes, false, LETDataToHostStream->s());
    localTree.boxSmoothing.d2h (  localTree.n_nodes, false, LETDataToHostStream->s());
    localTree.multipole.d2h    (3*localTree.n_nodes, false, LETDataToHostStream->s());

    double t10 = get_time();

    //The below functions block until the MPI send and receive has been completed,
    //the GPU->Host data copies happen async
    LOGF(stderr,"distributeBoundaries updateOnly: %d \n", doOnlyUpdate);
    if(doOnlyUpdate)
    {
        this->updateCurrentInfoGrpTree();
    }
    else
    {
        this->sendCurrentInfoGrpTree();
    }

    double t20 = get_time();

    LOGF(stderr,"distributeBoundaries took: %lg  updateOnly: %d \n", t20-t10, doOnlyUpdate);
    devContext->popNVTX();
}

void octree::makeDensityLET()
{
    //Before we start making LET structures we need to wait till the data is available to the CPU
    localTree.boxSizeInfo.waitForCopyEvent();
    localTree.boxCenterInfo.waitForCopyEvent();
    localTree.boxSmoothing.waitForCopyEvent();
    localTree.multipole.waitForCopyEvent();

    std::vector<real4> topLevelsBuffer;
    std::vector<uint2> treeSizeAndOffset;
    int copyTreeUpToLevel = 0;
    //Start LET kernels
    essential_tree_exchangeV2(localTree,
                              remoteTree,
                              topLevelsBuffer,
                              treeSizeAndOffset,
                              copyTreeUpToLevel,
                              LET_METHOD_DENS);

    letRunning = false;
}

void octree::makeDerivativeLET()
{
    //Before we start making LET structures we need to wait till the data is available to the CPU
    localTree.boxSizeInfo.waitForCopyEvent();
    localTree.boxCenterInfo.waitForCopyEvent();
    localTree.boxSmoothing.waitForCopyEvent();
    localTree.multipole.waitForCopyEvent();

    //TODO this copy should be moved to the group exchange, we need velocities
    //during boundary test for derivative. So we can do the d2h copy somewhere earlier
    //and then make a function to update the group data, with the actual velocities
    //TODO 25 Aug 2017, note we cannot do it earlier if we store the gradient value in the w component
    //of the velocity.

    localTree.bodies_Pvel.d2h(localTree.n, false, LETDataToHostStream->s());
    localTree.bodies_Pvel.waitForCopyEvent();

    std::vector<real4> topLevelsBuffer;
    std::vector<uint2> treeSizeAndOffset;
    int copyTreeUpToLevel = 0;
    //Start LET kernels
    essential_tree_exchangeV2(localTree,
                              remoteTree,
                              topLevelsBuffer,
                              treeSizeAndOffset,
                              copyTreeUpToLevel,
                              LET_METHOD_DRVT);

    letRunning = false;
}

void octree::makeHydroLET()
{
    //TODO this copy should be moved to the group exchange, we need velocities
    //during boundary test for derivative. So we can do the d2h copy somewhere earlier
    //and then make a function to update the group data, with the actual velocities

//    localTree.bodies_Pvel.d2h(localTree.n, false, LETDataToHostStream->s());
//    localTree.bodies_Pvel.waitForCopyEvent();

    localTree.bodies_dens.d2h(localTree.n, false, LETDataToHostStream->s());
    localTree.bodies_dens.waitForCopyEvent();
    localTree.bodies_hydro.d2h(localTree.n, false, LETDataToHostStream->s());
    localTree.bodies_hydro.waitForCopyEvent();

    std::vector<real4> topLevelsBuffer;
    std::vector<uint2> treeSizeAndOffset;
    int copyTreeUpToLevel = 0;
    //Start LET kernels
    essential_tree_exchangeV2(localTree,
                              remoteTree,
                              topLevelsBuffer,
                              treeSizeAndOffset,
                              copyTreeUpToLevel,
                              LET_METHOD_HYDR);

    letRunning = false;
}



void octree::makeLET()
{
   //LET code test
  double t00 = get_time();


  double t10 = get_time();
  //Exchange domain grpTrees, while memory copies take place
  this->sendCurrentInfoGrpTree();

  //this->updateCurrentInfoGrpTree();

  double t20 = get_time();



  //Start copies, while grpTree info is exchanged
  localTree.boxSizeInfo.d2h  (  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxCenterInfo.d2h(  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxSmoothing.d2h (  localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.multipole.d2h    (3*localTree.n_nodes, false, LETDataToHostStream->s());
  localTree.boxSizeInfo.waitForCopyEvent();
  localTree.boxCenterInfo.waitForCopyEvent();
  localTree.boxSmoothing.waitForCopyEvent();
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
                            copyTreeUpToLevel,
                            LET_METHOD_GRAV);

  letRunning = false;
}

void countInteractions(tree_structure &tree, MPI_Comm mpiCommWorld, int procId)
{

#if 1
  //Count the number of tree-opening tests
  tree.interactions.d2h();
  tree.bodies_grad.d2h();
  long long openingTestSum  = 0;
  long long distanceTestSum = 0;
  int minOps = 10e6, maxOps = 0;
  long long int interactionUsefull = 0;
  long long int interactionTotal   = 0;
  for(int i=0; i < tree.n; i++) {
      openingTestSum     += tree.interactions[i].x; distanceTestSum     += tree.interactions[i].y;
      minOps = min(minOps, tree.interactions[i].y); maxOps = max(maxOps, tree.interactions[i].y);
      interactionTotal   += (int) tree.bodies_grad[i].z;  interactionUsefull   += (int)tree.bodies_grad[i].y;
//      fprintf(stderr,"Body: %d\t\t%d\t%d\n", i, tree.interactions[i].x, tree.interactions[i].y);
  }
  fprintf(stderr,"Number of opening angle checks: %lld [ %lld ] distance test: %lld [ Avg: %lld Min: %d Max: %d ] Interactions: avg-total: %lld avg-useful: %lld Total useful: %lld\n",
          openingTestSum,  openingTestSum  / tree.n,
          distanceTestSum, distanceTestSum / tree.n, minOps, maxOps,
          interactionTotal / tree.n, interactionUsefull / tree.n,
          interactionUsefull);

  unsigned long long tmp  = 0, tmpb = 0;
  unsigned long long tmp2 = interactionUsefull;
  unsigned long long tmp2b = interactionTotal;
  MPI_Allreduce(&tmp2,&tmp,1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,mpiCommWorld);
  MPI_Allreduce(&tmp2b,&tmpb,1, MPI_UNSIGNED_LONG_LONG, MPI_SUM,mpiCommWorld);
  if(procId == 0) fprintf(stderr,"Global useful count: %lld \t\tTotal: %lld\n", tmp, tmpb);


  MPI_Barrier(mpiCommWorld);
#endif
}


void octree::iterate_setup(IterationData &idata) {

  if(execStream == NULL)          execStream          = new my_dev::dev_stream(0);
  if(gravStream == NULL)          gravStream          = new my_dev::dev_stream(0);
  if(copyStream == NULL)          copyStream          = new my_dev::dev_stream(0);
  if(LETDataToHostStream == NULL) LETDataToHostStream = new my_dev::dev_stream(0);

  CU_SAFE_CALL(cudaEventCreate(&startLocalGrav));
  CU_SAFE_CALL(cudaEventCreate(&endLocalGrav));
  CU_SAFE_CALL(cudaEventCreate(&startRemoteGrav));
  CU_SAFE_CALL(cudaEventCreate(&endRemoteGrav));

  devContext->writeLogEvent("Start execution\n");

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
  idata.startTime = get_time();
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

    dumpData(); 

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


      setPressure.set_args(0, &this->localTree.n, this->localTree.bodies_dens.p(), this->localTree.bodies_grad.p(), this->localTree.bodies_hydro.p());
      setPressure.setWork(this->localTree.n, 128);
      setPressure.execute2(gravStream->s());

       //Approximate gravity
       t1 = get_time();
       //devContext.startTiming(gravStream->s());
       //approximate_gravity(this->localTree);
       approximate_density(this->localTree);

       double tDens = get_time() - t1;

       setPressure.set_args(0, &this->localTree.n, this->localTree.bodies_dens.p(), this->localTree.bodies_grad.p(), this->localTree.bodies_hydro.p());
       setPressure.setWork(this->localTree.n, 128);
       setPressure.execute2(gravStream->s());


       LOGF(stderr,"Start Hydro\n");

        double tStartHydro = get_time();
        approximate_hydro(this->localTree);

        double tHydro = get_time() - tStartHydro;

        idata.totalDensityTime += tDens;
        idata.totalHydroTime   += tHydro;

        mpiSync();
        countInteractions(this->localTree, mpiCommWorld, procId);
        mpiSync();

#if 0
        if(0)
        {
         this->localTree.bodies_dens_out.d2h();
         this->localTree.bodies_grad.d2h();
         this->localTree.bodies_hydro.d2h();
         this->localTree.bodies_acc1.d2h();
         this->localTree.bodies_ids.d2h();
         this->localTree.bodies_Pvel.d2h();

         for(int j=0; j < this->nProcs; j++) {
             mpiSync();

         //char fileName[128];  sprintf(fileName, "interact-%d-%d.txt", procId, nProcs); FILE* outFile = fopen(fileName, "w");
         if(j == procId)
         for(int i=0; i < this->localTree.n; i++)
         {
             ullong tempID = this->localTree.bodies_ids[i] >= 100000000 ? this->localTree.bodies_ids[i]-100000000 : this->localTree.bodies_ids[i];
             if(tempID < 10 || std::isinf(this->localTree.bodies_acc1[i].x))
                 LOGF(stderr,"Rho out: %d %lld || Pos: %f %f %f %lg\t || Vel: %f %f %f gradh: %f || Dens: %lg %f\t|| Drvt: %f %f %f %f\t|| Hydro: %f %f %f %f || Acc: %f %f %f %f\n",
                     i,
                     tempID, //this->localTree.bodies_ids[i],
                     this->localTree.bodies_Ppos[i].x,
                     this->localTree.bodies_Ppos[i].y,
                     this->localTree.bodies_Ppos[i].z,
                     this->localTree.bodies_Ppos[i].w,
                     this->localTree.bodies_Pvel[i].x,
                     this->localTree.bodies_Pvel[i].y,
                     this->localTree.bodies_Pvel[i].z,
                     this->localTree.bodies_Pvel[i].w,
                     this->localTree.bodies_dens_out[i].x,
                     this->localTree.bodies_dens_out[i].y,
                     this->localTree.bodies_grad[i].w,
                     this->localTree.bodies_grad[i].x,
                     this->localTree.bodies_grad[i].y,
                     this->localTree.bodies_grad[i].z,
                     this->localTree.bodies_hydro[i].x,
                     this->localTree.bodies_hydro[i].y,
                     this->localTree.bodies_hydro[i].z,
                     this->localTree.bodies_hydro[i].w,
                     this->localTree.bodies_acc1[i].x,
                     this->localTree.bodies_acc1[i].y,
                     this->localTree.bodies_acc1[i].z,
                     this->localTree.bodies_acc1[i].w);
         }
         }//for j
//         if(t_current > 0)
         {
             mpiSync();
//             exit(0);
         }
        }
#endif

//      devContext.stopTiming("Approximation", 4, gravStream->s());

      runningLETTimeSum = 0;

     //TODO uncomment if(nProcs > 1) makeLET();
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
#if 0 //enable when load-balancing, gets the accurate GPU time from events
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
#if 0
        Disabled this part as it messes up our stack-size which makes debugging messy
        double tDens1 = get_time();
        const DENSITY dens(mpiCommWorld, procId, nProcs, localTree.n,
                           &localTree.bodies_pos[0],
                           &localTree.bodies_vel[0],
                           &localTree.bodies_ids[0],
                           1, 2.33e9, 20, "density", t_current);

        double tDens2 = get_time();
        if(procId == 0) LOGF(stderr,"Density took: Copy: %lg Create: %lg \n", tDens1-tDens0, tDens2-tDens1);
#endif

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

        localTree.bodies_dens. d2h(localTree.n, ioSharedData.Den);
        localTree.bodies_hydro.d2h(localTree.n, ioSharedData.Hyd);
        localTree.bodies_grad. d2h(localTree.n, ioSharedData.Drv);


        ioSharedData.writingFinished = false;
        if(nProcs <= 16) while (!ioSharedData.writingFinished);
      }
    }


    if (iter >= iterEnd) return true;

    if(t_current >= tEnd)
    {
      compute_energies(this->localTree);
      double totalTime = get_time() - idata.startTime;
      LOG("Finished: %f > %f \tLoop alone took: %f\n", t_current, tEnd, totalTime);
      my_dev::base_mem::printMemUsage();


//      std::ofstream out("sphDUMP.txt", std::ofstream::out);
//      for(int i=0; i < this->localTree.n; i++)
//      {
//          char buff[2048];
//          sprintf(buff,"%d %lld || Pos: %f %f %f %lg\t Vel: %f %f %f || Dens: %lg %f\t|| Drvt: %f %f %f %f\t|| Hydro: %f %f %f %f || Acc: %f %f %f %f\n",
//              i,
//              this->localTree.bodies_ids[i],
//              this->localTree.bodies_Ppos[i].x,
//              this->localTree.bodies_Ppos[i].y,
//              this->localTree.bodies_Ppos[i].z,
//              this->localTree.bodies_Ppos[i].w,
//              this->localTree.bodies_Pvel[i].x,
//              this->localTree.bodies_Pvel[i].y,
//              this->localTree.bodies_Pvel[i].z,
//              this->localTree.bodies_dens_out[i].x,
//              this->localTree.bodies_dens_out[i].y,
//              this->localTree.bodies_grad[i].w,
//              this->localTree.bodies_grad[i].x,
//              this->localTree.bodies_grad[i].y,
//              this->localTree.bodies_grad[i].z,
//              this->localTree.bodies_hydro[i].x,
//              this->localTree.bodies_hydro[i].y,
//              this->localTree.bodies_hydro[i].z,
//              this->localTree.bodies_hydro[i].w,
//              this->localTree.bodies_acc1[i].x,
//              this->localTree.bodies_acc1[i].y,
//              this->localTree.bodies_acc1[i].z,
//              this->localTree.bodies_acc1[i].w);
//          out << buff;
//      }

      char fname[512];
      sprintf(fname, "sphOut-%f-%d.txt", t_current, procId);
      std::ofstream out(fname, std::ofstream::out);
      out << "# id\tx\ty\tz\tm\tvx\tvy\tvz\tdensity\th\tu\tP\n";
      for(int i=0; i < this->localTree.n; i++)
      {
            ullong tempID = this->localTree.bodies_ids[i] >= 100000000 ? this->localTree.bodies_ids[i]-100000000 : this->localTree.bodies_ids[i];
            char buff[2048];
            sprintf(buff,"%lld\t%f\t%f\t%f\t%lg\t%f\t%f\t%f\t%lg\t%f\t%f\t%f\n",
                tempID,
                //this->localTree.bodies_ids[i],
                this->localTree.bodies_Ppos[i].x,
                this->localTree.bodies_Ppos[i].y,
                this->localTree.bodies_Ppos[i].z,
                this->localTree.bodies_Ppos[i].w,
                this->localTree.bodies_Pvel[i].x,
                this->localTree.bodies_Pvel[i].y,
                this->localTree.bodies_Pvel[i].z,
                this->localTree.bodies_dens_out[i].x,
                this->localTree.bodies_dens_out[i].y,
                this->localTree.bodies_hydro[i].z,
                this->localTree.bodies_hydro[i].x);
            out << buff;
      }
      out.close();




      return true;
    }
    iter++; 

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

void octree::iterate() {
  IterationData idata;
  iterate_setup(idata);


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
    devContext->writeLogEvent(textBuff);
    sprintf(textBuff,"TIME [%02d] TOTAL-SPH: %g\t Hydro: %g (Density %g , Hydro: %g)\n",
                      procId, totalTime,
                      idata.totalDensityTime + idata.totalHydroTime,
                      idata.totalDensityTime,
                      idata.totalHydroTime);
    devContext->writeLogEvent(textBuff);

    if (procId == 0)
    {
//      LOGF(stderr,"%s", textBuff);
//      LOGF(stdout,"%s", textBuff);
    }


    this->writeLogToFile();     //Write the logdata to file

    if(stopRun) break;
  } //end while
  
  iterate_teardown(idata);
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

    //Enforce global time-step
    t_current = this->AllMin(t_current);


    //Set valid list to zero  <- TODO should we act on this comment?

    predictParticles.set_args(0, &tree.n, &t_current, &t_previous, tree.bodies_pos.p(), tree.bodies_vel.p(),
                    tree.bodies_acc0.p(), tree.bodies_time.p(), tree.bodies_Ppos.p(), tree.bodies_Pvel.p(),
                    tree.bodies_hydro.p(), tree.bodies_ids.p());
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


void octree::approximate_density    (tree_structure &tree)
{
    uint2 node_begend = {tree.level_list[tree.startLevelMin].x,
                         tree.level_list[tree.startLevelMin].y};

    LOG("node begend: %d %d iter-> %d\n", node_begend.x, node_begend.y, iter);

    bool isFinalLaunch = (nProcs == 1);

    SPHDensity.set_args(0,
                           &tree.n_active_groups,
                           &tree.n,
                           &(this->eps2),
                           &node_begend,
                           &isFinalLaunch,
                           tree.active_group_list.p(),
                           //i particle properties
                           &tree.group_body,
                           tree.activePartlist.p(),
                           tree.interactions.p(),
                           tree.boxSizeInfo.p(),
                           tree.groupSizeInfo.p(),
                           tree.boxCenterInfo.p(),
                           tree.groupCenterInfo.p(),
                           tree.multipole.p(),
                           tree.generalBuffer1.p(),  //The buffer to store the tree walks
                           //j particle properties
                           &tree.group_body,    //Note i and j particles are the same in local call
                           //Result buffers
                           tree.bodies_acc1.p(),
                           tree.bodies_dens_out.p(),
                           tree.bodies_hydro_out.p(),
                           tree.bodies_grad.p(),
                           tree.bodies_ids.p());
    SPHDensity.set_texture<real4>(0,  tree.boxSizeInfo,    "texNodeSize");
    SPHDensity.set_texture<real4>(1,  tree.boxCenterInfo,  "texNodeCenter");
    SPHDensity.set_texture<real4>(2,  tree.multipole,      "texMultipole");
    SPHDensity.set_texture<real4>(3,  tree.bodies_Ppos,    "texBody");

    SPHDensity.setWork(-1, NTHREAD, nBlocksForTreeWalk);

    for(int i=0; i < 3; i++)
    {
        tree.interactions.zeroMemGPUAsync(gravStream->s());
        tree.activePartlist.zeroMemGPUAsync(gravStream->s());
        tree.bodies_grad.zeroMemGPUAsync(gravStream->s());
        tree.bodies_dens_out.zeroMemGPUAsync(gravStream->s());
        tree.bodies_acc1.zeroMemGPUAsync(gravStream->s()); //Used for computing gradh


        cudaEventRecord(startLocalGrav, gravStream->s());
        double tStart = get_time();
        SPHDensity.execute2(gravStream->s());  //First iteration
        cudaEventRecord(endLocalGrav, gravStream->s());

        cudaDeviceSynchronize();
//        countInteractions(tree, mpiCommWorld, procId);
    //    double tEnd = get_time();
    //
    //    float ms;
    //    CU_SAFE_CALL(cudaEventElapsedTime(&ms, startLocalGrav, endLocalGrav));
    //    fprintf(stderr,"SPH GPU step took: %f ms\t%lg sec\n", ms,  tEnd-tStart);

        if(nProcs > 1)
        {
            distributeBoundaries(false); //Always full update for now
//            makeDensityLET();
            makeDerivativeLET();
        }

        //Update the current density and smoothing radius based on that density
        tree.bodies_dens.copy_devonly_async(tree.bodies_dens_out, tree.n, 0,gravStream->s());

        //TODO should we update the tree? No as we only update and use particle search radii.
        //Actually yes, since we need proper tree-node smoothing values after the iterations
        compute_properties(tree);

        cudaDeviceSynchronize();
        //wait on the LET to finish before we start the interaction computations
        gravStream->sync();

        countInteractions(tree, mpiCommWorld, procId);
    } //For i
}

void octree::approximate_density_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool isFinalLaunch)
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

      void *multiLoc = remoteTree.fullRemoteTree.a(2*(remoteP) + 2*(remoteN+nodeTexOffset));
      void *boxSILoc = remoteTree.fullRemoteTree.a(2*(remoteP));
      void *boxCILoc = remoteTree.fullRemoteTree.a(2*(remoteP) + remoteN + nodeTexOffset);
      void *velLoc   = remoteTree.fullRemoteTree.a(1*(remoteP));

      bodyProps j_body;
      j_body.body_pos   = (real4*)(remoteTree.fullRemoteTree.d());
      j_body.body_vel   = (real4*)velLoc;
      j_body.body_dens  = (float2*)tree.bodies_dens.d();
      j_body.body_grad  = (real4*)tree.bodies_grad.d();
      j_body.body_hydro = (real4*)tree.bodies_hydro.d();


      SPHDensityLET.set_args(0, &tree.n_active_groups,
                             &tree.n,
                             &(this->eps2),
                             &node_begend,
                             &isFinalLaunch,
                             tree.active_group_list.p(),
                             //i particle properties
                             &tree.group_body,
                             tree.activePartlist.p(),
                             tree.interactions.p(),
                             &boxSILoc,
                             tree.groupSizeInfo.p(),
                             &boxCILoc,
                             tree.groupCenterInfo.p(),
                             &multiLoc,
                             tree.generalBuffer1.p(),  //The buffer to store the tree walks
                             //j particle properties
                             &j_body,
                             //Result buffers
                             tree.bodies_acc1.p(),
                             tree.bodies_dens_out.p(),
                             tree.bodies_hydro_out.p(),
                             tree.bodies_grad.p(),
                             tree.bodies_ids.p());
      SPHDensityLET.set_texture<real4>(0,  remoteTree.fullRemoteTree, "texNodeSize",  1*(remoteP), remoteN);
      SPHDensityLET.set_texture<real4>(1,  remoteTree.fullRemoteTree, "texNodeCenter",1*(remoteP) + (remoteN + nodeTexOffset),     remoteN);
      SPHDensityLET.set_texture<real4>(2,  remoteTree.fullRemoteTree, "texMultipole", 1*(remoteP) + 2*(remoteN + nodeTexOffset), 3*remoteN);
      SPHDensityLET.set_texture<real4>(3,  remoteTree.fullRemoteTree, "texBody"      ,0,                                           remoteP);

      SPHDensityLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);
    //  SPHDensity.setWork(-1, 32, 1);

      remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
      tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Resets atomics


      CU_SAFE_CALL(cudaEventRecord(startRemoteGrav, gravStream->s()));
      SPHDensityLET.execute2(gravStream->s());
      CU_SAFE_CALL(cudaEventRecord(endRemoteGrav, gravStream->s()));
      letRunning = true;

      return;
}


void octree::approximate_hydro(tree_structure &tree)
{
    uint2 node_begend;
    int level_start = tree.startLevelMin;
    node_begend.x   = tree.level_list[level_start].x;
    node_begend.y   = tree.level_list[level_start].y;

    bool isFinalLaunch = (nProcs == 1);

    SPHHydro.set_args(0, &tree.n_active_groups,
             &tree.n,
             &(this->eps2),
             &node_begend,
             &isFinalLaunch,
             tree.active_group_list.p(),
             //i particle properties
             &tree.group_body,
             tree.activePartlist.p(),
             tree.interactions.p(),
             tree.boxSizeInfo.p(),
             tree.groupSizeInfo.p(),
             tree.boxCenterInfo.p(),
             tree.groupCenterInfo.p(),
             tree.multipole.p(),
             tree.generalBuffer1.p(),  //The buffer to store the tree walks
             //j particle properties
             &tree.group_body,    //Note i and j particles are the same in local call
             //Result buffers
             tree.bodies_acc1.p(),
             tree.bodies_dens_out.p(),
             tree.bodies_hydro_out.p(),
             tree.bodies_grad.p(),
             tree.bodies_ids.p());

     SPHHydro.set_texture<real4>(0,  tree.boxSizeInfo,    "texNodeSize");
     SPHHydro.set_texture<real4>(1,  tree.boxCenterInfo,  "texNodeCenter");
     SPHHydro.set_texture<real4>(2,  tree.multipole,      "texMultipole");
     SPHHydro.set_texture<real4>(3,  tree.bodies_Ppos,    "texBody");
     SPHHydro.setWork(-1, NTHREAD, nBlocksForTreeWalk);
    //  SPHDerivative.setWork(-1, 32, 1);

     //Note, I abuse the compute_prop function/opening angle criteria, so you can not use
     //this for computing gravity at this point.
     compute_properties(tree); //TODO make this smarter as in only compute cells not groups. And switch between SPH and Gravity


     //Reset bodies_grad, since we reuse/abuse it to store the dt
     tree.bodies_grad.zeroMemGPUAsync(gravStream->s());
     tree.bodies_acc1.zeroMemGPUAsync(gravStream->s()); //Reset as we used it before to store gradh


     tree.interactions.zeroMemGPUAsync(gravStream->s());
     tree.activePartlist.zeroMemGPUAsync(gravStream->s());
     SPHHydro.execute2(gravStream->s());  //Hydro force

     cudaDeviceSynchronize();

     gravStream->sync();
     countInteractions(tree, mpiCommWorld, procId);


     if(nProcs > 1)
     {
         distributeBoundaries(false); //Always full update for now
         makeHydroLET();
     }
}

void octree::approximate_hydro_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool isFinalLaunch)
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

      LOG("SPHHydro LET node begend [%d]: %d %d iter-> %d\n", procId, node_begend.x, node_begend.y, iter);

      void *multiLoc = remoteTree.fullRemoteTree.a(4*(remoteP) + 2*(remoteN+nodeTexOffset));
      void *boxSILoc = remoteTree.fullRemoteTree.a(4*(remoteP));
      void *boxCILoc = remoteTree.fullRemoteTree.a(4*(remoteP) + remoteN + nodeTexOffset);

      bodyProps j_body;
      j_body.body_pos   = (real4*)( remoteTree.fullRemoteTree.d());
      j_body.body_vel   = (real4*)  remoteTree.fullRemoteTree.a(1*(remoteP));
      j_body.body_dens  = (float2*) remoteTree.fullRemoteTree.a(2*(remoteP));      //Per particle density (x) and nnb (y)
      j_body.body_grad  = (real4*)tree.bodies_grad.d();
      j_body.body_hydro = (real4*)  remoteTree.fullRemoteTree.a(3*(remoteP));


      SPHHydro.set_args(0, &tree.n_active_groups,
                             &tree.n,
                             &(this->eps2),
                             &node_begend,
                             &isFinalLaunch,
                             tree.active_group_list.p(),
                             //i particle properties
                             &tree.group_body,
                             tree.activePartlist.p(),
                             tree.interactions.p(),
                             &boxSILoc,
                             tree.groupSizeInfo.p(),
                             &boxCILoc,
                             tree.groupCenterInfo.p(),
                             &multiLoc,
                             tree.generalBuffer1.p(),  //The buffer to store the tree walks
                             //j particle properties
                             &j_body,
                             //Result buffers
                             tree.bodies_acc1.p(),
                             tree.bodies_dens_out.p(),
                             tree.bodies_hydro_out.p(),
                             tree.bodies_grad.p(),
                             tree.bodies_ids.p());
      SPHHydro.set_texture<real4>(0,  remoteTree.fullRemoteTree, "texNodeSize",  3*(remoteP), remoteN);
      SPHHydro.set_texture<real4>(1,  remoteTree.fullRemoteTree, "texNodeCenter",3*(remoteP) + (remoteN + nodeTexOffset),     remoteN);
      SPHHydro.set_texture<real4>(2,  remoteTree.fullRemoteTree, "texMultipole", 3*(remoteP) + 2*(remoteN + nodeTexOffset), 3*remoteN);
      SPHHydro.set_texture<real4>(3,  remoteTree.fullRemoteTree, "texBody"      ,0,                                           remoteP);

      SPHHydro.setWork(-1, NTHREAD, nBlocksForTreeWalk);

      remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
      tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Resets atomics

      CU_SAFE_CALL(cudaEventRecord(startRemoteGrav, gravStream->s()));
//      tree.bodies_acc1.zeroMemGPUAsync(gravStream->s()); //Reset as we used it before to store gradh
      SPHHydro.execute2(gravStream->s());
      CU_SAFE_CALL(cudaEventRecord(endRemoteGrav, gravStream->s()));
      letRunning = true;

      return;
}

void octree::approximate_gravity(tree_structure &tree)
{ 

  uint2 node_begend;
  int level_start = tree.startLevelMin;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  LOG("node begend: %d %d iter-> %d\n", node_begend.x, node_begend.y, iter);

  tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Set our atomic counter to zero :)
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
  cudaDeviceSynchronize();
  tEnd = get_time();
  CU_SAFE_CALL(cudaEventElapsedTime(&ms, startLocalGrav, endLocalGrav));
  fprintf(stderr,"Gravity step took: %f ms\t%lg sec\n", ms,  tEnd-tStart);

  //Count the number of tree-opening tests
  tree.interactions.d2h();
  openingTestSum  = 0;
  distanceTestSum = 0;

  for(int i=0; i < tree.n; i++) { openingTestSum     += tree.interactions[i].x; distanceTestSum     += tree.interactions[i].y;   }
  fprintf(stderr,"Number of opening angle checks: %lld [ %lld ] direct ops: %lld [ %lld ] \n",
          openingTestSum,  openingTestSum  / tree.n,
          distanceTestSum, distanceTestSum / tree.n);
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
    getNActive.set_args(sizeof(int)*128, &tree.n, tree.activePartlist.p(), this->nactive.p());
    getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
    getNActive.execute2(execStream->s());

    //Reduce the last parts on the host
    this->nactive.d2h();
    tree.n_active_particles = this->nactive[0];
    for (int i = 1; i < NBLOCK_REDUCE ; i++)
        tree.n_active_particles += this->nactive[i];
  #endif
  LOG("Active particles: %d Total: %d\n", tree.n_active_particles, tree.n);


  my_dev::dev_mem<float2>  float2Buffer;
  my_dev::dev_mem<real4>   real4Buffer1;

  int memOffset = float2Buffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
      memOffset = real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, memOffset);


  correctParticles.set_args(0, &tree.n, &t_current, tree.bodies_time.p(), tree.activePartlist.p(),
                            tree.bodies_vel.p(), tree.bodies_acc0.p(), tree.bodies_acc1.p(),
                            tree.bodies_h.p(), tree.bodies_dens.p(), tree.bodies_pos.p(),
                            tree.bodies_Ppos.p(), tree.bodies_Pvel.p(), tree.oriParticleOrder.p(),
                            real4Buffer1.p(), float2Buffer.p(),
                            tree.bodies_hydro.p(), tree.bodies_ids.p());
  correctParticles.setWork(tree.n, 128);
  correctParticles.execute2(execStream->s());

  //Copy the shuffled items back to their original buffers
  tree.bodies_acc0.copy_devonly(real4Buffer1, tree.n);
  tree.bodies_time.copy_devonly(float2Buffer, float2Buffer.get_size());


  #ifdef DO_BLOCK_TIMESTEP
    computeDt.set_args(0, &tree.n, &t_current, &(this->eta), &(this->dt_limit), &(this->eps2),
                          tree.bodies_time.p(), tree.bodies_vel.p(), tree.ngb.p(), tree.bodies_pos.p(),
                          tree.bodies_acc0.p(), tree.activePartlist.p(), &timeStep, tree.bodies_grad.p());
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
      //if(i < 0)
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
#if 0
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

  if(std::isnan(Etot)){
      LOGF(stderr,"NaN detected, exit\n");
      exit(0);
  }
  mpiSync(); //TODO(jbedorf) remove this and the lines above if we solved NaN problems

  return de;
}


void octree::approximate_derivative  (tree_structure &tree)
{
    uint2 node_begend = {tree.level_list[tree.startLevelMin].x,
                         tree.level_list[tree.startLevelMin].y};

    bool isFinalLaunch = (nProcs == 1);

    SPHDerivative.set_args(0,   &tree.n_active_groups,
                                  &tree.n,
                                  &(this->eps2),
                                  &node_begend,
                                  &isFinalLaunch,
                                  tree.active_group_list.p(),
                                  //i particle properties
                                  tree.bodies_Ppos.p(),
                                  tree.bodies_Pvel.p(),
                                  tree.bodies_dens.p(),
                                  tree.bodies_grad.p(),
                                  tree.bodies_hydro.p(),
                                  tree.activePartlist.p(),
                                  tree.interactions.p(),
                                  tree.boxSizeInfo.p(),
                                  tree.groupSizeInfo.p(),
                                  tree.boxCenterInfo.p(),
                                  tree.groupCenterInfo.p(),
                                  tree.multipole.p(),
                                  tree.generalBuffer1.p(),  //The buffer to store the tree walks
                                  //j particle properties
                                  tree.bodies_Ppos.p(),
                                  tree.bodies_Pvel.p(),
                                  tree.bodies_dens.p(),     //Per particle density (x) and nnb (y)
                                  tree.bodies_grad.p(),
                                  tree.bodies_hydro.p(),
                                  //Result buffers
                                  tree.bodies_acc1.p(),
                                  tree.bodies_dens_out.p(),
                                  tree.bodies_hydro_out.p(),
                                  tree.bodies_grad.p(),
                                  tree.bodies_ids.p());

      SPHDerivative.set_texture<real4>(0,  tree.boxSizeInfo,    "texNodeSize");
      SPHDerivative.set_texture<real4>(1,  tree.boxCenterInfo,  "texNodeCenter");
      SPHDerivative.set_texture<real4>(2,  tree.multipole,      "texMultipole");
      SPHDerivative.set_texture<real4>(3,  tree.bodies_Ppos,    "texBody");
      SPHDerivative.setWork(-1, NTHREAD, nBlocksForTreeWalk);
     //  SPHDerivative.setWork(-1, 32, 1);
      //      size_t sz = 1048576 * 25;
      //      cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

      tree.bodies_grad.zeroMemGPUAsync(gravStream->s());
      tree.activePartlist.zeroMemGPUAsync(gravStream->s());
      SPHDerivative.execute2(gravStream->s());  //Derivative

      cudaDeviceSynchronize();

      tree.bodies_dens_out.d2h();
      tree.bodies_grad.d2h();
      tree.bodies_hydro.d2h();
      tree.bodies_acc1.d2h();
      tree.bodies_ids.d2h();


      if(nProcs > 1)
      {
         distributeBoundaries(false); //Always full update for now
         makeDerivativeLET();
      }
      else
      {
//         setPressure.set_args(0, &tree.n, tree.bodies_dens.p(), tree.bodies_grad.p(), tree.bodies_hydro.p());
//         setPressure.setWork(tree.n, 128);
//         setPressure.execute2(gravStream->s());
      }

}

void octree::approximate_derivative_let(tree_structure &tree, tree_structure &remoteTree, int bufferSize, bool isFinalLaunch)
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

      LOG("SPHDerivative LET node begend [%d]: %d %d iter-> %d\n", procId, node_begend.x, node_begend.y, iter);

      void *multiLoc = remoteTree.fullRemoteTree.a(2*(remoteP) + 2*(remoteN+nodeTexOffset));
      void *boxSILoc = remoteTree.fullRemoteTree.a(2*(remoteP));
      void *boxCILoc = remoteTree.fullRemoteTree.a(2*(remoteP) + remoteN + nodeTexOffset);
      void *velLoc   = remoteTree.fullRemoteTree.a(1*(remoteP));

      SPHDerivativeLET.set_args(0, &tree.n_active_groups,
                             &tree.n,
                             &(this->eps2),
                             &node_begend,
                             &isFinalLaunch,
                             tree.active_group_list.p(),
                             //i particle properties
                             tree.bodies_Ppos.p(),
                             tree.bodies_Pvel.p(),
                             tree.bodies_dens.p(),
                             tree.bodies_grad.p(),
                             tree.bodies_hydro.p(),
                             tree.activePartlist.p(),
                             tree.interactions.p(),
                             &boxSILoc,
                             tree.groupSizeInfo.p(),
                             &boxCILoc,
                             tree.groupCenterInfo.p(),
                             &multiLoc,
                             tree.generalBuffer1.p(),  //The buffer to store the tree walks
                             //j particle properties
                             remoteTree.fullRemoteTree.p(),
                             &velLoc, //tree.bodies_Pvel.p(),
                             tree.bodies_dens.p(),     //Per particle density (x) and nnb (y)
                             tree.bodies_grad.p(),
                             tree.bodies_hydro.p(),
                             //Result buffers
                             tree.bodies_acc1.p(),
                             tree.bodies_dens_out.p(),
                             tree.bodies_hydro_out.p(),
                             tree.bodies_grad.p(),
                             tree.bodies_ids.p());
      SPHDerivativeLET.set_texture<real4>(0,  remoteTree.fullRemoteTree, "texNodeSize",  1*(remoteP), remoteN);
      SPHDerivativeLET.set_texture<real4>(1,  remoteTree.fullRemoteTree, "texNodeCenter",1*(remoteP) + (remoteN + nodeTexOffset),     remoteN);
      SPHDerivativeLET.set_texture<real4>(2,  remoteTree.fullRemoteTree, "texMultipole", 1*(remoteP) + 2*(remoteN + nodeTexOffset), 3*remoteN);
      SPHDerivativeLET.set_texture<real4>(3,  remoteTree.fullRemoteTree, "texBody"      ,0,                                           remoteP);

      SPHDerivativeLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);

      remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
      tree.activePartlist.zeroMemGPUAsync(gravStream->s()); //Resets atomics

      CU_SAFE_CALL(cudaEventRecord(startRemoteGrav, gravStream->s()));
      SPHDerivativeLET.execute2(gravStream->s());
      CU_SAFE_CALL(cudaEventRecord(endRemoteGrav, gravStream->s()));
      letRunning = true;

      if(isFinalLaunch)
      {
          setPressure.set_args(0, &tree.n, tree.bodies_dens.p(), tree.bodies_grad.p(), tree.bodies_hydro.p());
          setPressure.setWork(tree.n, 128);
          setPressure.execute2(gravStream->s());
      }

      return;
}

