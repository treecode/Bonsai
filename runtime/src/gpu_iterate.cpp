#include "octree.h"

#include <iostream>
#include <algorithm>
#include <iomanip>
using namespace std;

static double de_max = 0;
static double dde_max = 0;  



void octree::makeLET()
{
   //LET code test
  double tTest = get_time();

  my_dev::dev_stream memCpyStream;
  
  localTree.bodies_Ppos.d2h(false, memCpyStream.s());
  localTree.bodies_Pvel.d2h(false, memCpyStream.s());
  localTree.multipole.d2h(false, memCpyStream.s());   
  localTree.boxSizeInfo.d2h(false, memCpyStream.s());
  localTree.boxCenterInfo.d2h(false, memCpyStream.s());
  
  //Exchange domain boundaries, while memory copies take place

  rMinLocalTreeGroups.w = this->maxLocalEps;
  sendCurrentRadiusInfo(rMinLocalTreeGroups,rMaxLocalTreeGroups);  

  memCpyStream.sync();  //Sync otherwise we are not certain the required data is on the host
    
  //Exchange particles and start LET kernels
  vector<real4> LETParticles;
  essential_tree_exchange(LETParticles, localTree, remoteTree);
  fprintf(stderr, "LET Exchange took (%d): %g \n", mpiGetRank(), get_time() - tTest);
  
  letRunning = false;
  execStream->sync();  //Sync LET execution

}

// returns true if this iteration is the last (t_current >= t_end), false otherwise
bool octree::iterate_once(IterationData &idata) {
    double t1 = 0;

    LOG("At the start of iterate:\n");

    //predict localtree
    devContext.startTiming();
    predict(this->localTree);
    devContext.stopTiming("Predict", 9);
    
    bool needDomainUpdate = true;
    
   //Redistribute the particles
    if(1)
    {      
      if(nProcs > 1)
      { 
       if(iter % rebuild_tree_rate == 0) 
//        if(0)
//        if(1)
        {     
          //If we do a redistribution we _always_ have to do 
          //an update of the particle domain, otherwise the boxes 
          //do not match and we get errors of particles outside
          //domains
          t1 = get_time();
          
          devContext.startTiming();
          gpu_updateDomainDistribution(idata.lastGravTime);          
          devContext.stopTiming("DomainUpdate", 6);
          
          devContext.startTiming();
          gpuRedistributeParticles();
          devContext.stopTiming("Exchange", 6);
          
          needDomainUpdate = false;
          
          idata.lastDomTime   = get_time() - t1;
          idata.totalDomTime += idata.lastDomTime;          
        }
        else
        {
          //Only send new box sizes, incase we do not exchange particles
          //but continue with the current tree_structure
          gpu_updateDomainOnly();
          
          needDomainUpdate = false;
        } //if (iter % X )
//         else
//         {
//           //Only exchange, do not update domain decomposition
//         }
      } //if nProcs > 1
    }//if (0)        
    
    
    //Build the tree using the predicted positions
   // bool rebuild_tree = Nact_since_last_tree_rebuild > 4*this->localTree.n;   
    bool rebuild_tree = true;

    rebuild_tree = ((iter % rebuild_tree_rate) == 0);    
    if(rebuild_tree)
    {
      t1 = get_time();
      //Rebuild the tree
      this->sort_bodies(this->localTree, needDomainUpdate);

      devContext.startTiming();
      this->build(this->localTree);
      devContext.stopTiming("Tree-construction", 2);

      devContext.startTiming();
      this->allocateTreePropMemory(this->localTree);
      devContext.stopTiming("Memory", 11);      

      devContext.startTiming();
      this->compute_properties(this->localTree);
      devContext.stopTiming("Compute-properties", 3);

      devContext.startTiming();
      setActiveGrpsFunc(this->localTree);
      devContext.stopTiming("setActiveGrpsFunc", 10);      
      idata.Nact_since_last_tree_rebuild = 0;
      
      idata.lastBuildTime   = get_time() - t1;
      idata.totalBuildTime += idata.lastBuildTime;  
    }
    else
    {
      //Dont rebuild only update the current boxes
      devContext.startTiming();
      this->compute_properties(this->localTree);
      devContext.stopTiming("Compute-properties", 3);
    }//end rebuild tree

    //Approximate gravity
    t1 = get_time();
//     devContext.startTiming();
    approximate_gravity(this->localTree);
//     devContext.stopTiming("Approximation", 4);
    
    
    if(nProcs > 1)  makeLET();
    
    execStream->sync();
    
    idata.lastGravTime   = get_time() - t1;
//     totalGravTime += lastGravTime;
    idata.totalGravTime += idata.lastGravTime - thisPartLETExTime;
//     lastGravTime -= thisPartLETExTime;
    
    LOGF(stderr, "APPTIME [%d]: Iter: %d\t%g \n", procId, iter, idata.lastGravTime);
    
    //Corrector
    devContext.startTiming();
    correct(this->localTree);
    devContext.stopTiming("Correct", 8);
    
    if(nProcs > 1)
    {
      t1 = get_time();
      devContext.startTiming();
      mpiSync();
      devContext.stopTiming("Unbalance", 12);
      idata.lastWaitTime  += get_time() - t1;
      idata.totalWaitTime += idata.lastWaitTime;
    }
    
    idata.Nact_since_last_tree_rebuild += this->localTree.n_active_particles;

    //Compute energies
    devContext.startTiming();
    double de = compute_energies(this->localTree); de=de;
    devContext.stopTiming("Energy", 7);

    if(snapshotIter > 0)
    {
      int time = (int)t_current;
      if((time >= nextSnapTime))
      {
        nextSnapTime += snapshotIter;
        string fileName; fileName.resize(256);
        sprintf(&fileName[0], "%s_%06d", snapshotFile.c_str(), time + snapShotAdd);

        localTree.bodies_pos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        write_dumbp_snapshot_parallel(&localTree.bodies_pos[0], &localTree.bodies_vel[0],
                                      &localTree.bodies_ids[0], localTree.n, fileName.c_str()) ;
      }
    }

    if(t_current >= tEnd)
    {
      compute_energies(this->localTree);
      double totalTime = get_time() - idata.startTime;
      cout << " Finished: "  << t_current << "  > "  << tEnd << " loop alone took: " << totalTime <<  endl;
     
      my_dev::base_mem::printMemUsage();

      if(execStream != NULL)
      {
        delete execStream;
        execStream = NULL;
      }
      
      return true;
    }
    
<<<<<<< HEAD
  iter++;
=======
   
    iter++; 
    /*if((iter % 50) == 0)
    {
      if(removeDistance > 0) checkRemovalDistance(this->localTree);
    }
    */
>>>>>>> added a runtime Rebuild tree parameter. Can be changed at runtime, see ./main help

  return false;
}

void octree::iterate_setup(IterationData &idata) {
  real4 r_min, r_max;
  
  if(execStream == NULL)
    execStream = new my_dev::dev_stream(0);
  
  letRunning = false;
     
  double t1;

  //Initial prediction/acceleration to setup the system
  //Will be at time 0
  //predict localtree
  predict(this->localTree);
  this->getBoundaries(localTree, r_min, r_max);
  //Build the tree using the predicted positions  
  //Compute the (new) node properties
  compute_properties(this->localTree);
  
  t1 = get_time();
   
  //Approximate gravity
//   devContext.startTiming();
  approximate_gravity(this->localTree);
//   devContext.stopTiming("Approximation", 4);

  if(nProcs > 1)  makeLET();

  execStream->sync();  

  
  idata.lastGravTime   = get_time() - t1;
  idata.totalGravTime += idata.lastGravTime;
  
  correct(this->localTree);
  compute_energies(this->localTree);

  //Print time 0 snapshot
  if(snapshotIter > 0 )
  {
      int time = (int)t_current;
      if((time >= nextSnapTime))
      {
        nextSnapTime += snapshotIter;
        string fileName; fileName.resize(256);
        sprintf(&fileName[0], "%s_%06d", snapshotFile.c_str(), time + snapShotAdd);

        localTree.bodies_pos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        write_dumbp_snapshot_parallel(&localTree.bodies_pos[0], &localTree.bodies_vel[0],
                                      &localTree.bodies_ids[0], localTree.n, fileName.c_str()) ;
      }
  }

  idata.startTime = get_time();
}

void octree::iterate_teardown(IterationData &idata) {
  double totalTime = get_time() - idata.startTime;
  fprintf(stderr,"TIME [%02d] TOTAL: %g\t GRAV: %g\tBUILD: %g\tCOMM: %g\t WAIT: %g\n", 
                  procId, totalTime, idata.totalGravTime, idata.totalBuildTime, 
                  idata.totalDomTime, idata.lastWaitTime);     
  
  if(execStream != NULL)
  {
    delete execStream;
    execStream = NULL;
  }
}

void octree::iterate() {
  IterationData idata;
  iterate_setup(idata);

  for(int i=0; i < 10000000; i++) //Large number, limit
  {
    if (true == iterate_once(idata))
<<<<<<< HEAD
        break;    
=======
        break;

#if 0
    iter++;
#endif
    
>>>>>>> added a runtime Rebuild tree parameter. Can be changed at runtime, see ./main help
  } //end for i
  
  iterate_teardown(idata);
  
} //end iterate


void octree::predict(tree_structure &tree)
{
  //Functions that predicts the particles to the next timestep

//   tend is time per particle
//   tnext is reduce result

  //First we get the minimum time, which is the next integration time
  int blockSize = NBLOCK_REDUCE ;
  getTNext.set_arg<int>(0,    &tree.n);
  getTNext.set_arg<cl_mem>(1, tree.bodies_time.p());
  getTNext.set_arg<cl_mem>(2, tnext.p());
  getTNext.set_arg<float>(3,  NULL, 128); //Dynamic shared memory
  getTNext.setWork(-1, 128, blockSize);
  getTNext.execute();

  //Reduce the last parts on the host
  tnext.d2h();
  t_previous = t_current;
  t_current  = tnext[0];
  for (int i = 1; i < blockSize ; i++)
  {
      t_current = std::min(t_current, tnext[i]);
  }

  tree.activeGrpList.zeroMem();      //Reset the active grps

  //Set valid list to zero
  predictParticles.set_arg<int>(0,    &tree.n);
  predictParticles.set_arg<float>(1,  &t_current);
  predictParticles.set_arg<float>(2,  &t_previous);
  predictParticles.set_arg<cl_mem>(3, tree.bodies_pos.p());
  predictParticles.set_arg<cl_mem>(4, tree.bodies_vel.p());
  predictParticles.set_arg<cl_mem>(5, tree.bodies_acc0.p());
  predictParticles.set_arg<cl_mem>(6, tree.bodies_time.p());
  predictParticles.set_arg<cl_mem>(7, tree.body2group_list.p());
  predictParticles.set_arg<cl_mem>(8, tree.activeGrpList.p());
  predictParticles.set_arg<cl_mem>(9, tree.bodies_Ppos.p());
  predictParticles.set_arg<cl_mem>(10, tree.bodies_Pvel.p());  

  predictParticles.setWork(tree.n, 128);
  predictParticles.execute();
  
  //Compact the valid list to get a list of valid groups
  gpuCompact(devContext, tree.activeGrpList, tree.active_group_list,
             tree.n_groups, &tree.n_active_groups);

  LOG("t_previous: %lg t_current: %lg dt: %lg Active groups: %d \n",
         t_previous, t_current, t_current-t_previous, tree.n_active_groups);

}
//End predict


void octree::setActiveGrpsFunc(tree_structure &tree)
{
  tree.activeGrpList.zeroMem();      //Reset the active grps

  //Set valid list to zero
  setActiveGrps.set_arg<int>(0,    &tree.n);
  setActiveGrps.set_arg<float>(1,  &t_current);
  setActiveGrps.set_arg<cl_mem>(2, tree.bodies_time.p());
  setActiveGrps.set_arg<cl_mem>(3, tree.body2group_list.p());
  setActiveGrps.set_arg<cl_mem>(4, tree.activeGrpList.p());

  setActiveGrps.setWork(tree.n, 128);
  setActiveGrps.execute();

  //Compact the valid list to get a list of valid groups
  gpuCompact(devContext, tree.activeGrpList, tree.active_group_list,
             tree.n_groups, &tree.n_active_groups);

  LOG("t_previous: %lg t_current: %lg dt: %lg Active groups: %d \n",
         t_previous, t_current, t_current-t_previous, tree.n_active_groups);

}

void octree::approximate_gravity(tree_structure &tree)
{ 
  uint2 node_begend;
  int level_start = 2;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  LOG("node begend: %d %d iter-> %d\n", node_begend.x, node_begend.y, iter);

  //Reset the active particles
  tree.activePartlist.zeroMem();

  //Set the kernel parameters, many!
  approxGrav.set_arg<int>(0,    &tree.n_active_groups);
  approxGrav.set_arg<int>(1,    &tree.n);
  approxGrav.set_arg<float>(2,  &(this->eps2));
  approxGrav.set_arg<uint2>(3,  &node_begend);
  approxGrav.set_arg<cl_mem>(4, tree.active_group_list.p());
  approxGrav.set_arg<cl_mem>(5, tree.bodies_Ppos.p());
  approxGrav.set_arg<cl_mem>(6, tree.multipole.p());
  approxGrav.set_arg<cl_mem>(7, tree.bodies_acc1.p());
  approxGrav.set_arg<cl_mem>(8, tree.ngb.p());
  approxGrav.set_arg<cl_mem>(9, tree.activePartlist.p());
  approxGrav.set_arg<cl_mem>(10, tree.interactions.p());
  approxGrav.set_arg<cl_mem>(11, tree.boxSizeInfo.p());
  approxGrav.set_arg<cl_mem>(12, tree.groupSizeInfo.p());
  approxGrav.set_arg<cl_mem>(13, tree.boxCenterInfo.p());
  approxGrav.set_arg<cl_mem>(14, tree.groupCenterInfo.p());
  approxGrav.set_arg<cl_mem>(15, tree.bodies_Pvel.p());
  approxGrav.set_arg<cl_mem>(16,  tree.generalBuffer1.p()); //Instead of using Local memory
  
  
  
  
  approxGrav.set_arg<real4>(17, tree.boxSizeInfo, 4, "texNodeSize");
  approxGrav.set_arg<real4>(18, tree.boxCenterInfo, 4, "texNodeCenter");
  approxGrav.set_arg<real4>(19, tree.multipole, 4, "texMultipole");
  approxGrav.set_arg<real4>(20, tree.bodies_Ppos, 4, "texBody");
    
 
  approxGrav.setWork(-1, NTHREAD, nBlocksForTreeWalk);
  approxGrav.execute(execStream->s());  //First half

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
    }
  
    //cerr << "Interaction at iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    //cerr << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << endl;

    cout << "Interaction at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;    
  #endif
  
  CU_SAFE_CALL(clFinish(0));
  
  if(mpiGetNProcs() == 1) //Only do it here if there is only one process
  {
    
    //Reduce the number of valid particles    
    getNActive.set_arg<int>(0,    &tree.n);
    getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
    getNActive.set_arg<cl_mem>(2, this->nactive.p());
    getNActive.set_arg<int>(3,    NULL, 128); //Dynamic shared memory , equal to number of threads
    getNActive.setWork(-1, 128,   NBLOCK_REDUCE);
    getNActive.execute();
    
    //Reduce the last parts on the host
    this->nactive.d2h();
    tree.n_active_particles = this->nactive[0];
    for (int i = 1; i < NBLOCK_REDUCE ; i++)
        tree.n_active_particles += this->nactive[i];

    LOG("Active particles: %d \n", tree.n_active_particles);
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

  printf("LET node begend [%d]: %d %d iter-> %d\n", procId, node_begend.x, node_begend.y, iter);
  fflush(stderr);
  fflush(stdout);


  //Set the kernel parameters, many!
  approxGravLET.set_arg<int>(0,    &tree.n_active_groups);
  approxGravLET.set_arg<int>(1,    &tree.n);
  approxGravLET.set_arg<float>(2,  &(this->eps2));
  approxGravLET.set_arg<uint2>(3,  &node_begend);
  approxGravLET.set_arg<cl_mem>(4, tree.active_group_list.p());
  approxGravLET.set_arg<cl_mem>(5, remoteTree.fullRemoteTree.p());

  void *multiLoc = remoteTree.fullRemoteTree.a(2*(remoteP) + 2*(remoteN+nodeTexOffset));
  approxGravLET.set_arg<cl_mem>(6, &multiLoc);  

  approxGravLET.set_arg<cl_mem>(7, tree.bodies_acc1.p());
  approxGravLET.set_arg<cl_mem>(8, tree.ngb.p());
  approxGravLET.set_arg<cl_mem>(9, tree.activePartlist.p());
  approxGravLET.set_arg<cl_mem>(10, tree.interactions.p());
  
  void *boxSILoc = remoteTree.fullRemoteTree.a(2*(remoteP));
  approxGravLET.set_arg<cl_mem>(11, &boxSILoc);  

  approxGravLET.set_arg<cl_mem>(12, tree.groupSizeInfo.p());

  void *boxCILoc = remoteTree.fullRemoteTree.a(2*(remoteP) + remoteN + nodeTexOffset);
  approxGravLET.set_arg<cl_mem>(13, &boxCILoc);  

  approxGravLET.set_arg<cl_mem>(14, tree.groupCenterInfo.p());  
  
  void *bdyVelLoc = remoteTree.fullRemoteTree.a(1*(remoteP));
  approxGravLET.set_arg<cl_mem>(15, &bdyVelLoc);  //<- Remote bodies velocity
  
  approxGravLET.set_arg<cl_mem>(16, tree.bodies_Ppos.p()); //<- Predicted local body positions
  approxGravLET.set_arg<cl_mem>(17, tree.bodies_Pvel.p()); //<- Predicted local body velocity
  approxGravLET.set_arg<cl_mem>(18, tree.generalBuffer1.p()); //<- Predicted local body velocity
  
  approxGravLET.set_arg<real4>(19, remoteTree.fullRemoteTree, 4, "LET::texNodeSize",
                               2*(remoteP), remoteN );
  approxGravLET.set_arg<real4>(20, remoteTree.fullRemoteTree, 4, "LET::texNodeCenter",
                               2*(remoteP) + (remoteN + nodeTexOffset),
                               remoteN);
  approxGravLET.set_arg<real4>(21, remoteTree.fullRemoteTree, 4, "LET::texMultipole",
                               2*(remoteP) + 2*(remoteN + nodeTexOffset), 
                               3*remoteN);
  approxGravLET.set_arg<real4>(22, remoteTree.fullRemoteTree, 4, "LET::texBody", 0, remoteP);  
    
  approxGravLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);
 
  LOG("LET Approx config: "); approxGravLET.printWorkSize();
    
  if(letRunning)
  {
    //dont want to overwrite the data of previous LET tree
    execStream->sync();
  }
  
  remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
  tree.activePartlist.zeroMem();
//   devContext.startTiming();  
  approxGravLET.execute(execStream->s());
//   devContext.stopTiming("Approximation_let", 5);   
  
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

  if(doActiveParticles)
  {
    //Reduce the number of valid particles
    getNActive.set_arg<int>(0,    &tree.n);
    getNActive.set_arg<cl_mem>(1, tree.activePartlist.p());
    getNActive.set_arg<cl_mem>(2, this->nactive.p());
    getNActive.set_arg<int>(3, NULL, 128); //Dynamic shared memory , equal to number of threads
    getNActive.setWork(-1, 128, NBLOCK_REDUCE);
    CU_SAFE_CALL(clFinish(0));
    getNActive.execute();
    
    //Reduce the last parts on the host
    this->nactive.d2h();
    tree.n_active_particles = this->nactive[0];
    for (int i = 1; i < NBLOCK_REDUCE ; i++)
        tree.n_active_particles += this->nactive[i];

    LOG("LET Active particles: %d (Process: %d ) \n",tree.n_active_particles, mpiGetRank());
  }
}
//end approximate



void octree::correct(tree_structure &tree)
{
  correctParticles.set_arg<int   >(0, &tree.n);
  correctParticles.set_arg<float >(1, &t_current);
  correctParticles.set_arg<cl_mem>(2, tree.bodies_time.p());
  correctParticles.set_arg<cl_mem>(3, tree.activePartlist.p());
  correctParticles.set_arg<cl_mem>(4, tree.bodies_vel.p());
  correctParticles.set_arg<cl_mem>(5, tree.bodies_acc0.p());
  correctParticles.set_arg<cl_mem>(6, tree.bodies_acc1.p());
  correctParticles.set_arg<cl_mem>(7, tree.bodies_pos.p());
  correctParticles.set_arg<cl_mem>(8, tree.bodies_Ppos.p());
  correctParticles.set_arg<cl_mem>(9, tree.bodies_Pvel.p());

  correctParticles.setWork(tree.n, 128);
  correctParticles.execute();
//   clFinish(devContext.get_command_queue());


  computeDt.set_arg<int>(0,    &tree.n);
  computeDt.set_arg<float>(1,  &t_current);
  computeDt.set_arg<float>(2,  &(this->eta));
  computeDt.set_arg<int>(3,    &(this->dt_limit));
  computeDt.set_arg<float>(4,  &(this->eps2));
  computeDt.set_arg<cl_mem>(5, tree.bodies_time.p());
  computeDt.set_arg<cl_mem>(6, tree.bodies_vel.p());
  computeDt.set_arg<cl_mem>(7, tree.ngb.p());
  computeDt.set_arg<cl_mem>(8, tree.bodies_pos.p());
  computeDt.set_arg<cl_mem>(9, tree.bodies_acc0.p());
  computeDt.set_arg<cl_mem>(10, tree.activePartlist.p());
  computeDt.set_arg<float >(11, &timeStep);

  computeDt.setWork(tree.n, 128);
  computeDt.execute();
//   clFinish(devContext.get_command_queue());
}


void octree::checkRemovalDistance(tree_structure &tree)                                                                                                     
{                                                                                                                                                           
  //Download all particle properties to the host                                                                                                            
                                                                                                                                                            
  tree.bodies_pos.d2h();    //The particles positions                                                                                                       
  tree.bodies_key.d2h();    //The particles keys                                                                                                            
  tree.bodies_vel.d2h();    //Velocities                                                                                                                    
  tree.bodies_acc0.d2h();    //Acceleration                                                                                                                 
  tree.bodies_acc1.d2h();    //Acceleration                                                                                                                 
  tree.bodies_time.d2h();  //The timestep details (.x=tb, .y=te                                                                                             
  tree.bodies_ids.d2h();                                                                                                                                    
                                                                                                                                                            
  bool modified = false;                                                                                                                                    
                                                                                                                                                            
  tree.multipole.d2h();                                                                                                                                     
  real4 com = tree.multipole[0];                                                                                                                            
                                                                                                                                                            
  int storeIdx = 0;               
  
  int NTotalT = 0, NFirstT = 0, NSecondT = 0, NThirdT = 0;
                                                                                                                                                            
  for(int i=0; i < tree.n ; i++)                                                                                                                            
  {                                                                                                                                                         
    real4 posi = tree.bodies_pos[i];                                                                                                                        
                                                                                                                                                            
    real4 r;                                                                                                                                                
    r.x = (posi.x-com.x); r.y = (posi.y-com.y);r.z = (posi.z-com.z);                                                                                        
    float dist = (r.x*r.x) + (r.y*r.y) + (r.z*r.z);                                                                                                         
    dist = sqrt(dist);                                                                                                                                      
                                                                                                                                                            
    tree.bodies_pos[storeIdx] = tree.bodies_pos[i];                                                                                                         
    tree.bodies_key[storeIdx] = tree.bodies_key[i];                                                                                                         
    tree.bodies_vel[storeIdx] = tree.bodies_vel[i];                                                                                                         
    tree.bodies_acc0[storeIdx] = tree.bodies_acc0[i];                                                                                                       
    tree.bodies_acc1[storeIdx] = tree.bodies_acc1[i];                                                                                                       
    tree.bodies_time[storeIdx] = tree.bodies_time[i];                                                                                                       
    tree.bodies_ids[storeIdx] = tree.bodies_ids[i];                

    if(dist > removeDistance)                                                                                                                               
    {                                                                                                                                                       
        //Remove this particle                                                                                                                              
        cerr << "Removing particle: " << i << " distance is: " << dist;                                                                                     
        cerr << "\tPOSM: " << posi.x << " " << posi.y << " " << posi.z << " " << posi.w;                                                                    
        cerr << "\tCOM: " << com.x << " " << com.y << " " << com.z << " " << com.w << endl;                                                                 
                                                                                                                                                            
        //Add this particles potential energy to the sum                                                                                                    
//         removedPot += hostbodies[i].w*0.5*hostacc0[i].w;                                                                                                 
        modified =  true;                                                                                                                                   
    }                                                                                                                                                       
    else                                                                                                                                                    
    {                                                                                                                                                       
      storeIdx++; //Increase the store position           

      NTotalT++;
      NFirstT = 0, NSecondT = 0, NThirdT = 0;    
      
      //Specific for Jeroens files
      if(tree.bodies_ids[i] >= 0 && tree.bodies_ids[i] < 100000000) NThirdT++;
      if(tree.bodies_ids[i] >= 100000000 && tree.bodies_ids[i] < 200000000) NSecondT++;
      if(tree.bodies_ids[i] >= 200000000 && tree.bodies_ids[i] < 300000000) NFirstT++;      
    }                                                                                                                                                       
  } //end for loop           


  NTotal  = NTotalT;
  NFirst  = NFirstT;
  NSecond = NSecondT;
  NThird  = NThirdT;

                                                                                                                                                            
  if(modified)                                                                                                                                              
  {                                                                                                                                                         
    tree.setN(storeIdx);                                                                                                                                    
                                                                                                                                                            
    //Now copy them back!!! Duhhhhh                                                                                                                         
    tree.bodies_pos.h2d();    //The particles positions                                                                                                     
    tree.bodies_key.h2d();    //The particles keys                                                                                                          
    tree.bodies_vel.h2d();    //Velocities                                                                                                                  
    tree.bodies_acc0.h2d();    //Acceleration                                                                                                               
    tree.bodies_acc1.h2d();    //Acceleration                                                                                                               
    tree.bodies_time.h2d();  //The timestep details (.x=tb, .y=te                                                                                           
    tree.bodies_ids.h2d();                                                                                                                                  
                                                                                                                                                            
    //Compute the energy!                                                                                                                                   
    store_energy_flag = true;                                                                                                                               
    compute_energies(tree);                                                                                                                                
  }//end if modified                                                                                                                                        
  else                                                                                                                                                      
  {                                                                                                                                                         
        cerr << "Nothing removed! :-) \n";                                                                                                                  
  }                   
  
  //TODO sync the number of particles with process 0 for correct header file
  
  
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
    
    fprintf(stderr, "%d\t Vel: %f %f %f Mass: %f Pot: %f \n", 
            i,vel.x, vel.y, vel.z,tree.bodies_pos[i].w, tree.bodies_acc0[i].w);

  }
  MPI_Barrier(MPI_COMM_WORLD);
  double hEtot = hEpot + hEkin;
  LOG("Energy (on host): Etot = %.10lg Ekin = %.10lg Epot = %.10lg \n", hEtot, hEkin, hEpot);
  #endif

  //float2 energy : x is kinetic energy, y is potential energy
  int blockSize = NBLOCK_REDUCE ;
  my_dev::dev_mem<double2>  energy(devContext);
  energy.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                      tree.generalBuffer1.get_flags(), 
                      tree.generalBuffer1.get_devMem(),
                      &tree.generalBuffer1[0], 0,  
                      blockSize, getAllignmentOffset(0));    
    
  computeEnergy.set_arg<int>(0,    &tree.n);
  computeEnergy.set_arg<cl_mem>(1, tree.bodies_pos.p());
  computeEnergy.set_arg<cl_mem>(2, tree.bodies_vel.p());
  computeEnergy.set_arg<cl_mem>(3, tree.bodies_acc0.p());
  computeEnergy.set_arg<cl_mem>(4, energy.p());
  computeEnergy.set_arg<double>(5, NULL, 128*2); //Dynamic shared memory, equal to number of threads times 2

  computeEnergy.setWork(-1, 128, blockSize);
  computeEnergy.execute();

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

  
  double de = (Etot - Etot0)/Etot0;
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
  LOG("iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n",
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);  
  LOGF(stderr, "iter=%d : time= %lg  Etot= %.10lg  Ekin= %lg   Epot= %lg : de= %lg ( %lg ) d(de)= %lg ( %lg ) t_sim=  %lg sec\n", 
		  iter, this->t_current, Etot, Ekin, Epot, de, de_max, dde, dde_max, get_time() - tinit);          
  }

  return de;
}
