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
  gravStream->sync();  //Sync LET execution

}

#if 1

extern void read_tipsy_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<int> &bodiesIDs,  float eps2, string fileName, 
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<int> &dustIDs, int reduce_bodies_factor, int reduce_dust_factor) ;
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
    string fileName = "modelC30kDust.bin";
    int rank =0;
    int procs = 1;
    int NTotal, NFirst, NSecond, Nthird = 0;
    read_tipsy_file_parallel(newGalaxy_pos, newGalaxy_vel, newGalaxy_ids, 0, fileName, 
                             rank, procs, NTotal, NFirst, NSecond, NThird, this,
                             newGalaxy_pos_dust, newGalaxy_vel_dust, newGalaxy_ids_dust, 1, 1);
    

    int n_addGalaxy      = (int) newGalaxy_pos.size();
    int n_addGalaxy_dust = (int) newGalaxy_pos_dust.size();
    //Put the dust with the main particles for orbit computations
    newGalaxy_pos.insert(newGalaxy_pos.end(), newGalaxy_pos_dust.begin(), newGalaxy_pos_dust.end());
    newGalaxy_vel.insert(newGalaxy_vel.end(), newGalaxy_vel_dust.begin(), newGalaxy_vel_dust.end());
    newGalaxy_ids.insert(newGalaxy_ids.end(), newGalaxy_ids_dust.begin(), newGalaxy_ids_dust.end());
  
  //First we need to compute the merger parameters
  //this can be done on host or device or just precomputed
  //However we need to center the galaxy on their center of mass
  //which requires some loops, while it should be possible to do on 
  //the device since the COM info is available in the tree-structure
  //but the center of mass velocity isnt
  
  //So modify the positions to COM of the current galaxy and the new one
  
  //Compute merger parameters
  
  
  //Modify the positions/velocity and angles of the galaxy on the device
  
  //Modify the to be added galaxy
  
  
  //Now put everything together:  
  int old_n     = this->localTree.n;
  int old_ndust = this->localTree.n_dust;
  
  
  //Increase the size of the buffers
  this->localTree.setN(n_addGalaxy + this->localTree.n);
  this->reallocateParticleMemory(this->localTree); //Resize preserves original data
  

  setupMergerModel(currentGalaxy_pos,currentGalaxy_vel ,currentGalaxy_ids ,
                   newGalaxy_pos, newGalaxy_vel, newGalaxy_ids);      
  
  #ifdef USE_DUST
    this->localTree.setNDust(n_addGalaxy_dust + this->localTree.n_dust);    
    //The dust function checks if it needs to resize or malloc
    this->allocateDustMemory(this->localTree); 
  #endif  
    
  //Get particle data back to the host so we can add our new data
  this->localTree.bodies_pos.d2h();
  this->localTree.bodies_acc0.d2h();  
  this->localTree.bodies_vel.d2h();
  this->localTree.bodies_time.d2h();
  this->localTree.bodies_ids.d2h();
  this->localTree.bodies_Ppos.d2h();
  this->localTree.bodies_Pvel.d2h();
  
  //Now we have to do some memory copy magic, IF we have USE_DUST defined the lay-out is like this:
  //[[tree.n galaxy1][tree.n_dust galaxy1][n_addGalaxy galaxy2][n_addGalaxy_dust galaxy2]]
  //So lets get that in the right arrays :-)
  memcpy(&this->localTree.bodies_pos [0], &currentGalaxy_pos[0], sizeof(real4)*old_n);
  memcpy(&this->localTree.bodies_pos [old_n], &currentGalaxy_pos[old_n + old_ndust], sizeof(real4)*n_addGalaxy);

  memcpy(&this->localTree.bodies_vel [0], &currentGalaxy_vel[0], sizeof(real4)*old_n);
  memcpy(&this->localTree.bodies_vel [old_n], &currentGalaxy_vel[old_n + old_ndust], sizeof(real4)*n_addGalaxy);
  
  memcpy(&this->localTree.bodies_ids[0], &currentGalaxy_ids[0], sizeof(int)*old_n);
  memcpy(&this->localTree.bodies_ids[old_n], &currentGalaxy_ids[old_n + old_ndust], sizeof(int)*n_addGalaxy);
  
  #ifdef USE_DUST
    if(old_ndust + n_addGalaxy_dust)
    {
      memcpy(&this->localTree.dust_pos[0], &currentGalaxy_pos[old_n], sizeof(real4)*old_ndust);
      memcpy(&this->localTree.dust_vel [0], &currentGalaxy_vel[old_n], sizeof(real4)*old_ndust);
      memcpy(&this->localTree.dust_ids[0], &currentGalaxy_ids[old_n], sizeof(int)*old_ndust);

      if(n_addGalaxy_dust > 0){
        memcpy(&this->localTree.dust_pos[old_ndust], 
              &currentGalaxy_pos[old_n + old_ndust + n_addGalaxy], sizeof(real4)*n_addGalaxy_dust);
        memcpy(&this->localTree.dust_vel [old_ndust], 
              &currentGalaxy_vel[old_n + old_ndust + n_addGalaxy], sizeof(real4)*n_addGalaxy_dust);
        memcpy(&this->localTree.dust_ids[old_ndust], 
              &currentGalaxy_ids[old_n + old_ndust + n_addGalaxy], sizeof(int)*n_addGalaxy_dust);
      }
      this->localTree.dust_pos.h2d();
      this->localTree.dust_vel.h2d();
      this->localTree.dust_ids.h2d();

      this->localTree.dust_acc0.d2h();
      for(int i=old_ndust; i < old_ndust + n_addGalaxy_dust; i++)
      {
        //Zero the accelerations of the new particles
        this->localTree.dust_acc0[i] = make_float4(0.0f,0.0f,0.0f,0.0f);
      }
    //  fprintf(stderr, "Dust info %d %d \n", old_ndust, n_addGalaxy_dust);      this->localTree.dust_acc0.h2d();
      this->localTree.dust_acc1.zeroMem();
    }

  #endif


  float2 curTime = this->localTree.bodies_time[0];
  for(int i=0; i < this->localTree.n; i++)
  {
    this->localTree.bodies_time[i] = curTime;
    //Zero the accelerations of the new particles
    if(i >= old_n)
    {
      this->localTree.bodies_acc0[i] = make_float4(0.0f,0.0f,0.0f,0.0f);
    }
  }
  this->localTree.bodies_acc1.zeroMem();
  
 
  this->localTree.bodies_pos.h2d();
  this->localTree.bodies_acc0.h2d();  
  this->localTree.bodies_vel.h2d();
  this->localTree.bodies_time.h2d();
  this->localTree.bodies_ids.h2d();
  
  //Fill the predicted arrays
  this->localTree.bodies_Ppos.copy(this->localTree.bodies_pos, localTree.n);
  this->localTree.bodies_Pvel.copy(this->localTree.bodies_pos, localTree.n);
  
  resetEnergy();
  
//   for(int i=0; i < this->localTree.n; i++)
//   {
//     fprintf(stderr, "%d\t%d \n", i, this->localTree.bodies_ids[i] );
//   }

  
  //Copy the new galaxy behind the current galaxy
//   memcpy(&m_tree->localTree.bodies_pos[old_n], 
//          &new_bodyPositions[0], sizeof(real4)*new_bodyPositions.size());
//   memcpy(&m_tree->localTree.bodies_Ppos[old_n], 
//          &new_bodyPositions[0], sizeof(real4)*new_bodyPositions.size());
  //...etc. 
  

  //Send everything back to the device
  
  //With some luck we can jump directly to iterate_once
  //if that doesnt work we might have to do some extra steps...to be tested  
    
  return true;
  
}
#endif




// returns true if this iteration is the last (t_current >= t_end), false otherwise
bool octree::iterate_once(IterationData &idata) {
    double t1 = 0;

    LOG("At the start of iterate:\n");
    
    bool forceTreeRebuild = false;
    
    if((iter % 5) == 0)
    {
  //    addGalaxy(0);
   //   forceTreeRebuild = true;
    }

    //predict localtree
    devContext.startTiming(execStream->s());
    predict(this->localTree);
    devContext.stopTiming("Predict", 9, execStream->s());
    
    #ifdef USE_DUST
      //Predict, sort and set properties
      predictDustStep(this->localTree);          
    #endif

    if (useDirectGravity)
    {
      devContext.startTiming(gravStream->s());
      direct_gravity(this->localTree);
      devContext.stopTiming("Direct_gravity", 4);

#ifdef USE_DUST
      devContext.startTiming(gravStream->s());
      direct_dust(this->localTree);
      devContext.stopTiming("Direct_dust", 4, gravStream->s());
#endif      
    }
    else
    {
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

            devContext.startTiming(execStream->s());
            gpu_updateDomainDistribution(idata.lastGravTime);          
            devContext.stopTiming("DomainUpdate", 6, execStream->s());

            devContext.startTiming(execStream->s());
            gpuRedistributeParticles();
            devContext.stopTiming("Exchange", 6, execStream->s());

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

      rebuild_tree = ((iter % rebuild_tree_rate) == 0) || forceTreeRebuild;    
      if(rebuild_tree)
      {
        t1 = get_time();
        //Rebuild the tree
        this->sort_bodies(this->localTree, needDomainUpdate);

        devContext.startTiming(execStream->s());
        this->build(this->localTree);
        devContext.stopTiming("Tree-construction", 2, execStream->s());

        devContext.startTiming(execStream->s());
        this->allocateTreePropMemory(this->localTree);
        devContext.stopTiming("Memory", 11, execStream->s());      

        devContext.startTiming(execStream->s());
        this->compute_properties(this->localTree);
        devContext.stopTiming("Compute-properties", 3, execStream->s());

#ifdef DO_BLOCK_TIMESTEP
        devContext.startTiming(execStream->s());
        setActiveGrpsFunc(this->localTree);
        devContext.stopTiming("setActiveGrpsFunc", 10, execStream->s());      
        idata.Nact_since_last_tree_rebuild = 0;
#endif

        idata.lastBuildTime   = get_time() - t1;
        idata.totalBuildTime += idata.lastBuildTime;  


#ifdef USE_DUST
        //Sort and set properties      
        sort_dust(this->localTree);
        make_dust_groups(this->localTree);
        setDustGroupProperties(this->localTree);        
#endif          

      }
      else
      {
        //Dont rebuild only update the current boxes
        devContext.startTiming(execStream->s());
        this->compute_properties(this->localTree);
        devContext.stopTiming("Compute-properties", 3, execStream->s());

#ifdef USE_DUST
        setDustGroupProperties(this->localTree);
#endif          

      }//end rebuild tree

      //Approximate gravity
      t1 = get_time();
      devContext.startTiming(gravStream->s());
      approximate_gravity(this->localTree);
      devContext.stopTiming("Approximation", 4, gravStream->s());


      if(nProcs > 1)  makeLET();

#ifdef USE_DUST
      devContext.startTiming(gravStream->s());
      approximate_dust(this->localTree);
      devContext.stopTiming("Approximation_dust", 4, gravStream->s());
#endif        
    }

    gravStream->sync();
    
    idata.lastGravTime   = get_time() - t1;
//     totalGravTime += lastGravTime;
    idata.totalGravTime += idata.lastGravTime - thisPartLETExTime;
//     lastGravTime -= thisPartLETExTime;
    
    LOGF(stderr, "APPTIME [%d]: Iter: %d\t%g \n", procId, iter, idata.lastGravTime);
    
    //Corrector
    devContext.startTiming(execStream->s());
    correct(this->localTree);
    devContext.stopTiming("Correct", 8, execStream->s());
    

    #ifdef USE_DUST
      //Correct
      correctDustStep(this->localTree);  
    #endif     
    
    if(nProcs > 1)
    {
      t1 = get_time();
      devContext.startTiming(execStream->s());
      mpiSync();
      devContext.stopTiming("Unbalance", 12, execStream->s());
      idata.lastWaitTime  += get_time() - t1;
      idata.totalWaitTime += idata.lastWaitTime;
    }
    
    idata.Nact_since_last_tree_rebuild += this->localTree.n_active_particles;

    //Compute energies
    devContext.startTiming(execStream->s());
    double de = compute_energies(this->localTree); de=de;
    devContext.stopTiming("Energy", 7, execStream->s());

    if(snapshotIter > 0)
    {
      int time = (int)t_current;
      if((time >= nextSnapTime))
      {
        nextSnapTime += snapshotIter;
        string fileName; fileName.resize(256);
        sprintf(&fileName[0], "%s_%06d", snapshotFile.c_str(), time + snapShotAdd);
        
        #ifdef USE_DUST
          //We move the dust data into the position data (on the device :) )
          localTree.bodies_pos.copy_devonly(localTree.dust_pos, localTree.n_dust, localTree.n);
          localTree.bodies_vel.copy_devonly(localTree.dust_vel, localTree.n_dust, localTree.n);
          localTree.bodies_ids.copy_devonly(localTree.dust_ids, localTree.n_dust, localTree.n);
        #endif         

        localTree.bodies_pos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        write_dumbp_snapshot_parallel(&localTree.bodies_pos[0], &localTree.bodies_vel[0],
              &localTree.bodies_ids[0], localTree.n + localTree.n_dust, fileName.c_str(), t_current) ;
      }
    }

    if(t_current >= tEnd)
    {
      compute_energies(this->localTree);
      double totalTime = get_time() - idata.startTime;
      cout << " Finished: "  << t_current << "  > "  << tEnd << " loop alone took: " << totalTime <<  endl;
     
      my_dev::base_mem::printMemUsage();

      return true;
    }
   
    iter++; 

    return false;
}

void octree::iterate_setup(IterationData &idata) {
  real4 r_min, r_max;
  
  if(execStream == NULL)
    execStream = new my_dev::dev_stream(0);

  if(gravStream == NULL)
    gravStream = new my_dev::dev_stream(0);

  if(copyStream == NULL)
    copyStream = new my_dev::dev_stream(0);
  
  //Start construction of the tree
  sort_bodies(localTree, true);
  build(localTree);
  allocateTreePropMemory(localTree);
  compute_properties(localTree);


  letRunning = false;
     
  double t1;
  sort_bodies(localTree, true);
  //Initial prediction/acceleration to setup the system
  //Will be at time 0
  //predict localtree
  predict(this->localTree);
  this->getBoundaries(localTree, r_min, r_max);
  //Build the tree using the predicted positions  
  //Compute the (new) node properties
  compute_properties(this->localTree);
  
#ifdef DO_BLOCK_TIMESTEP
        devContext.startTiming(execStream->s());
        setActiveGrpsFunc(this->localTree);
        devContext.stopTiming("setActiveGrpsFunc", 10, execStream->s());      
//         idata.Nact_since_last_tree_rebuild = 0;
#endif  
  
  t1 = get_time();
   
  //Approximate gravity
  devContext.startTiming(gravStream->s());
  approximate_gravity(this->localTree);
  devContext.stopTiming("Approximation", 4, gravStream->s());
  
  #ifdef USE_DUST
      //Sort the dust
      sort_dust(localTree);
      //make the dust groups
      make_dust_groups(localTree);
      //Predict
      predictDustStep(this->localTree);
      
      //Set the group properties of dust
      setDustGroupProperties(this->localTree);
      
      devContext.startTiming(gravStream->s());
      approximate_dust(this->localTree);
      devContext.stopTiming("Approximatin_dust", 4, gravStream->s());
      //Correct
      correctDustStep(this->localTree);  
  #endif
  
  

  if(nProcs > 1)  makeLET();

  gravStream->sync();  

  
  idata.lastGravTime   = get_time() - t1;
  idata.totalGravTime += idata.lastGravTime;
  
  correct(this->localTree);
  compute_energies(this->localTree);

  //Print time 0 snapshot
  if(snapshotIter > 0 )
  {
      int time = (int)t_current;
      
      //We always snapshot the state at the current time, so we have the start
      //of the simulation included in our snapshots. This also allows us to
      //adjust the nextSnapTime to the correct starting point now that we can start
      //at time != 0
      //if((time >= nextSnapTime))
      nextSnapTime = time;
      if(1)
      {
        nextSnapTime += snapshotIter;       
        
        
        string fileName; fileName.resize(256);
        sprintf(&fileName[0], "%s_%06d", snapshotFile.c_str(), time + snapShotAdd);
        
        #ifdef USE_DUST
          //We move the dust data into the position data (on the device :) )
          localTree.bodies_pos.copy_devonly(localTree.dust_pos, localTree.n_dust, localTree.n);
          localTree.bodies_vel.copy_devonly(localTree.dust_vel, localTree.n_dust, localTree.n);
          localTree.bodies_ids.copy_devonly(localTree.dust_ids, localTree.n_dust, localTree.n);
        #endif         

        localTree.bodies_pos.d2h();
        localTree.bodies_vel.d2h();
        localTree.bodies_ids.d2h();

        write_dumbp_snapshot_parallel(&localTree.bodies_pos[0], &localTree.bodies_vel[0],
          &localTree.bodies_ids[0], localTree.n + localTree.n_dust, fileName.c_str(), t_current) ;
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

  if(gravStream != NULL)
  {
    delete gravStream;
    gravStream = NULL;
  }

  if(copyStream != NULL)
  {
    delete copyStream;
    copyStream = NULL;
  }
}

void octree::iterate() {
  IterationData idata;
  iterate_setup(idata);

  for(int i=0; i < 10000000; i++) //Large number, limit
  {
    if (true == iterate_once(idata))
      break;    
  } //end for i
  
  iterate_teardown(idata);
  
} //end iterate


void octree::predict(tree_structure &tree)
{
  //Functions that predicts the particles to the next timestep

//   tend is time per particle
//   tnext is reduce result

  //First we get the minimum time, which is the next integration time
  #ifdef DO_BLOCK_TIMESTEP
    int blockSize = NBLOCK_REDUCE ;
    getTNext.set_arg<int>(0,    &tree.n);
    getTNext.set_arg<cl_mem>(1, tree.bodies_time.p());
    getTNext.set_arg<cl_mem>(2, tnext.p());
    getTNext.set_arg<float>(3,  NULL, 128); //Dynamic shared memory
    getTNext.setWork(-1, 128, blockSize);
    getTNext.execute(execStream->s());

    //Reduce the last parts on the host
    tnext.d2h();
    t_previous = t_current;
    t_current  = tnext[0];
    for (int i = 1; i < blockSize ; i++)
    {
        t_current = std::min(t_current, tnext[i]);
    }
    tree.activeGrpList.zeroMem();      //Reset the active grps
  #else
    static int temp = 0;
    t_previous =  t_current;
    if(temp > 0)
      t_current  += timeStep;
    else
       temp = 1;
  #endif

    
  //Set valid list to zero
  predictParticles.set_arg<int>(0,    &tree.n);
  predictParticles.set_arg<float>(1,  &t_current);
  predictParticles.set_arg<float>(2,  &t_previous);
  predictParticles.set_arg<cl_mem>(3, tree.bodies_pos.p());
  predictParticles.set_arg<cl_mem>(4, tree.bodies_vel.p());
  predictParticles.set_arg<cl_mem>(5, tree.bodies_acc0.p());
  predictParticles.set_arg<cl_mem>(6, tree.bodies_time.p());
  predictParticles.set_arg<cl_mem>(7, tree.bodies_Ppos.p());
  predictParticles.set_arg<cl_mem>(8, tree.bodies_Pvel.p());  

  predictParticles.setWork(tree.n, 128);
  predictParticles.execute(execStream->s());
  

  #ifdef DO_BLOCK_TIMESTEP
    //Compact the valid list to get a list of valid groups
    gpuCompact(devContext, tree.activeGrpList, tree.active_group_list,
              tree.n_groups, &tree.n_active_groups);
  #else
    tree.n_active_groups = tree.n_groups;
  #endif


  LOG("t_previous: %lg t_current: %lg dt: %lg Active groups: %d \n",
         t_previous, t_current, t_current-t_previous, tree.n_active_groups);
  
} //End predict


void octree::setActiveGrpsFunc(tree_structure &tree)
{
  tree.activeGrpList.zeroMem();      //Reset the active grps
  this->resetCompact();              //Make sure compact has been reset
  
  //Set valid list to zero
  setActiveGrps.set_arg<int>(0,    &tree.n);
  setActiveGrps.set_arg<float>(1,  &t_current);
  setActiveGrps.set_arg<cl_mem>(2, tree.bodies_time.p());
  setActiveGrps.set_arg<cl_mem>(3, tree.body2group_list.p());
  setActiveGrps.set_arg<cl_mem>(4, tree.activeGrpList.p());

  setActiveGrps.setWork(tree.n, 128);
  setActiveGrps.execute(execStream->s());

  //Compact the valid list to get a list of valid groups
  gpuCompact(devContext, tree.activeGrpList, tree.active_group_list,
             tree.n_groups, &tree.n_active_groups);

  this->resetCompact();   
  LOG("t_previous: %lg t_current: %lg dt: %lg Active groups: %d (Total: %d)\n",
         t_previous, t_current, t_current-t_previous, tree.n_active_groups, tree.n_groups);
  

}

void octree::direct_gravity(tree_structure &tree)
{
  directGrav.set_arg<cl_mem>(0, tree.bodies_acc1.p());
  directGrav.set_arg<cl_mem>(1, tree.bodies_Ppos.p());
  directGrav.set_arg<cl_mem>(2, tree.bodies_Ppos.p());
  directGrav.set_arg<int>(3,    &tree.n);
  directGrav.set_arg<int>(4,    &tree.n);
  directGrav.set_arg<float>(5,  &(this->eps2));
  directGrav.set_arg<float4>(6, NULL, 256);
  std::vector<size_t> localWork(2), globalWork(2);
  localWork[0] = 256; localWork[1] = 1;
  globalWork[0] = 256 * ((tree.n + 255) / 256);
  globalWork[1] = 1;
  directGrav.setWork(globalWork, localWork);
  directGrav.execute(gravStream->s());  //First half
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
  approxGrav.set_arg<cl_mem>(8, tree.bodies_Ppos.p());
  approxGrav.set_arg<cl_mem>(9, tree.ngb.p());
  approxGrav.set_arg<cl_mem>(10, tree.activePartlist.p());
  approxGrav.set_arg<cl_mem>(11, tree.interactions.p());
  approxGrav.set_arg<cl_mem>(12, tree.boxSizeInfo.p());
  approxGrav.set_arg<cl_mem>(13, tree.groupSizeInfo.p());
  approxGrav.set_arg<cl_mem>(14, tree.boxCenterInfo.p());
  approxGrav.set_arg<cl_mem>(15, tree.groupCenterInfo.p());
  approxGrav.set_arg<cl_mem>(16, tree.bodies_Pvel.p());
  approxGrav.set_arg<cl_mem>(17,  tree.generalBuffer1.p()); //Instead of using Local memory
  
  
  
  
  approxGrav.set_arg<real4>(18, tree.boxSizeInfo, 4, "texNodeSize");
  approxGrav.set_arg<real4>(19, tree.boxCenterInfo, 4, "texNodeCenter");
  approxGrav.set_arg<real4>(20, tree.multipole, 4, "texMultipole");
  approxGrav.set_arg<real4>(21, tree.bodies_Ppos, 4, "texBody");
    
 
  approxGrav.setWork(-1, NTHREAD, nBlocksForTreeWalk);
  approxGrav.execute(gravStream->s());  //First half

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
      
      fprintf(stderr, "%d\t Direct: %d\tApprox: %d\t Group: %d \n",
              i, tree.interactions[i].y, tree.interactions[i].x, 
              tree.body2group_list[i]);
    }
  
    //cerr << "Interaction at iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    //cerr << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << endl;

    cout << "Interaction at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n << "\tavg appr: " << apprSum / tree.n << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n)  <<  endl;    
    exit(0);
  #endif
  
  //CU_SAFE_CALL(clFinish(0));

  
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
  approxGravLET.set_arg<cl_mem>(8, tree.bodies_Ppos.p());
  approxGravLET.set_arg<cl_mem>(9, tree.ngb.p());
  approxGravLET.set_arg<cl_mem>(10, tree.activePartlist.p());
  approxGravLET.set_arg<cl_mem>(11, tree.interactions.p());
  
  void *boxSILoc = remoteTree.fullRemoteTree.a(2*(remoteP));
  approxGravLET.set_arg<cl_mem>(12, &boxSILoc);  

  approxGravLET.set_arg<cl_mem>(13, tree.groupSizeInfo.p());

  void *boxCILoc = remoteTree.fullRemoteTree.a(2*(remoteP) + remoteN + nodeTexOffset);
  approxGravLET.set_arg<cl_mem>(14, &boxCILoc);  

  approxGravLET.set_arg<cl_mem>(15, tree.groupCenterInfo.p());  
  
//   void *bdyVelLoc = remoteTree.fullRemoteTree.a(1*(remoteP));
//   approxGravLET.set_arg<cl_mem>(16, &bdyVelLoc);  //<- Remote bodies velocity
  
  approxGravLET.set_arg<cl_mem>(16, tree.bodies_Pvel.p()); //<- Predicted local body velocity
  approxGravLET.set_arg<cl_mem>(17, tree.generalBuffer1.p()); //<- Predicted local body velocity
  
  approxGravLET.set_arg<real4>(18, remoteTree.fullRemoteTree, 4, "LET::texNodeSize",
                               2*(remoteP), remoteN );
  approxGravLET.set_arg<real4>(19, remoteTree.fullRemoteTree, 4, "LET::texNodeCenter",
                               2*(remoteP) + (remoteN + nodeTexOffset),
                               remoteN);
  approxGravLET.set_arg<real4>(20, remoteTree.fullRemoteTree, 4, "LET::texMultipole",
                               2*(remoteP) + 2*(remoteN + nodeTexOffset), 
                               3*remoteN);
  approxGravLET.set_arg<real4>(21, remoteTree.fullRemoteTree, 4, "LET::texBody", 0, remoteP);  
    
  approxGravLET.setWork(-1, NTHREAD, nBlocksForTreeWalk);
 
  LOG("LET Approx config: "); approxGravLET.printWorkSize();
    
  if(letRunning)
  {
    //dont want to overwrite the data of previous LET tree
    gravStream->sync();
  }
  
  remoteTree.fullRemoteTree.h2d(bufferSize); //Only copy required data
  tree.activePartlist.zeroMem();
//   devContext.startTiming(gravStream->s());  
  approxGravLET.execute(gravStream->s());
//   devContext.stopTiming("Approximation_let", 5, gravStream->s());   
  
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
    //CU_SAFE_CALL(clFinish(0));
    getNActive.execute(execStream->s());
    
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
  my_dev::dev_mem<float2>  float2Buffer(devContext);
  my_dev::dev_mem<real4>   real4Buffer1(devContext);

  int memOffset = float2Buffer.cmalloc_copy(tree.generalBuffer1, 
                                             tree.n, 0);
      memOffset = real4Buffer1.cmalloc_copy(tree.generalBuffer1, 
                                             tree.n, memOffset);  
  
 
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
  correctParticles.set_arg<cl_mem>(10, tree.oriParticleOrder.p());
  correctParticles.set_arg<cl_mem>(11, real4Buffer1.p());
  correctParticles.set_arg<cl_mem>(12, float2Buffer.p());

#if 1
  //Buffers required for storing the position of selected particles
  correctParticles.set_arg<cl_mem>(13, tree.bodies_ids.p());
  correctParticles.set_arg<cl_mem>(14, specialParticles.p());

#endif

  correctParticles.setWork(tree.n, 128);
  correctParticles.execute(execStream->s());
 
/* specialParticles.d2h();
fprintf(stderr, "Sun: %f %f %f %f \n", specialParticles[0].x, 
  specialParticles[0].y, specialParticles[0].z, specialParticles[0].w); 
  
fprintf(stderr, "m31: %f %f %f %f \n", specialParticles[1].x, 
  specialParticles[1].y, specialParticles[1].z, specialParticles[1].w);  */


  tree.bodies_acc0.copy(real4Buffer1, tree.n);
  tree.bodies_time.copy(float2Buffer, float2Buffer.get_size()); 
  

  #ifdef DO_BLOCK_TIMESTEP
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
    computeDt.execute(execStream->s());
  #endif

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
  energy.cmalloc_copy(tree.generalBuffer1, blockSize, 0);
  

    
  computeEnergy.set_arg<int>(0,    &tree.n);
  computeEnergy.set_arg<cl_mem>(1, tree.bodies_pos.p());
  computeEnergy.set_arg<cl_mem>(2, tree.bodies_vel.p());
  computeEnergy.set_arg<cl_mem>(3, tree.bodies_acc0.p());
  computeEnergy.set_arg<cl_mem>(4, energy.p());
  computeEnergy.set_arg<double>(5, NULL, 128*2); //Dynamic shared memory, equal to number of threads times 2

  computeEnergy.setWork(-1, 128, blockSize);
  computeEnergy.execute(execStream->s());

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
