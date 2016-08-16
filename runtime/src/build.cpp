#include "octree.h"
#include "build.h"

void octree::allocateParticleMemory(tree_structure &tree)
{
  //Allocates the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;


  //MULTI_GPU_MEM_INCREASE% extra space, only in parallel when
  if(nProcs > 1) n_bodies = (int)(n_bodies*MULTI_GPU_MEM_INCREASE);    //number of particles can fluctuate

  //Particle properties
  tree.bodies_pos.cmalloc(n_bodies+1, true);   //+1 to set end pos, host mapped? TODO not needed right since we use Ppos
  tree.bodies_key.cmalloc(n_bodies+1, false);   //+1 to set end key
  tree.bodies_ids.cmalloc(n_bodies+1, false);   //+1 to set end key

  tree.bodies_Ppos.cmalloc(n_bodies+1, true);   //Memory to store predicted positions, host mapped
  tree.bodies_Pvel.cmalloc(n_bodies+1, true);   //Memory to store predicted velocities, host mapped

  tree.bodies_vel.cmalloc(n_bodies, false);
  tree.bodies_acc0.ccalloc(n_bodies, false);    //ccalloc -> init to 0
  tree.bodies_acc1.ccalloc(n_bodies, false);    //ccalloc -> init to 0
  tree.bodies_time.ccalloc(n_bodies, false);    //ccalloc -> init to 0

  //density
  tree.bodies_h.cmalloc(n_bodies, true);
  tree.bodies_dens.cmalloc(n_bodies, true);
  //Init to -1
  for(int i=0; i < n_bodies; i++) tree.bodies_h[i] = -1;
  tree.bodies_h.h2d();


  tree.oriParticleOrder.cmalloc(n_bodies, false);      //To desort the bodies tree later on
  //iteration properties / information
  tree.activePartlist.ccalloc(n_bodies+2, false);   //+2 since we use the last two values as a atomicCounter (for grp count and semaphore access)
  tree.ngb.ccalloc(n_bodies, false);
  tree.interactions.cmalloc(n_bodies, false);

  tree.body2group_list.cmalloc(n_bodies, false);

  tree.level_list.cmalloc(MAXLEVELS);
  tree.node_level_list.cmalloc(MAXLEVELS*2 , false);


  //The generalBuffer is also used during the tree-walk, so the size has to be at least
  //large enough to store the tree-walk stack. Add 4096 for extra memory alignment space
  //Times 2 since first half is used for regular walks, 2nd half for walks that go outside
  //the memory stack and require another walk with 'unlimited' stack size
#if 0
  int treeWalkStackSize = (2*LMEM_STACK_SIZE*NTHREAD*nBlocksForTreeWalk) + 4096;
#else
  int treeWalkStackSize = (2*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*nBlocksForTreeWalk) + 4096;
#endif


  int tempSize   = max(n_bodies, 4096);   //Use minimum of 4096 to prevent offsets mess up with small N
  tempSize       = 3*tempSize *4 + 4096;  //Add 4096 to give some space for memory alignment
  tempSize       = max(tempSize, treeWalkStackSize);

  //General buffer is used at multiple locations and reused in different functions
  tree.generalBuffer1.cmalloc(tempSize, true);


  //Tree properties, tree size is not known at fore hand so
  //allocate worst possible outcome
  int tempmem = n_bodies ; //Some default size in case tree.n is small
  if(tree.n < 1024)
    tempmem = 2048;

  tree.n_children.cmalloc (tempmem, false);
  tree.node_bodies.cmalloc(tempmem, false);

  //General memory buffers

  //Set the context for the memory
  this->tnext.setContext(devContext);
  this->tnext.ccalloc(NBLOCK_REDUCE,false);

  this->nactive.setContext(devContext);
  this->nactive.ccalloc(NBLOCK_REDUCE,false);

  this->devMemRMIN.setContext(devContext);
  this->devMemRMIN.cmalloc(NBLOCK_BOUNDARY, false);

  this->devMemRMAX.setContext(devContext);
  this->devMemRMAX.cmalloc(NBLOCK_BOUNDARY, false);

  this->devMemCounts.setContext(devContext);
  this->devMemCounts.cmalloc(NBLOCK_PREFIX, false);

  this->devMemCountsx.setContext(devContext);
  this->devMemCountsx.cmalloc(NBLOCK_PREFIX, true);

  this->specialParticles.setContext(devContext);
  this->specialParticles.cmalloc(16, true);


  if(mpiGetNProcs() > 1)
  {
    int remoteSize = (int)(n_bodies*0.5); //TODO some more realistic number
    if(remoteSize < 1024 ) remoteSize = 2048;

    this->remoteTree.fullRemoteTree.cmalloc(remoteSize, true);
    tree.parallelBoundaries.cmalloc(mpiGetNProcs()+1, true);
  }

}


void octree::reallocateParticleMemory(tree_structure &tree)
{
  //Reallocate the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;


  if(tree.activePartlist.get_size() < tree.n)
    n_bodies *= MULTI_GPU_MEM_INCREASE;


  bool reduce = false;  //Set this to true to limit memory usage by only allocating what
                        //is required. If its false, then memory is not reduced and a larger
                        //buffer is kept

  //Particle properties
  tree.bodies_pos.cresize(n_bodies+1, reduce);   //+1 to set boundary condition
  tree.bodies_key.cresize(n_bodies+1, reduce);   //+1 to set boundary condition
  tree.bodies_ids.cresize(n_bodies+1, reduce);   //

  tree.bodies_Ppos.cresize(n_bodies+1, reduce);   //Memory to store predicted positions
  tree.bodies_Pvel.cresize(n_bodies+1, reduce);   //Memory to store predicted velocities

  tree.bodies_vel.cresize (n_bodies, reduce);
  tree.bodies_acc0.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  tree.bodies_acc1.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  tree.bodies_time.cresize(n_bodies, reduce);    //ccalloc -> init to 0
  
  //Density
  const int oldHsize = tree.bodies_h.get_size();
  tree.bodies_h.cresize(n_bodies, reduce);
  tree.bodies_dens.cresize(n_bodies, reduce);
  
  tree.oriParticleOrder.cresize(n_bodies,   reduce);     //To desort the bodies tree later on
  //iteration properties / information
  tree.activePartlist.cresize(  n_bodies+2, reduce);      //+1 since we use the last value as a atomicCounter
  tree.ngb.cresize(             n_bodies,   reduce);
  tree.interactions.cresize(    n_bodies,   reduce);

  tree.body2group_list.cresize(n_bodies, reduce);

  //Tree properties, tree size is not known at forehand so
  //allocate worst possible outcome
  int tempmem = n_bodies ; //Some default size in case tree.n is small
  if(n_bodies < 1024)  tempmem = 2048;
  tree.n_children.cresize(tempmem, reduce);
  tree.node_bodies.cresize(tempmem, reduce);


  //Don't forget to resize the generalBuffer....
  int treeWalkStackSize = (2*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*nBlocksForTreeWalk) + 4096;

  int tempSize   = max(n_bodies, 4096);   //Use minimum of 4096 to prevent offsets mess up with small N
  tempSize       = 3*tempSize *4 + 4096;  //Add 4096 to give some space for memory alignment
  tempSize       = max(tempSize, treeWalkStackSize);

  //General buffer is used at multiple locations and reused in different functions
  tree.generalBuffer1.cresize(tempSize, reduce);

  my_dev::base_mem::printMemUsage();
}

void octree::allocateTreePropMemory(tree_structure &tree)
{
  devContext.startTiming(execStream->s());
  int n_nodes = tree.n_nodes;

  //Allocate memory
  if(tree.groupCenterInfo.get_size() > 0)
  {
    if(tree.boxSizeInfo.get_size() <= n_nodes)
      n_nodes *= MULTI_GPU_MEM_INCREASE;

    //Resize, so we don't allocate if we already have mem allocated
    tree.multipole.cresize_nocpy(3*n_nodes,     false);
    tree.boxSizeInfo.cresize_nocpy(n_nodes,     false); //host allocated
    tree.boxCenterInfo.cresize_nocpy(n_nodes,   false); //host allocated

    int n_groups = tree.n_groups;
    if(tree.groupSizeInfo.get_size() <= n_groups)
      n_groups *= MULTI_GPU_MEM_INCREASE;

    tree.groupSizeInfo.cresize_nocpy(n_groups,   false);
    tree.groupCenterInfo.cresize_nocpy(n_groups, false);
  }
  else
  {
    //First call to this function
    n_nodes = (int)(n_nodes * 1.1f);
    tree.multipole.cmalloc(3*n_nodes, true); //host allocated

    tree.boxSizeInfo.cmalloc(n_nodes, true);     //host allocated
    tree.groupSizeInfo.cmalloc(tree.n_groups, true);

    tree.boxCenterInfo.cmalloc(n_nodes, true); //host allocated
    tree.groupCenterInfo.cmalloc(tree.n_groups,true);
  }
  devContext.stopTiming("Memory", 11, execStream->s());
}

void octree::build (tree_structure &tree) {

  devContext.startTiming(execStream->s());
  int level      = 0;
  int validCount = 0;
  int offset     = 0;

  this->resetCompact();

  /******** create memory buffers **********/


  my_dev::dev_mem<uint>   validList(devContext);
  my_dev::dev_mem<uint>   compactList(devContext);
  my_dev::dev_mem<uint>   levelOffset(devContext);
  my_dev::dev_mem<uint>   maxLevel(devContext);
  my_dev::dev_mem<uint4>  node_key(devContext);



  int memBufOffset = validList.cmalloc_copy  (tree.generalBuffer1, tree.n*2, 0);
      memBufOffset = compactList.cmalloc_copy(tree.generalBuffer1, tree.n*2, memBufOffset);
  int memBufOffsetValidList = memBufOffset;

  int tempmem = std::max(2048, tree.n); //Some default size in case tree.n is small

  memBufOffset = node_key.cmalloc_copy   (tree.generalBuffer1, tempmem,  memBufOffset);
  memBufOffset = levelOffset.cmalloc_copy(tree.generalBuffer1, 256,      memBufOffset);
  memBufOffset = maxLevel.cmalloc_copy   (tree.generalBuffer1, 256,      memBufOffset);

  //Memory layout of the above (in uint):
  //[[validList--2*tree.n],[compactList--2*tree.n],[node_key--4*tree.n],
  //[levelOffset--256], [maxLevel--256], [free--at-least: 12-8*tree.n-256]]

  //Set the default values to zero
  validList.  zeroMemGPUAsync(execStream->s());
  levelOffset.zeroMemGPUAsync(execStream->s());
  maxLevel.   zeroMemGPUAsync(execStream->s());
  //maxLevel.zeroMem is required to let the tree-construction work properly.
  //It assumes maxLevel is zero for determining the start-level / min-level value.


  /******** set kernels parameters **********/


  build_key_list.set_arg<cl_mem>(0,   tree.bodies_key.p());
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<int>(2,      &tree.n);
  build_key_list.set_arg<real4>(3,    &tree.corner);
  build_key_list.setWork(tree.n, 128);

  build_valid_list.set_arg<int>(0, &tree.n);
  build_valid_list.set_arg<int>(1, &level);
  build_valid_list.set_arg<cl_mem>(2,  tree.bodies_key.p());
  build_valid_list.set_arg<cl_mem>(3,  validList.p());
  build_valid_list.set_arg<cl_mem>(4,  this->devMemCountsx.p());
  build_valid_list.setWork(tree.n, 128);

  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 120*32;  globalWork[1] = 4;
  localWork [0] = 128;      localWork [1] = 1;

  build_nodes.setWork(globalWork, localWork);
  build_nodes.set_arg<cl_mem>(1,  this->devMemCountsx.p());
  build_nodes.set_arg<cl_mem>(2,  levelOffset.p());
  build_nodes.set_arg<cl_mem>(3,  maxLevel.p());
  build_nodes.set_arg<cl_mem>(4,  tree.level_list.p());
  build_nodes.set_arg<cl_mem>(5,  compactList.p());
  build_nodes.set_arg<cl_mem>(6,  tree.bodies_key.p());
  build_nodes.set_arg<cl_mem>(7,  node_key.p());
  build_nodes.set_arg<cl_mem>(8,  tree.n_children.p());
  build_nodes.set_arg<cl_mem>(9,  tree.node_bodies.p());

  link_tree.set_arg<int>(0,     &offset);
  link_tree.set_arg<cl_mem>(1,  tree.n_children.p());
  link_tree.set_arg<cl_mem>(2,  tree.node_bodies.p());
  link_tree.set_arg<cl_mem>(3,  tree.bodies_Ppos.p());
  link_tree.set_arg<real4>(4,   &tree.corner);
  link_tree.set_arg<cl_mem>(5,  tree.level_list.p());
  link_tree.set_arg<cl_mem>(6,  validList.p());
  link_tree.set_arg<cl_mem>(7,  node_key.p());
  link_tree.set_arg<cl_mem>(8,  tree.bodies_key.p());

  /******  build the levels *********/

  // make sure previous resetCompact() has finished.
  this->devMemCountsx.waitForCopyEvent();

//  devContext.startTiming(execStream->s());

  if(nProcs > 1)
  {
    //Start copying the particle positions to the host, will overlap with tree-construction
    localTree.bodies_Ppos.d2h(tree.n, false, LETDataToHostStream->s());
  }

//  double tBuild0 = get_time();

#if 0
  build_tree_node_levels(*this, validList, compactList, levelOffset, maxLevel, execStream->s());
#else
  for (level = 0; level < MAXLEVELS; level++) {
    // mark bodies to be combined into nodes
    build_valid_list.set_arg<int>(1, &level);
    build_valid_list.execute(execStream->s());

    //gpuCompact to get number of created nodes
    gpuCompact(devContext, validList, compactList, tree.n*2, 0);

    // assemble nodes
    build_nodes.set_arg<int>(0, &level);
    build_nodes.execute(execStream->s());
  } //end for level

  // reset counts to 1 so next compact proceeds...
  this->resetCompact();
#endif

//  execStream->sync();
//  const double dt = get_time() - tBuild0;
//  fprintf(stderr, " done in %g sec : %g Mptcl/sec\n", dt, tree.n/1e6/dt);
//  devContext.stopTiming("Create-nodes", 10, execStream->s());

  maxLevel.d2h(1);
  level = maxLevel[0];

  levelOffset.d2h(1);
  offset = levelOffset[0];

  tree.n_nodes = offset;


  /***** Link the tree ******/

  //The maximum number of levels that can be used is MAXLEVEl
  //if max level is larger than that the program will exit
  LOG("Max level : %d \n", level);
  if(level >= MAXLEVELS)
  {
    cerr << "The tree has become too deep, the program will exit. \n";
    cerr << "Consider the removal of far away particles to prevent a too large box. \n";
    exit(0);
  }

  link_tree.setWork(tree.n_nodes , 128);
  link_tree.printWorkSize("Link_tree: ");

  tree.n_levels = level-1;

  tree.level_list.d2h();
  tree.startLevelMin = 0;
  for(int i=0; i < level; i++)
  {
    LOG("%d\t%d\t%d\n", i, tree.level_list[i].x, tree.level_list[i].y);
    //Determine which level is to used as min_level
    if(((tree.level_list[i].y - tree.level_list[i].x) > START_LEVEL_MIN_NODES) && (tree.startLevelMin == 0))
    {
      tree.startLevelMin = i;
    }
  }
  LOG("Start at: Level: %d  begin-end: %d %d \n", tree.startLevelMin,
      tree.level_list[tree.startLevelMin].x, tree.level_list[tree.startLevelMin].y);


  //Link the tree
  link_tree.set_arg<int>(0, &offset);   //Offset=number of nodes
  link_tree.set_arg<int>(9, &tree.startLevelMin);
  link_tree.execute(execStream->s());

  //After executing link_tree, the id_list contains for each node the ID of its parent.
  //Valid_list contains for each node if its a leaf (valid) or a normal node -> non_valid
  //Execute a split on the validList to get separate id lists
  //for the leafs and nodes. Used when computing multipole expansions

  if(tree.leafNodeIdx.get_size() > 0)
    tree.leafNodeIdx.cresize_nocpy(tree.n_nodes, false);
  else
    tree.leafNodeIdx.cmalloc(tree.n_nodes , false);

  //Split the leaf nodes and non-leaf nodes
  gpuSplit(devContext, validList, tree.leafNodeIdx, tree.n_nodes, &tree.n_leafs);

  LOG("Total nodes: %d N_leafs: %d  non-leafs: %d \n", tree.n_nodes, tree.n_leafs, tree.n_nodes - tree.n_leafs);


  build_level_list.set_arg<int>(0, &tree.n_nodes);
  build_level_list.set_arg<int>(1, &tree.n_leafs);
  build_level_list.set_arg<cl_mem>(2, tree.leafNodeIdx.p());
  build_level_list.set_arg<cl_mem>(3, tree.node_bodies.p());
  build_level_list.set_arg<cl_mem>(4, validList.p());
  build_level_list.setWork(tree.n_nodes-tree.n_leafs, 128);

  validList.zeroMemGPUAsync(execStream->s());

  //Build the level list based on the leafIdx list, required for easy
  //access during the compute node properties / multipole computation
  build_level_list.execute(execStream->s());

  //Compact the node-level boundaries into the node_level_list
  gpuCompact(devContext, validList, tree.node_level_list,
             2*(tree.n_nodes-tree.n_leafs), 0);

  /************   Start building the particle groups   *************/

  //We use the minimum tree-level to set extra boundaries, which ensures
  //that groups are based on the tree-structure and will not be very big

  //Now use the previous computed offsets to build all boundaries.
  //The ones based on top level boundaries and the group ones (every NCRIT particles)
  validList.zeroMemGPUAsync(execStream->s());

  define_groups.set_arg<int>    (0, &tree.n);
  define_groups.set_arg<cl_mem> (1, validList.p());
  define_groups.set_arg<uint2>  (2, &tree.level_list[tree.startLevelMin+1]); //Bit deeper then the minimum for extra cuts
  define_groups.set_arg<cl_mem> (3, tree.node_bodies.p());
  define_groups.set_arg<cl_mem> (4, tree.node_level_list.p());
  define_groups.set_arg<int>    (5, &level);
  define_groups.setWork(tree.n, 128);
  define_groups.execute(execStream->s());

  //Ccopy the node_level_list back to host since we need it in compute properties
  LOG("Finished level list \n");
  tree.node_level_list.d2h();
  for(int i=0; i < level; i++)
  {
    LOG("node_level_list: %d \t%d\n", i, tree.node_level_list[i]);
  }

  //Now compact validList to get the list of group ids
  gpuCompact(devContext, validList, compactList, tree.n*2, &validCount);
  this->resetCompact();
  tree.n_groups = validCount/2;
  LOG("Found number of groups: %d \n", tree.n_groups);

  if(tree.group_list.get_size() > 0)
    tree.group_list.cresize_nocpy(tree.n_groups, false);
  else
    tree.group_list.cmalloc(tree.n_groups , false);


  store_groups.set_arg<int>(0,    &tree.n);
  store_groups.set_arg<int>(1,    &tree.n_groups);
  store_groups.set_arg<cl_mem>(2, compactList.p());
  store_groups.set_arg<cl_mem>(3, tree.body2group_list.p());
  store_groups.set_arg<cl_mem>(4, tree.group_list.p());
  store_groups.setWork(-1, NCRIT, tree.n_groups);
  store_groups.execute(execStream->s());


  //Memory allocation for the valid group lists
  if(tree.active_group_list.get_size() > 0)
  {
    tree.active_group_list.cresize_nocpy(tree.n_groups, false);
    tree.activeGrpList.cresize_nocpy(tree.n_groups, false);
  }
  else
  {
    tree.active_group_list.cmalloc(tree.n_groups, false);
    tree.activeGrpList.cmalloc(tree.n_groups, false);
  }

  LOG("Tree built complete!\n");
  devContext.stopTiming("Tree-construction", 2, execStream->s());

  /*************************/
}


//This function builds a hash-table for the particle-keys which is required for the
//domain distribution based on the SFC
void octree::parallelDataSummary(tree_structure &tree,
                                 float lastExecTime, float lastExecTime2,
                                 double &domComp, double &domExch,
                                 bool initialSetup) {
  double t0 = get_time();

  bool updateBoundaries = false;

  //Update if the maximum duration is 10% larger than average duration
  //and always update the first couple of iterations to create load-balance
  if(iter < 32 || (  100*((maxExecTimePrevStep-avgExecTimePrevStep) / avgExecTimePrevStep) > 10 ))
  {
    updateBoundaries = true;
  }

  //updateBoundaries = true; //TEST, keep always update for now


  real4 r_min = {+1e10, +1e10, +1e10, +1e10};
  real4 r_max = {-1e10, -1e10, -1e10, -1e10};
  this->getBoundaries(tree, r_min, r_max); //Used for predicted position keys further down

  if(updateBoundaries)
  {
    //Build keys on current positions, since those are already sorted, while predicted are not
    build_key_list.set_arg<cl_mem>(0,   tree.bodies_key.p());
    build_key_list.set_arg<cl_mem>(1,   tree.bodies_pos.p());
    build_key_list.set_arg<int>(2,      &tree.n);
    build_key_list.set_arg<real4>(3,    &tree.corner);
    build_key_list.setWork(tree.n, 128); //128 threads per block
    build_key_list.execute(execStream->s());

    /* added by evghenii, needed for 2D domain decomposition in parallel.cpp */
    tree.bodies_key.d2h(true,execStream->s());
  }

   //Get the global boundaries and compute the corner / size of tree
   this->sendCurrentRadiusInfo(r_min, r_max);
   real size     = 1.001f*std::max(r_max.z - r_min.z,
                          std::max(r_max.y - r_min.y, r_max.x - r_min.x));

   tree.corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                              0.5f*(r_min.y + r_max.y) - 0.5f*size,
                              0.5f*(r_min.z + r_max.z) - 0.5f*size,
                              size/(1 << MAXLEVELS));

   if(updateBoundaries)
     execStream->sync(); //This one has to be finished when we start updating the domain
                         //as it contains the keys on which we sample to update boundaries

   //Compute keys again, needed for the redistribution
   //Note we can call this in parallel with the computation of the domain.
   //This is done on predicted positions, to make sure that particles AFTER
   //prediction are separated by boundaries
   build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
   build_key_list.set_arg<real4>(3,    &tree.corner);
   build_key_list.execute(execStream->s());

   if(updateBoundaries)
   {
     exchangeSamplesAndUpdateBoundarySFC(NULL, 0, NULL,
                                         NULL,  NULL, 0,
                                         &tree.parallelBoundaries[0], lastExecTime,
                                         initialSetup);
   }


    domComp = get_time()-t0;
    char buff5[1024];
    sprintf(buff5,"EXCHANGEA-%d: tUpdateBoundaries: %lg\n", procId,  domComp);
    devContext.writeLogEvent(buff5);

    //Boundaries computed, now echange the particles
    LOGF(stderr, "Computing, exchanging and recompute of domain boundaries took: %f \n",domComp);
    t0 = get_time();
    gpuRedistributeParticles_SFC(&tree.parallelBoundaries[0]); //Redistribute the particles
    domExch = get_time()-t0;

    LOGF(stderr, "Redistribute domain took: %f\n", get_time()-t0);

  /*************************/

}


