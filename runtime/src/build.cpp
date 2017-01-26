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

  //SPH related, TODO check which one need pinnend and which do not
  tree.bodies_grad.     cmalloc(n_bodies, true);
  tree.bodies_dens.     cmalloc(n_bodies, true);
  tree.bodies_dens_out. cmalloc(n_bodies, true);
  tree.bodies_h.        cmalloc(n_bodies, true);
  tree.bodies_hydro.    cmalloc(n_bodies, true);
  tree.bodies_hydro_out.cmalloc(n_bodies, true);


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
  if(NLEAF < 16) tempmem *= 2;
  if(tree.n < 1024)
    tempmem = 2048;

  tree.n_children.cmalloc (tempmem, false);
  tree.node_bodies.cmalloc(tempmem, false);

  //General memory buffers

  //Allocate shared buffers
  this->tnext.		  ccalloc(NBLOCK_REDUCE,false);
  this->nactive.	  ccalloc(NBLOCK_REDUCE,false);
  this->devMemRMIN.   cmalloc(NBLOCK_BOUNDARY, false);
  this->devMemRMAX.	  cmalloc(NBLOCK_BOUNDARY, false);
  this->devMemCounts. cmalloc(NBLOCK_PREFIX, false);
  this->devMemCountsx.cmalloc(NBLOCK_PREFIX, true);

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
  
  //SPH Related
  tree.bodies_grad.     cresize(n_bodies, reduce);
  tree.bodies_dens.     cresize(n_bodies, reduce);
  tree.bodies_dens_out. cresize(n_bodies, reduce);
  tree.bodies_h.        cresize(n_bodies, reduce);
  tree.bodies_hydro.    cresize(n_bodies, reduce);
  tree.bodies_hydro_out.cresize(n_bodies, reduce);
  
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
  devContext->startTiming(execStream->s());
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
    tree.boxSmoothing.cresize_nocpy (n_nodes,   false); //host allocated

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
    tree.boxSmoothing.cmalloc (n_nodes,   true);
  }
  devContext->stopTiming("Memory", 11, execStream->s());
}


int octree::gpuDetermineBoundary(tree_structure &tree, int maxDepth, uint2 node_begend,
                                  my_dev::dev_mem<uint >  &validList, my_dev::dev_mem<uint >  &stackList,
                                  my_dev::dev_mem<int4>  &newValid,   my_dev::dev_mem<int4>  &finalValid,
                                  my_dev::dev_mem<float4> &finalBuff)
{
  //Determine the boundaries of the tree-structure. This is used during multi-GPU execution.
   gpuBoundaryTree.set_args(0, &node_begend,
                               &maxDepth,
                               stackList.p(),
                               tree.n_children.p(),
                               tree.node_bodies.p(),
                               validList.p(),
                               newValid.p());
   gpuBoundaryTree.setWork(32, 32, 1);
   gpuBoundaryTree.execute2(execStream->s());

   newValid.d2h(1);

   //Allocate the buffers to store in the boundary tree and the indices used to generate the boundary tree
   finalValid.resizeOrAlloc(1+newValid[0].x, false, false);
   finalValid.copy_devonly(newValid, 1+newValid[0].x);

   int finalTreeSize = 1 + newValid[0].y + 6*newValid[0].x; //Header+nBody+nNode*6 (size/cnt/smth/multi123)

   finalBuff.resizeOrAlloc(finalTreeSize, false, true);    //BoundaryTree goes to host so pinned memory

   LOGF(stderr,"Boundary props from GPU extraction nNode: %d nBody: %d start: %d end: %d\n", newValid[0].x,newValid[0].y, newValid[0].z, newValid[0].w);

   return finalTreeSize;
}


void octree::build (tree_structure &tree) {

  devContext->startTiming(execStream->s());
  int level      = 0;
  int validCount = 0;
  int offset     = 0;

  this->resetCompact();

  /******** create memory buffers **********/

  my_dev::dev_mem<uint>   validList;
  my_dev::dev_mem<uint>   compactList;
  my_dev::dev_mem<uint>   levelOffset;
  my_dev::dev_mem<uint>   maxLevel;
  my_dev::dev_mem<uint4>  node_key;



  int memBufOffset = validList.cmalloc_copy  (tree.generalBuffer1, tree.n*2, 0);
      memBufOffset = compactList.cmalloc_copy(tree.generalBuffer1, tree.n*2, memBufOffset);
  int memBufOffsetValidList = memBufOffset;

  const int factor = (NLEAF < 16) ? 2 : 1;
  int tempmem = std::max(2048, tree.n*factor); //Some default size in case tree.n is small

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



  build_valid_list.set_args(0, &tree.n, &level, tree.bodies_key.p(),  validList.p(), this->devMemCountsx.p());
  build_valid_list.setWork(tree.n, 128);

  build_nodes.set_args(0, &level, devMemCountsx.p(),  levelOffset.p(), maxLevel.p(),
                       tree.level_list.p(), compactList.p(), tree.bodies_key.p(), node_key.p(),
                       tree.n_children.p(), tree.node_bodies.p());
  build_nodes.setWork(vector<size_t>{120*32,4}, vector<size_t>{128,1});


  /******  build the levels *********/
  // make sure previous resetCompact() has finished.
  this->devMemCountsx.waitForCopyEvent();
//  devContext.startTiming(execStream->s());

 // if(nProcs > 1)
  {
    //Start copying the particle positions to the host, will overlap with tree-construction
    localTree.bodies_Ppos.d2h(tree.n, false, LETDataToHostStream->s());
  }

//  double tBuild0 = get_time();

#if 0
  build_tree_node_levels(*this, validList, compactList, levelOffset, maxLevel, execStream->s());
#else
  for (level = 0; level < MAXLEVELS; level++) {
    build_valid_list.execute2(execStream->s());         //Mark bodies to be combined into nodes
    gpuCompact(validList, compactList, tree.n*2, 0);    //Retrieve the number of created nodes
    build_nodes.execute2(execStream->s());              //Assemble the nodes
  } //end for level

  // reset counts to 1 so next compact proceeds...
  this->resetCompact();
#endif

//  execStream->sync();
//  const double dt = get_time() - tBuild0;
//  fprintf(stderr, " done in %g sec : %g Mptcl/sec\n", dt, tree.n/1e6/dt);
//  devContext.stopTiming("Create-nodes", 10, execStream->s());

  maxLevel.d2h(1);      level  = maxLevel[0];
  levelOffset.d2h(1);   offset = levelOffset[0];

  /***** Link the tree ******/



  //The maximum number of levels that can be used is MAXLEVEl
  LOG("Tree built with %d levels\n", level);
  if(level >= MAXLEVELS)
  {
    std::cerr << "The tree has become too deep, the program will exit. \n";
    std::cerr << "Consider the removal of far away particles to prevent a too large box. \n";
    exit(0);
  }

  tree.n_nodes       = offset;
  tree.n_levels      = level-1;
  tree.startLevelMin = 0;
  tree.level_list.d2h();
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

  link_tree.set_args(0, &offset,  tree.n_children.p(), tree.node_bodies.p(), tree.bodies_Ppos.p(), &tree.corner,
                        tree.level_list.p(), validList.p(), node_key.p(), tree.bodies_key.p(), &tree.startLevelMin);
  link_tree.setWork(tree.n_nodes , 128);
  link_tree.execute2(execStream->s());

  //After executing link_tree, the id_list contains for each node the ID of its parent.
  //Valid_list contains for each node if its a leaf (valid) or a normal node -> non_valid
  //Execute a split on the validList to get separate id lists
  //for the leafs and nodes. Used when computing multipole expansions

  if(tree.leafNodeIdx.get_size() > 0) tree.leafNodeIdx.cresize_nocpy(tree.n_nodes, false);
  else                                tree.leafNodeIdx.cmalloc      (tree.n_nodes , false);

  //Split the leaf nodes and non-leaf nodes
  gpuSplit(validList, tree.leafNodeIdx, tree.n_nodes, &tree.n_leafs);

  LOG("Total nodes: %d N_leafs: %d  non-leafs: %d \n", tree.n_nodes, tree.n_leafs, tree.n_nodes - tree.n_leafs);


  //For multi-GPU execution we now determine the outer boundaries of this tree-structure
//  if(nProcs > 1)
  {
      my_dev::dev_mem<int4>   newValid;
      my_dev::dev_mem<uint>   stackList;

      memBufOffset = stackList.cmalloc_copy(tree.generalBuffer1, 32*1024,      memBufOffsetValidList);
      memBufOffset = newValid.cmalloc_copy (tree.generalBuffer1, tree.n_nodes, memBufOffset);

      //First do this for the smallBoundary tree
      int4 startEndDepth       = getSearchPropertiesBoundaryTrees();
      uint2 node_begend        = make_uint2(startEndDepth.x, startEndDepth.y);
      boundaryTreeDimensions.x = gpuDetermineBoundary(tree,startEndDepth.z, node_begend, validList, stackList, newValid, tree.smallBoundaryTreeIndices, tree.smallBoundaryTree);

      //And now for the fullBoundary tree, which has no limitations on start and end
      node_begend              = tree.level_list[tree.startLevelMin];
      boundaryTreeDimensions.y = gpuDetermineBoundary(tree,99, node_begend, validList, stackList, newValid, tree.fullBoundaryTreeIndices, tree.fullBoundaryTree);
  }



  //Build the level list based on the leafIdx list, required for easy
  //access during the compute node properties / multipole computation
  build_level_list.set_args(0, &tree.n_nodes,  &tree.n_leafs, tree.leafNodeIdx.p(), tree.node_bodies.p(), validList.p());
  build_level_list.setWork(tree.n_nodes-tree.n_leafs, 128);
  validList.zeroMemGPUAsync(execStream->s());
  build_level_list.execute2(execStream->s());

  //Compact the node-level boundaries into the node_level_list
  gpuCompact(validList, tree.node_level_list, 2*(tree.n_nodes-tree.n_leafs), 0);

  /************   Start building the particle groups   *************

      We use the minimum tree-level to set extra boundaries, which ensures
      that groups are based on the tree-structure and will not be very big

      The previous computed offsets are used to build all boundaries.
      The ones based on top level boundaries and the group ones (every NCRIT particles)
  */

  validList.zeroMemGPUAsync(execStream->s());

  define_groups.set_args(0, &tree.n, validList.p(), &tree.level_list[tree.startLevelMin+1], tree.node_bodies.p(),
                            tree.node_level_list.p(), &level);
  define_groups.setWork(tree.n, 128);
  define_groups.execute2(execStream->s());

  //Copy the node_level_list back to host since we need it to compute the tree properties
  LOG("Finished level list \n");
  tree.node_level_list.d2h();
  for(int i=0; i < level; i++)
  {
    LOG("node_level_list: %d \t%d\n", i, tree.node_level_list[i]);
  }

  //Compact the validList to get the list of group IDs
  gpuCompact(validList, compactList, tree.n*2, &validCount);
  this->resetCompact();
  tree.n_groups = validCount/2;
  LOG("Found number of groups: %d \n", tree.n_groups);

  if(tree.group_list.get_size() > 0) tree.group_list.cresize_nocpy(tree.n_groups, false);
  else                               tree.group_list.cmalloc      (tree.n_groups, false);


  store_groups.set_args(0, &tree.n, &tree.n_groups, compactList.p(), tree.body2group_list.p(), tree.group_list.p());
  store_groups.setWork(-1, NCRIT, tree.n_groups);
  store_groups.execute2(execStream->s());


  //Memory allocation for the valid group lists
  //TODO get rid of this if by calling cresize when cmalloc is already called from inside the cmalloc call
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
  devContext->stopTiming("Tree-construction", 2, execStream->s());

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

  build_key_list.set_args(0, tree.bodies_key.p(), tree.bodies_pos.p(), &tree.n, &tree.corner);
  if(updateBoundaries)
  {
    //Build keys on current positions, since those are already sorted, while predicted are not
    build_key_list.set_args(0, tree.bodies_key.p(), tree.bodies_pos.p(), &tree.n, &tree.corner);
    build_key_list.setWork(tree.n, 128);
    build_key_list.execute2(execStream->s());

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
   build_key_list.reset_arg(1,   tree.bodies_Ppos.p());
   build_key_list.execute2(execStream->s());

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
    devContext->writeLogEvent(buff5);

    //Boundaries computed, now exchange the particles
    LOGF(stderr, "Computing, exchanging and recompute of domain boundaries took: %f \n",domComp);
    t0 = get_time();
    gpuRedistributeParticles_SFC(&tree.parallelBoundaries[0]); //Redistribute the particles
    domExch = get_time()-t0;

    LOGF(stderr, "Redistribute domain took: %f\n", get_time()-t0);

  /*************************/

}


