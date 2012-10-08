#include "octree.h"
#include "build.h"

void octree::allocateParticleMemory(tree_structure &tree)
{
  //Allocates the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;
  

  
  if(nProcs > 1)                //10% extra space, only in parallel when
    n_bodies = (int)(n_bodies*1.1f);    //number of particles can fluctuate
  
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

  tree.oriParticleOrder.cmalloc(n_bodies, false);      //To desort the bodies tree later on
  //iteration properties / information
  tree.activePartlist.ccalloc(n_bodies+2, false);   //+2 since we use the last two values as a atomicCounter (for grp count and semaphore access)
  tree.ngb.ccalloc(n_bodies, false);  
  tree.interactions.cmalloc(n_bodies, false);
  
  tree.body2group_list.cmalloc(n_bodies, false);
  
  tree.level_list.cmalloc(MAXLEVELS);  
  tree.node_level_list.cmalloc(MAXLEVELS*2 , false);    
  

  //The generalBuffer is also used during the tree-walk, so the size has to be at least
  //large enough to store the tree-walk stack. Add 4096 for extra memory allignment space
  //Times 2 since first half is used for regular walks, 2nd half for walks that go outside 
  //the memory stack and require another walk with 'unlimited' stack size
#if 0
  int treeWalkStackSize = (2*LMEM_STACK_SIZE*NTHREAD*nBlocksForTreeWalk) + 4096;
#else
  int treeWalkStackSize = (2*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*nBlocksForTreeWalk) + 4096;
#endif
    

  int tempSize   = max(n_bodies, 4096);   //Use minium of 4096 to prevent offsets mess up with small N
  tempSize       = 3*tempSize *4 + 4096;  //Add 4096 to give some space for memory allignment  
  tempSize = max(tempSize, treeWalkStackSize);
  
  //General buffer is used at multiple locations and reused in different functions
  // MJH for some reason this is crashing if pinned, when running on Fermi)
 // if (this->getDevContext()->getComputeCapability() < 300)
 //   tree.generalBuffer1.cmalloc(tempSize, false);
 // else
  //TODO JB look into the above
    tree.generalBuffer1.cmalloc(tempSize, true);  
  
  
  
  #ifdef USE_B40C
    sorter = new Sort90(n_bodies, tree.generalBuffer1.d());
//     sorter = new Sort90(n_bodies);
  #endif  

  //Tree properties, tree size is not known at forehand so
  //allocate worst possible outcome  
  n_bodies = n_bodies / 1;
  tree.n_children.cmalloc(n_bodies, false);
  tree.node_bodies.cmalloc(n_bodies, false);  
  
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
//    int remoteSize = (n_bodies*0.1) +  (n_bodies*0.1); //TODO some more realistic number
    int remoteSize = (int)(n_bodies*0.5); //TODO some more realistic number
    this->remoteTree.fullRemoteTree.cmalloc(remoteSize, true);    

    tree.parallelBoundaries.cmalloc(mpiGetNProcs()+1, true);
    //Some default value for number of hashes, will be increased if required
    //should be ' n_bodies / NPARALLEL' for now just alloc 10%
    tree.parallelHashes.cmalloc(n_bodies*0.1, true);
  }
  
}


void octree::reallocateParticleMemory(tree_structure &tree)
{
  //Realloc the memory to hold the particles data
  //and the arrays that have the same size as there are
  //particles. Eg valid arrays used in tree construction
  int n_bodies = tree.n;
  

  
  
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
  
  tree.oriParticleOrder.cresize(n_bodies, reduce);     //To desort the bodies tree later on         
  //iteration properties / information
  tree.activePartlist.cresize(n_bodies+2, reduce);      //+1 since we use the last value as a atomicCounter
  tree.ngb.cresize(n_bodies, reduce);  
  tree.interactions.cresize(n_bodies, reduce);
  
  tree.body2group_list.cresize(n_bodies, reduce);  

  //Tree properties, tree size is not known at forehand so
  //allocate worst possible outcome  
  n_bodies = n_bodies / 1;
  tree.n_children.cresize(n_bodies, reduce);
  tree.node_bodies.cresize(n_bodies, reduce);  
  
  
  //Dont forget to resize the generalBuffer....
#if 0
  int treeWalkStackSize = (2*LMEM_STACK_SIZE*NTHREAD*nBlocksForTreeWalk) + 4096;
#else
  int treeWalkStackSize = (2*(LMEM_STACK_SIZE*NTHREAD + LMEM_EXTRA_SIZE)*nBlocksForTreeWalk) + 4096;
#endif
    
  int tempSize   = max(n_bodies, 4096);   //Use minium of 4096 to prevent offsets mess up with small N
  tempSize       = 3*tempSize *4 + 4096;  //Add 4096 to give some space for memory allignment  
  tempSize = max(tempSize, treeWalkStackSize);
  
  //General buffer is used at multiple locations and reused in different functions
  tree.generalBuffer1.cresize(tempSize, reduce);    
  
  #ifdef USE_B40C
    delete sorter;
    sorter = new Sort90(n_bodies, tree.generalBuffer1.d());
//      sorter = new Sort90(n_bodies);
  #endif
        
  
  my_dev::base_mem::printMemUsage();
}

void octree::allocateTreePropMemory(tree_structure &tree)
{ 
  int n_nodes = tree.n_nodes;

  //Allocate memory
  if(tree.groupCenterInfo.get_size() > 0)
  {
    n_nodes = (int)(n_nodes * 1.1f);
    //Resize, so we dont alloc if we already have mem alloced
    tree.multipole.cresize_nocpy(3*n_nodes,     false);
    
    tree.boxSizeInfo.cresize_nocpy(n_nodes,     false);  //host alloced
    tree.groupSizeInfo.cresize_nocpy(tree.n_groups,   false);

    tree.boxCenterInfo.cresize_nocpy(n_nodes,   false); //host alloced
    tree.groupCenterInfo.cresize_nocpy(tree.n_groups, false);
  }
  else
  {    
    //TODO only host alloc if nProcs > 1
    n_nodes = (int)(n_nodes * 1.1f); 
    tree.multipole.cmalloc(3*n_nodes, true); //host alloced
        
    tree.boxSizeInfo.cmalloc(n_nodes, true);     //host alloced
    tree.groupSizeInfo.cmalloc(tree.n_groups, true);

    tree.boxCenterInfo.cmalloc(n_nodes, true); //host alloced
    tree.groupCenterInfo.cmalloc(tree.n_groups,true);
  }
}

void octree::build (tree_structure &tree) {

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
      memBufOffset = node_key.cmalloc_copy   (tree.generalBuffer1, tree.n,   memBufOffset);
      memBufOffset = levelOffset.cmalloc_copy(tree.generalBuffer1, 256,      memBufOffset);
      memBufOffset = maxLevel.cmalloc_copy   (tree.generalBuffer1, 256,      memBufOffset);

      //Memory layout of the above (in uint):
      //[[validList--2*tree.n],[compactList--2*tree.n],[node_key--4*tree.n],
      // [levelOffset--256], [maxLevel--256], [free--at-least: 12-8*tree.n-256]]

  //Set the default values to zero
  validList.zeroMemGPUAsync(execStream->s());
  levelOffset.zeroMemGPUAsync(execStream->s());
//  maxLevel.zeroMemGPUAsync(execStream->s());

  
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


  /********** build  list of keys ********/
  
//   build_key_list.execute();  
  
  /******  build the levels *********/

  // make sure previous resetCompact() has finished.
  this->devMemCountsx.waitForCopyEvent();


#if 0
  build_tree_node_levels(*this, validList, compactList, levelOffset, maxLevel, execStream->s());
#else
  //int nodeSum = 0;


  for (level = 0; level < MAXLEVELS; level++) {
    // mark bodies to be combined into nodes
    build_valid_list.set_arg<int>(1, &level);
    build_valid_list.execute(execStream->s());
      
    //gpuCompact to get number of created nodes    
    gpuCompact(devContext, validList, compactList, tree.n*2, 0);
                   
    // assemble nodes           
    build_nodes.set_arg<int>(0, &level);
    build_nodes.execute(execStream->s());
  } //end for lvl


  // reset counts to 1 so next compact proceeds...
  this->resetCompact();
#endif
  


  maxLevel.d2h(1);
  level = maxLevel[0];

  levelOffset.d2h(1);
  offset = levelOffset[0];
    
  int n_nodes  = offset;
  tree.n_nodes = n_nodes;
 
 
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
  
  link_tree.setWork(n_nodes, 128);
  LOG("Link_tree: "); link_tree.printWorkSize();
  
  tree.n_levels = level-1;

  tree.level_list.d2h();
  tree.startLevelMin = 0;
  for(int i=0; i < level; i++)
  {
    LOG("%d\t%d\t%d\n", i, tree.level_list[i].x, tree.level_list[i].y);
    //Figure out which level is used as min_level
    if(((tree.level_list[i].y - tree.level_list[i].x) > START_LEVEL_MIN_NODES) && (tree.startLevelMin == 0))
    {
      tree.startLevelMin = i;
    }
  }
  LOG("Start at: Level: %d  begend: %d %d \n", tree.startLevelMin,
      tree.level_list[tree.startLevelMin].x, tree.level_list[tree.startLevelMin].y);
 

  //Link the tree      
  link_tree.set_arg<int>(0, &offset);   //Offset=number of nodes
  link_tree.set_arg<int>(9, &tree.startLevelMin);
  link_tree.execute(execStream->s());
  


  //After executing link_tree, the id_list contains for each node the ID of its parent.
  //Valid_list contains for each node if its a leaf (valid) or a normal node -> non_valid
  //Execute a split on the validList to get separate id lists
  //for the leafs and nodes. Used when computing multipoles
    
  tree.leafNodeIdx.cmalloc(tree.n_nodes , false);
  
  //Split the leaf ids and non-leaf node ids
  gpuSplit(devContext, validList, tree.leafNodeIdx, tree.n_nodes, &tree.n_leafs);     
                 
  LOG("Total nodes: %d N_leafs: %d  non-leafs: %d \n", tree.n_nodes, tree.n_leafs, tree.n_nodes - tree.n_leafs);
  

  build_level_list.set_arg<int>(0, &tree.n_nodes);
  build_level_list.set_arg<int>(1, &tree.n_leafs);
  build_level_list.set_arg<cl_mem>(2, tree.leafNodeIdx.p());
  build_level_list.set_arg<cl_mem>(3, tree.node_bodies.p());
  build_level_list.set_arg<cl_mem>(4, validList.p());  
  build_level_list.setWork(tree.n_nodes-tree.n_leafs, 128);
  
  validList.zeroMemGPUAsync(execStream->s());

  //Build the level list based on the leafIdx list
  //required for easy access in the compute node properties
  build_level_list.execute(execStream->s());  

  int levelThing;  
//  gpuCompact(devContext, validList, tree.node_level_list,
//             2*(tree.n_nodes-tree.n_leafs), &levelThing);
  gpuCompact(devContext, validList, tree.node_level_list, 
             2*(tree.n_nodes-tree.n_leafs), 0);
  
  ///******   Start building the particle groups *******///////

  //Compute the box size, the max length of one of the sides of the rectangle
  real size     = std::max(fabs(rMaxLocalTree.z - rMinLocalTree.z), 
                           std::max(fabs(rMaxLocalTree.y - rMinLocalTree.y),
                                    fabs(rMaxLocalTree.x - rMinLocalTree.x)));
  real dist     = ((rMaxLocalTree.z - rMinLocalTree.z) * (rMaxLocalTree.z - rMinLocalTree.z) + 
                   (rMaxLocalTree.y - rMinLocalTree.y) * (rMaxLocalTree.y - rMinLocalTree.y) +
                   (rMaxLocalTree.x - rMinLocalTree.x) * (rMaxLocalTree.x - rMinLocalTree.x));      
                   
//  float maxDist = sqrt(dist) / 10;
//  maxDist *= maxDist; //Square since we dont do sqrt on device
//
//  LOG("Box max size: %f en max dist: %f \t %f en %f  \n", size, dist, sqrt(dist), maxDist);
  
  //The coarse group boundaries are based on the tree-structure. These coarse boundaries
  //will be used for getting the multi-box/group group boundaries. Also it serves as a 
  //boundary mechanism independent of the max-distance trick between particles that we used 
  //before. So first find the min level from where we can take the boundaries
//  const int minCoarseGroups = 50;
//  int minCoarseGroupLevelIdx = 0;
//  for(int i=0; i < level; i++){
//    if( (tree.level_list[i].y - tree.level_list[i].x) >  minCoarseGroups)
//    {
//      minCoarseGroupLevelIdx = i;
//      break;
//    }
//  }
//  tree.courseGroupIdx = minCoarseGroupLevelIdx;

  //In the updated version we use the minimum tree-level, which ensures
  //that groups are based on the tree-structure. The +1
  tree.courseGroupIdx        = tree.startLevelMin ;
  int minCoarseGroupLevelIdx = tree.startLevelMin ;
  
  
  //Now use the previous computed offsets to build all boundaries. The coarse ones and the
  //group ones (eg every number of particles)
  validList.zeroMemGPUAsync(execStream->s());
  //The newest group creation method!
  define_groups.set_arg<int>    (0, &tree.n);  
  define_groups.set_arg<cl_mem> (1, validList.p());    
  define_groups.set_arg<uint2>  (2, &tree.level_list[minCoarseGroupLevelIdx]);
  define_groups.set_arg<cl_mem> (3, tree.node_bodies.p());
  define_groups.set_arg<cl_mem> (4, tree.node_level_list.p());
  define_groups.set_arg<int>    (5, &level);
  define_groups.setWork(tree.n, 128);  
  define_groups.execute(execStream->s());
  
   
  //Have to copy it back to host since we need it in compute props
  LOG("Finished level list \n");
  tree.node_level_list.d2h();
  for(int i=0; i < (level); i++)
  {
    LOG("node_level_list: %d \t%d\n", i, tree.node_level_list[i]);
  }
  
  //Now compact validList to get the list of group ids
  gpuCompact(devContext, validList, compactList, tree.n*2, &validCount);  
  this->resetCompact();
  tree.n_groups = validCount/2;
  LOG("Found number of groups: %d \n", tree.n_groups);
 

  //Mark the boundaries of the coarse groups. We have to do this again since it was combined
  //with all other groups in the previous step
  validList.zeroMemGPUAsync(execStream->s());
  mark_coarse_group_boundaries.set_arg<uint2> (0, &tree.level_list[minCoarseGroupLevelIdx]);
  mark_coarse_group_boundaries.set_arg<cl_mem>(1, validList.p());
  mark_coarse_group_boundaries.set_arg<cl_mem>(2, tree.node_bodies.p());
  mark_coarse_group_boundaries.set_arg<cl_mem>(3, tree.node_level_list.p());
  mark_coarse_group_boundaries.setWork(tree.level_list[minCoarseGroupLevelIdx].y, 128);  
  mark_coarse_group_boundaries.execute(execStream->s());

  //Store the group list but also compact the coarse group boundaries
  my_dev::dev_mem<uint>  coarseGroupValidList(devContext);

  memBufOffset = coarseGroupValidList. cmalloc_copy(tree.generalBuffer1, 
                                                    tree.n_groups, memBufOffsetValidList);

  tree.coarseGroupCompact.cmalloc(tree.level_list[minCoarseGroupLevelIdx].y, false);

  tree.group_list.cmalloc(tree.n_groups , false);  
  tree.n_active_groups = tree.n_groups; //Set all groups active in shared-time-step mode
 
  store_groups.set_arg<int>(0, &tree.n);  
  store_groups.set_arg<int>(1, &tree.n_groups);  
  store_groups.set_arg<cl_mem>(2, compactList.p());    
  store_groups.set_arg<cl_mem>(3, tree.body2group_list.p());     
  store_groups.set_arg<cl_mem>(4, tree.group_list.p());     
  store_groups.set_arg<cl_mem>(5, validList.p());  
  store_groups.set_arg<cl_mem>(6, coarseGroupValidList.p());
  store_groups.setWork(-1, NCRIT, tree.n_groups);  
  store_groups.execute(execStream->s());  

  //Now compact validList to get the list of group ids

  int n_course = 0;
  gpuCompact(devContext, coarseGroupValidList, tree.coarseGroupCompact, tree.n_groups, &n_course);
  this->resetCompact();
  LOGF(stderr, "Coarse groups %d \n", n_course);
  tree.n_coarse_groups = n_course;

#if 0
  tree.coarseGroupCompact.d2h();
  tree.group_list.d2h();


mpiSync();
if(procId == 0){
  for(int i=0; i < n_course; i++)
  {
	    int start = tree.group_list[tree.coarseGroupCompact[i]].x;
	    int end   = tree.group_list[tree.coarseGroupCompact[i]].y;

	    if(i == n_course-1){
	    	   LOGF(stderr,"Compactgrps Final %d \t %d \t %d\t%d\t Part: %d \t %d \n",
	    			   i,tree.coarseGroupCompact[i], tree.n_groups,
	    			   tree.n_groups - tree.coarseGroupCompact[i],
	    			   start, end);
	    }
	    else
	    {
	    int end2 = tree.group_list[tree.coarseGroupCompact[i+1]].x;


	    LOGF(stderr,"Compactgrps %d \t %d \t %d\t%d\t Part: %d \t %d \n",
				  i, tree.coarseGroupCompact[i],  tree.coarseGroupCompact[i+1],
				  tree.coarseGroupCompact[i+1] -  tree.coarseGroupCompact[i],
				  start, end2);
	    }
  }

  //Print tree level information
  tree.n_children.d2h();
  tree.node_bodies.d2h();
//  for(int i=0; i < 4; i++)
  {
    int i = tree.startLevelMin;
    for(int j=tree.level_list[i].x; j < tree.level_list[i].y; j++)
    {

      int childinfo = tree.n_children[j];
      uint2 bij     = tree.node_bodies[j];
      uint level = (bij.x &  LEVELMASK) >> BITLEVELS;
      uint bi    =  bij.x & ILEVELMASK;
      uint bj    =  bij.y;
      bool leaf = 0;
      const int LEVEL_MIN = 3;
      if ((int)level > (int)(LEVEL_MIN - 1))
        leaf = ((bj - bi) <= NLEAF);
      int child, nchild;
      if(!leaf)
      {
        //Node
        child    =    childinfo & 0x0FFFFFFF;                         //Index to the first child of the node
        nchild   = (((childinfo & 0xF0000000) >> 28)) ;         //The number of children this node has
      }
      else
      {
        //Leaf
        child   =   childinfo & BODYMASK;                                     //thre first body in the leaf
        nchild  = (((childinfo & INVBMASK) >> LEAFBIT)+1);     //number of bodies in the leaf masked with the flag
      }

      LOGF(stderr,"Tree node: %d @ level: %d From Particle; %d to %d  Child info: %d %d leaf: %d\n",
          j, i, bi, bj, child, nchild, leaf);
    }
  }

}

//  exit(0);
#endif

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
  
  /*************************/

}




//This function builds a hash-table for the particle-keys which is required for the
//domain distribution based on the SFC

void octree::parallelDataSummary(tree_structure &tree, float lastExecTime, float lastExecTime2) {

  int level      = 0;
  int validCount = 0;
  int offset     = 0;

  double t0 = get_time();
  /******** create memory buffers and reserve memory using the shared buffer **********/

  my_dev::dev_mem<uint>   validList(devContext);
  my_dev::dev_mem<uint>   compactList(devContext);
  my_dev::dev_mem<uint4>  parGrpBlockKey(devContext);
  my_dev::dev_mem<uint2>  parGrpBlockInfo(devContext);
  my_dev::dev_mem<uint>   startBoundaryIndex(devContext);
  my_dev::dev_mem<uint>   atomicValues(devContext);

  int memBufOffset = validList.cmalloc_copy  (tree.generalBuffer1, tree.n*2, 0);
      memBufOffset = compactList.cmalloc_copy(tree.generalBuffer1, tree.n*2, memBufOffset);
      memBufOffset = parGrpBlockKey.cmalloc_copy(tree.generalBuffer1, tree.n, memBufOffset);
      memBufOffset = parGrpBlockInfo.cmalloc_copy(tree.generalBuffer1, tree.n, memBufOffset);
      memBufOffset = startBoundaryIndex.cmalloc_copy(tree.generalBuffer1, tree.n+1, memBufOffset); //+1 to set final border
      memBufOffset = atomicValues.cmalloc_copy(tree.generalBuffer1, 256, memBufOffset); //+1 to set final border

  validList.zeroMemGPUAsync(copyStream->s());
  atomicValues.zeroMemGPUAsync(copyStream->s());
  startBoundaryIndex.zeroMemGPUAsync(copyStream->s());

  //Total amount of space in the generalbuffer: at least:
  //- 3x n_bodies*uint4/float4 , so 12*int size
  // validList          = 0-2*n_bodies (int)             Remaining: 10*n_bodies
  // compactList        = 2*n_bodies - 4*n_bodies (int)  Remaining:  8*n_bodies
  // parGrpBlockKey     = 4*n_bodies - 8*n_bodies (int)  Remaining:  4*n_bodies
  // parGrpBlockInfo     = 8*n_bodies - 10*n_bodies (int)  Remaining:  2*n_bodies
  // startBoundaryIndex = 10*n_bodies - 11*n_bodies+1 (int) Remianing: n_bodies-1


  real4 r_min = {+1e10, +1e10, +1e10, +1e10};
  real4 r_max = {-1e10, -1e10, -1e10, -1e10};

  /******** set kernels parameters **********/

  build_valid_list.set_arg<int>(0,     &tree.n);
  build_valid_list.set_arg<int>(1,     &level);
  build_valid_list.set_arg<cl_mem>(2,  tree.bodies_key.p());
  build_valid_list.set_arg<cl_mem>(3,  validList.p());
  build_valid_list.setWork(tree.n, 128);

  build_parallel_grps.set_arg<int>(0,     &validCount);
  build_parallel_grps.set_arg<int>(1,     &offset);
  build_parallel_grps.set_arg<cl_mem>(2,  compactList.p());
  build_parallel_grps.set_arg<cl_mem>(3,  tree.bodies_key.p());
  build_parallel_grps.set_arg<cl_mem>(4,  parGrpBlockKey.p());
  build_parallel_grps.set_arg<cl_mem>(5,  parGrpBlockInfo.p());
  build_parallel_grps.set_arg<cl_mem>(6,  startBoundaryIndex.p());


  /********** build  list of keys ********/

  getBoundaries(tree, r_min, r_max); //Used for predicted position keys further down
  LOGF(stderr, "BVfore hashes took: %f \n", get_time()-t0);


  //Compute keys based on the non-predicted positions to make sure
  //particles are sorted. Improves efficiency of hash creation (less hashes required)
  //and sorting (less items to sort and hashes are already in sorted order).

  build_key_list.set_arg<cl_mem>(0,   tree.bodies_key.p());
//  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_pos.p());
  build_key_list.set_arg<int>(2,      &tree.n);
  build_key_list.set_arg<real4>(3,    &tree.corner);
  build_key_list.setWork(tree.n, 128); //128 threads per block
  build_key_list.execute(execStream->s());

  /******  build the levels *********/

  int nodeSum = 0;

  this->resetCompact();
  this->devMemCountsx.waitForCopyEvent();
  copyStream->sync();


  //TODO rewrite this in similar way as build_tree loop, to remove copy
  //TODO change this in a way that information is extracted from the tree-structure
  //this way it should be must faster than it is now
  for (level = 0; level < MAXLEVELS; level++) {
    // mark bodies to be combined into nodes
    build_valid_list.set_arg<int>(1, &level);
    build_valid_list.execute(execStream->s());

    //gpuCompact to get number of created nodes
    gpuCompact(devContext, validList, compactList, tree.n*2, &validCount);

    nodeSum += validCount / 2;
    LOG("ValidCount (%d): %d \tSum: %d Offset: %d\n", 0, validCount, nodeSum, offset);

    validCount /= 2;

    if (validCount == 0) break;

    // Assemble nodes
    build_parallel_grps.set_arg<int>(0, &validCount);
    build_parallel_grps.set_arg<int>(1, &offset);
    build_parallel_grps.setWork(-1, 128, validCount);
    build_parallel_grps.execute(execStream->s());

    offset += validCount;
  } //end for level

  this->resetCompact();
  this->devMemCountsx.waitForCopyEvent();

  //Compact the starts of each block/group. Since this is based on the particle positions
  //this gives us a way to sort the hash-group-keys without having to actually sort the
  //list. This sorting/reordering will take place in the segmentedSummaryBasic kernel
  gpuCompact(devContext, startBoundaryIndex, compactList, tree.n+1, &validCount);

  LOGF(stderr,"Number of hash-blocks: %d , of which valid: %d\n", offset, validCount);

  //Get the properties: key+number of particles with that key, + possibly interaction count
  execStream->sync();
  tree.parallelHashes.cresize(validCount, false); //Possibly increase but never decrease

  segmentedSummaryBasic.set_arg<int>(0,     &validCount);
  segmentedSummaryBasic.set_arg<cl_mem>(1,  compactList.p());
  segmentedSummaryBasic.set_arg<cl_mem>(2,  atomicValues.p());
  segmentedSummaryBasic.set_arg<cl_mem>(3,  parGrpBlockInfo.p());
  segmentedSummaryBasic.set_arg<cl_mem>(4,  parGrpBlockKey.p());
  segmentedSummaryBasic.set_arg<cl_mem>(5,  tree.parallelHashes.p());
  segmentedSummaryBasic.set_arg<cl_mem>(6,  tree.bodies_key.p());
  //Fix number of groups to reduce launch overhead for small groups , persistent threads
  segmentedSummaryBasic.setWork(-1, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);
  segmentedSummaryBasic.execute(execStream->s());

  tree.parallelHashes.d2h(validCount, false, execStream->s());

  //Sync the boundary over the various processes
  if(this->mpiGetNProcs() > 1)
  {
    this->sendCurrentRadiusInfo(r_min, r_max);
  }

  //Compute the boundarys of the tree
  real size     = 1.001f*std::max(r_max.z - r_min.z,
                         std::max(r_max.y - r_min.y, r_max.x - r_min.x));

  tree.corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                             0.5f*(r_min.y + r_max.y) - 0.5f*size,
                             0.5f*(r_min.z + r_max.z) - 0.5f*size, size);
  tree.corner.w = size/(1 << MAXLEVELS);


  execStream->sync(); //Make sure segmentedSummaryBasic and d2h completed

  //Compute keys again, needed for the redistribution
  // Note we can call this in parallel with the communication of the hashes.
  //THis is done on predicted positions, to make sure that particles AFTER
  //prediction are separated by boundaries
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<real4>(3,    &tree.corner);
  build_key_list.execute(execStream->s());

  LOGF(stderr, "Compute hashes took: %f \n", get_time()-t0);

   //No leak in this call, checked. TODO clean up comment
  gpu_collect_hashes(validCount, &tree.parallelHashes[0], &tree.parallelBoundaries[0], 
		     lastExecTime, lastExecTime2);
  

  LOGF(stderr, "Computing and exchanging and update domain took: %f \n", get_time()-t0);
  t0 = get_time();

  //TODO IMPORTANT, make sure we get the corret particle
  //info wwhen we get the info since we do not sort all data
  //after predict, never mind we have not clled sort at this point
  //in execution

//  fflush(stderr);
//  fflush(stdout);
//  mpiSync();
//  tree.bodies_key.d2h();
//  for(int i=0; i < tree.n; i++)
//  {
//    if(procId == 0)
//    LOGF(stderr, "Particle at: %d\t%d %d %d %d \n",
//        i, tree.bodies_key[i].x,tree.bodies_key[i].y,
//           tree.bodies_key[i].z,tree.bodies_key[i].w);
//
//  }


   gpuRedistributeParticles_SFC(&tree.parallelBoundaries[0]);

   LOGF(stderr, "Redistribute domain took: %f \n", get_time()-t0);
//  tree.bodies_key.d2h();
//  for(int i=0; i < tree.n; i++)
//  {
//    if(procId == 1)
//    LOGF(stderr, "Particle at: %d\t%d %d %d %d \n",
//        i, tree.bodies_key[i].x,tree.bodies_key[i].y,
//           tree.bodies_key[i].z,tree.bodies_key[i].w);
//
//  }
//
//
//  mpiSync();
//
//fprintf(stderr,"TEST TEST TEST\n");
//exit(0);
  /*************************/

}


