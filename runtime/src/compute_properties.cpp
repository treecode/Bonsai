#include "octree.h"



void octree::compute_properties(tree_structure &tree) {
  /*****************************************************          
    Assign the memory buffers, note that we check the size first
    and if needed we increase the size of the generalBuffer1
    Size required:
      - multipoleD -> double4*3_n_nodes -> 6*n_nodes*uint4 
      - lower/upperbounds ->               2*n_nodes*uint4
      - node lower/upper  ->               2*n_nodes*uint4
      - SUM: 10*n_nodes*uint4 
      - generalBuffer1 has default size: 3*N*uint4
      
    check if 10*n_nodes < 3*N if so increase buffer size
    
   *****************************************************/
  devContext->startTiming(execStream->s());
  
  if(10*tree.n_nodes > 3*tree.n)
  {
    LOG("Resize generalBuffer1 in compute_properties\n");
    tree.generalBuffer1.cresize(10*tree.n_nodes*4, false);
  }
  
  my_dev::dev_mem<double4> multipoleD;      //Double precision buffer to store temporary results
  my_dev::dev_mem<real4>   nodeLowerBounds; //Lower bounds used for computing box sizes
  my_dev::dev_mem<real4>   nodeUpperBounds; //Upper bounds used for computing box sizes
  
  int memBufOffset = multipoleD.cmalloc_copy     (tree.generalBuffer1, 3*tree.n_nodes, 0);
      memBufOffset = nodeLowerBounds.cmalloc_copy(tree.generalBuffer1,   tree.n_nodes, memBufOffset);
      memBufOffset = nodeUpperBounds.cmalloc_copy(tree.generalBuffer1,   tree.n_nodes, memBufOffset);

  double t0 = get_time();
  this->resetCompact(); //Make sure compact has been reset, for setActiveGrp later on

  //Set the group properties
  setPHGroupData.set_args(0, &tree.n_groups, &tree.n, tree.bodies_Ppos.p(), tree.group_list.p(),
                             tree.groupCenterInfo.p(),tree.groupSizeInfo.p());
  const int nthreads = std::max(32,NCRIT);
  setPHGroupData.setWork(-1, nthreads, tree.n_groups);
  setPHGroupData.execute2(copyStream->s());

  //Set valid list to zero to reset the active particles
  tree.activeGrpList.zeroMemGPUAsync(execStream->s());

  setActiveGrps.set_args(0, &tree.n, &t_current, tree.bodies_time.p(), tree.body2group_list.p(), tree.activeGrpList.p());
  setActiveGrps.setWork(tree.n, 128);
  setActiveGrps.execute2(execStream->s());


  //Compact the valid list to get a list of valid groups
  gpuCompact(tree.activeGrpList, tree.active_group_list,
             tree.n_groups, &tree.n_active_groups);

//  this->resetCompact();
  LOG("t_previous: %lg t_current: %lg dt: %lg Active groups: %d (Total: %d)\n",
       t_previous, t_current, t_current-t_previous, tree.n_active_groups, tree.n_groups);

  double tA = get_time();

  //Density, compute h_min
  float sizex = rMaxGlobal.x - rMinGlobal.x;
  float sizey = rMaxGlobal.y - rMinGlobal.y;
  float sizez = rMaxGlobal.z - rMinGlobal.z;
  float sizeM = max(max(sizex,sizey), sizez);
  float h_min = sizeM / (powf(2.0,tree.n_levels));
  //End density


  //Computes the tree-properties (size, cm, monopole, quadrupole, etc)

  //start the kernel for the leaf-type nodes
  propsLeafD.set_args(0, &tree.n_leafs, tree.leafNodeIdx.p(), tree.node_bodies.p(), tree.bodies_Ppos.p(),
                         multipoleD.p(), nodeLowerBounds.p(), nodeUpperBounds.p(),
                         tree.bodies_Pvel.p(),  //Velocity to get max eps
                         tree.bodies_ids.p(),   //Ids to distinguish DM and stars
                         tree.bodies_h.p(),     //Density search radius
                         &h_min,                //minimum size of search radius)
                         tree.bodies_dens.p()); //Density to get max smoothing value
  propsLeafD.setWork(tree.n_leafs, 128);
  LOG("PropsLeaf: on number of leaves: %d \n", tree.n_leafs);
  propsLeafD.execute2(execStream->s());

   
  
  int curLevel = tree.n_levels;
  propsNonLeafD.set_args(0, &curLevel, tree.leafNodeIdx.p(), tree.node_level_list.p(), tree.n_children.p(), multipoleD.p(),
                         nodeLowerBounds.p(), nodeUpperBounds.p());

  //Work from the bottom up
  for(curLevel=tree.n_levels; curLevel >= 1; curLevel--)
  {   
    int totalOnThisLevel = tree.node_level_list[curLevel]-tree.node_level_list[curLevel-1];
    propsNonLeafD.setWork(totalOnThisLevel, 128);
    propsNonLeafD.execute2(execStream->s());

    LOG("PropsNonLeaf, nodes on level %d : %d (start: %d end: %d)\t\n",
            curLevel, totalOnThisLevel,tree.node_level_list[curLevel-1], tree.node_level_list[curLevel]);
  }
  
  propsScalingD.set_args(0, &tree.n_nodes, multipoleD.p(), nodeLowerBounds.p(), nodeUpperBounds.p(),
                            tree.n_children.p(), tree.multipole.p(), &theta, tree.boxSizeInfo.p(),
                            tree.boxCenterInfo.p(), tree.node_bodies.p());
  propsScalingD.setWork(tree.n_nodes, 128);
  LOG("propsScaling: on number of nodes: %d \n", tree.n_nodes); // propsScalingD.printWorkSize();
  propsScalingD.execute2(execStream->s());

  #ifdef INDSOFT
    //If we use individual softening we need to get the max softening value
    //to be broadcasted during the exchange of the LET boundaries.
    //Only copy the root node that contains the max value
    my_dev::dev_stream memCpyStream;
    tree.multipole.d2h(3, false, memCpyStream.s());
  #endif

  //Keep this sync for now since otherwise we run the risk that memory objects are destroyed
  //while still being in use (like multipoleD).
  double t1 = get_time();
  execStream->sync();
  LOGF(stderr, "Compute properties took: %lg  wait: %lg \n", get_time()-t0, get_time()-t1);


  //this->dumpTreeStructureToFile(tree);

   devContext->stopTiming("Compute-properties", 3, execStream->s());
} //compute_propertiesD


