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
  devContext.startTiming(execStream->s());
  
  if(10*tree.n_nodes > 3*tree.n)
  {
    LOG("Resize generalBuffer1 in compute_properties\n");
    tree.generalBuffer1.cresize(10*tree.n_nodes*4, false);
  }
  
  my_dev::dev_mem<double4> multipoleD;      //Double precision buffer to store temp results
  my_dev::dev_mem<real4>   nodeLowerBounds; //Lower bounds used for computing box sizes
  my_dev::dev_mem<real4>   nodeUpperBounds; //Upper bounds used for computing box sizes
  
  int memBufOffset = multipoleD.cmalloc_copy          (tree.generalBuffer1, 3*tree.n_nodes, 0);
      memBufOffset = nodeLowerBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);
      memBufOffset = nodeUpperBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);

  double t0 = get_time();
  this->resetCompact(); //Make sure compact has been reset, for setActiveGrp later on

  //Set the group properties
  setPHGroupData.set_arg<int>(0,    &tree.n_groups);
  setPHGroupData.set_arg<int>(1,    &tree.n);
  setPHGroupData.set_arg<cl_mem>(2, tree.bodies_Ppos.p());
  setPHGroupData.set_arg<cl_mem>(3, tree.group_list.p());
  setPHGroupData.set_arg<cl_mem>(4, tree.groupCenterInfo.p());
  setPHGroupData.set_arg<cl_mem>(5, tree.groupSizeInfo.p());
  setPHGroupData.setWork(-1, NCRIT, tree.n_groups);
  setPHGroupData.execute(copyStream->s());

  //Set valid list to zero to reset the active particles
  tree.activeGrpList.zeroMemGPUAsync(execStream->s());
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
  propsLeafD.set_arg<int>(0,    &tree.n_leafs);
  propsLeafD.set_arg<cl_mem>(1, tree.leafNodeIdx.p());
  propsLeafD.set_arg<cl_mem>(2, tree.node_bodies.p());
  propsLeafD.set_arg<cl_mem>(3, tree.bodies_Ppos.p());  
  propsLeafD.set_arg<cl_mem>(4, multipoleD.p());
  propsLeafD.set_arg<cl_mem>(5, nodeLowerBounds.p());
  propsLeafD.set_arg<cl_mem>(6, nodeUpperBounds.p());
  propsLeafD.set_arg<cl_mem>(7, tree.bodies_Pvel.p()); //Velocity to get max eps
  propsLeafD.set_arg<cl_mem>(8, tree.bodies_ids.p());  //Ids to distinguish DM and stars
  propsLeafD.set_arg<cl_mem>(9, tree.bodies_h.p());    //Density search radius
  propsLeafD.set_arg<float >(10, &h_min);              //minimum size of search radius
  propsLeafD.setWork(tree.n_leafs, 128);
  LOG("PropsLeaf: on number of leaves: %d \n", tree.n_leafs);
  propsLeafD.execute(execStream->s()); 
   
  
  int temp = tree.n_nodes-tree.n_leafs;
  propsNonLeafD.set_arg<int>(0,    &temp);
  propsNonLeafD.set_arg<cl_mem>(1, tree.leafNodeIdx.p());
  propsNonLeafD.set_arg<cl_mem>(2, tree.node_level_list.p());
  propsNonLeafD.set_arg<cl_mem>(3, tree.n_children.p());  
  propsNonLeafD.set_arg<cl_mem>(4, multipoleD.p());
  propsNonLeafD.set_arg<cl_mem>(5, nodeLowerBounds.p());
  propsNonLeafD.set_arg<cl_mem>(6, nodeUpperBounds.p());

  //Work from the bottom up
  for(int i=tree.n_levels; i >= 1; i--)
  {   
    int totalOnThisLevel = tree.node_level_list[i]-tree.node_level_list[i-1];
    propsNonLeafD.setWork(totalOnThisLevel, 128);
    propsNonLeafD.set_arg<int>(0,    &i); //set the level
    propsNonLeafD.execute(execStream->s());

    LOG("PropsNonLeaf, nodes on level %d : %d (start: %d end: %d)\t\n",
          i, totalOnThisLevel,tree.node_level_list[i-1], tree.node_level_list[i]);
  }
  
  propsScalingD.set_arg<int>(0,    &tree.n_nodes);
  propsScalingD.set_arg<cl_mem>(1, multipoleD.p());
  propsScalingD.set_arg<cl_mem>(2, nodeLowerBounds.p());
  propsScalingD.set_arg<cl_mem>(3, nodeUpperBounds.p());
  propsScalingD.set_arg<cl_mem>(4, tree.n_children.p());  
  propsScalingD.set_arg<cl_mem>(5, tree.multipole.p());
  propsScalingD.set_arg<float >(6, &theta);
  propsScalingD.set_arg<cl_mem>(7, tree.boxSizeInfo.p());
  propsScalingD.set_arg<cl_mem>(8, tree.boxCenterInfo.p());
  propsScalingD.set_arg<cl_mem>(9, tree.node_bodies.p());
  propsScalingD.setWork(tree.n_nodes, 128);
  LOG("propsScaling: on number of nodes: %d \n", tree.n_nodes); // propsScalingD.printWorkSize();
  propsScalingD.execute(execStream->s());   

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


#if 0


if(iter == 20)
{
   char fileName[256];
    sprintf(fileName, "groups-%d.bin", mpiGetRank());
    ofstream nodeFile;
    nodeFile.open(fileName, ios::out | ios::binary);
    if(nodeFile.is_open())
    {
      nodeFile.write((char*)&tree.n_groups, sizeof(int));

      for(int i=0; i < tree.n_groups; i++)
      {
        nodeFile.write((char*)&tree.groupSizeInfo[i],  sizeof(real4)); //size
        nodeFile.write((char*)&tree.groupCenterInfo[i], sizeof(real4)); //center
      }
    }
  }

 //Write the tree-structure
 if(iter == 20)
 {
   tree.multipole.d2h();
  tree.boxSizeInfo.d2h();
  tree.boxCenterInfo.d2h();
  tree.bodies_Ppos.d2h();

    char fileName[256];
    sprintf(fileName, "fullTreeStructure-%d.bin", mpiGetRank());
    ofstream nodeFile;
    //nodeFile.open(nodeFileName.c_str());
    nodeFile.open(fileName, ios::out | ios::binary);
    if(nodeFile.is_open())
    {
      uint2 node_begend;
      int level_start = tree.startLevelMin;
      node_begend.x   = tree.level_list[level_start].x;
      node_begend.y   = tree.level_list[level_start].y;

      nodeFile.write((char*)&node_begend.x, sizeof(int));
      nodeFile.write((char*)&node_begend.y, sizeof(int));
      nodeFile.write((char*)&tree.n_nodes, sizeof(int));
      nodeFile.write((char*)&tree.n, sizeof(int));

      for(int i=0; i < tree.n; i++)
      {
        nodeFile.write((char*)&tree.bodies_Ppos[i], sizeof(real4));
      }

      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.multipole[3*i+0], sizeof(real4));
        nodeFile.write((char*)&tree.multipole[3*i+1], sizeof(real4));
        nodeFile.write((char*)&tree.multipole[3*i+2], sizeof(real4));;
      }

      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.boxSizeInfo[i], sizeof(real4));
      }
      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.boxCenterInfo[i], sizeof(real4));
      }

      nodeFile.close();
    }
}
#endif

   devContext.stopTiming("Compute-properties", 3, execStream->s());
} //compute_propertiesD


