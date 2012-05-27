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
      
    check if 10*n_nodes < 3*N if so realloc
    
   *****************************************************/
  
  if(10*tree.n_nodes > 3*tree.n)
  {
    LOG("Resize generalBuffer1 in compute_properties\n");
    tree.generalBuffer1.cresize(10*tree.n_nodes*4, false);
  }
  
  my_dev::dev_mem<double4> multipoleD(devContext);
  
  int memBufOffset = multipoleD.cmalloc_copy          (tree.generalBuffer1, 3*tree.n_nodes, 0);
      memBufOffset = tree.nodeLowerBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);
      memBufOffset = tree.nodeUpperBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);  
  
  
  
  //Computes the tree-properties (size, cm, monopole, quadropole, etc)
  //start the kernel for the leaf-type nodes
  propsLeafD.set_arg<int>(0,    &tree.n_leafs);
  propsLeafD.set_arg<cl_mem>(1, tree.leafNodeIdx.p());
  propsLeafD.set_arg<cl_mem>(2, tree.node_bodies.p());
  propsLeafD.set_arg<cl_mem>(3, tree.bodies_Ppos.p());  
  propsLeafD.set_arg<cl_mem>(4, multipoleD.p());
  propsLeafD.set_arg<cl_mem>(5, tree.nodeLowerBounds.p());
  propsLeafD.set_arg<cl_mem>(6, tree.nodeUpperBounds.p());
  propsLeafD.set_arg<cl_mem>(7, tree.bodies_Pvel.p());  //Velocity to get max eps
  propsLeafD.set_arg<cl_mem>(8, tree.bodies_ids.p());  //Ids to distinguish DM and stars

  
  propsLeafD.setWork(tree.n_leafs, 128);
  LOG("PropsLeaf: "); propsLeafD.printWorkSize();
  propsLeafD.execute(execStream->s()); 
   
  
  int temp = tree.n_nodes-tree.n_leafs;
  propsNonLeafD.set_arg<int>(0,    &temp);
  propsNonLeafD.set_arg<cl_mem>(1, tree.leafNodeIdx.p());
  propsNonLeafD.set_arg<cl_mem>(2, tree.node_level_list.p());
  propsNonLeafD.set_arg<cl_mem>(3, tree.n_children.p());  
  propsNonLeafD.set_arg<cl_mem>(4, multipoleD.p());
  propsNonLeafD.set_arg<cl_mem>(5, tree.nodeLowerBounds.p());
  propsNonLeafD.set_arg<cl_mem>(6, tree.nodeUpperBounds.p());

  //Work from the bottom up
  for(int i=tree.n_levels; i >= 1; i--)
  {   
      propsNonLeafD.set_arg<int>(0,    &i);  
      {    
        vector<size_t> localWork(2), globalWork(2);
        int totalOnThisLevel;
      
        totalOnThisLevel = tree.node_level_list[i]-tree.node_level_list[i-1];
        
        propsNonLeafD.setWork(totalOnThisLevel, 128);
        
        LOG("PropsNonLeaf, nodes on level %d : %d (start: %d end: %d) , config: \t", i, totalOnThisLevel,
               tree.node_level_list[i-1], tree.node_level_list[i]); 
        propsNonLeafD.printWorkSize();
      }      
      propsNonLeafD.set_arg<int>(0,    &i); //set the level
      propsNonLeafD.execute(execStream->s());     
  }
  

  float theta2 = theta;
  
  propsScalingD.set_arg<int>(0,    &tree.n_nodes);
  propsScalingD.set_arg<cl_mem>(1, multipoleD.p());
  propsScalingD.set_arg<cl_mem>(2, tree.nodeLowerBounds.p());
  propsScalingD.set_arg<cl_mem>(3, tree.nodeUpperBounds.p());
  propsScalingD.set_arg<cl_mem>(4, tree.n_children.p());  
  propsScalingD.set_arg<cl_mem>(5, tree.multipole.p());
  propsScalingD.set_arg<float >(6, &theta2);
  propsScalingD.set_arg<cl_mem>(7, tree.boxSizeInfo.p());
  propsScalingD.set_arg<cl_mem>(8, tree.boxCenterInfo.p());
  propsScalingD.set_arg<cl_mem>(9, tree.node_bodies.p());
  
  propsScalingD.setWork(tree.n_nodes, 128);
  LOG("propsScaling: \t "); propsScalingD.printWorkSize();
  propsScalingD.execute(execStream->s());   


  #ifdef INDSOFT
    //If we use individual softening we need to get the max softening value
    //to be broadcasted during the exchange of the LET boundaries.
    //Only copy the root node that contains the max value
    my_dev::dev_stream memCpyStream;
    tree.multipole.d2h(3, false, memCpyStream.s());
  #endif

    
  //Set the group properties, note that it is not based on the nodes anymore
  //but on self created groups based on particle order setPHGroupData    
  copyNodeDataToGroupData.set_arg<int>(0,    &tree.n_groups);
  copyNodeDataToGroupData.set_arg<int>(1,    &tree.n);
  copyNodeDataToGroupData.set_arg<cl_mem>(2, tree.bodies_Ppos.p());  
  copyNodeDataToGroupData.set_arg<cl_mem>(3, tree.group_list_test.p());
  copyNodeDataToGroupData.set_arg<cl_mem>(4, tree.groupCenterInfo.p());  
  copyNodeDataToGroupData.set_arg<cl_mem>(5, tree.groupSizeInfo.p());
  copyNodeDataToGroupData.setWork(-1, NCRIT, tree.n_groups);    
  copyNodeDataToGroupData.printWorkSize();
  copyNodeDataToGroupData.execute(execStream->s());
  
  #ifdef INDSOFT  
    memCpyStream.sync();  
    this->maxLocalEps = tree.multipole[0*3 + 1].w; //Softening value
  #else
  #endif
  
  //Get the local domain boundary based on group positions and sizes
  if(nProcs > 1)
  {
    real4 r_min, r_max;
    getBoundariesGroups(tree, r_min, r_max); 
  }

}
