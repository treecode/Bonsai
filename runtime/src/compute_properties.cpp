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
  my_dev::dev_mem<real4>  nodeLowerBounds(devContext); //Lower bounds used for scaling? TODO
  my_dev::dev_mem<real4>  nodeUpperBounds(devContext); //Upper bounds used for scaling? TODO    
  
  int memBufOffset = multipoleD.cmalloc_copy          (tree.generalBuffer1, 3*tree.n_nodes, 0);
      memBufOffset = nodeLowerBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);
      memBufOffset = nodeUpperBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);  
  
  
  
  //Computes the tree-properties (size, cm, monopole, quadropole, etc)
  //start the kernel for the leaf-type nodes
  propsLeafD.set_arg<int>(0,    &tree.n_leafs);
  propsLeafD.set_arg<cl_mem>(1, tree.leafNodeIdx.p());
  propsLeafD.set_arg<cl_mem>(2, tree.node_bodies.p());
  propsLeafD.set_arg<cl_mem>(3, tree.bodies_Ppos.p());  
  propsLeafD.set_arg<cl_mem>(4, multipoleD.p());
  propsLeafD.set_arg<cl_mem>(5, nodeLowerBounds.p());
  propsLeafD.set_arg<cl_mem>(6, nodeUpperBounds.p());
  propsLeafD.set_arg<cl_mem>(7, tree.bodies_Pvel.p());  //Velocity to get max eps
  propsLeafD.set_arg<cl_mem>(8, tree.bodies_ids.p());  //Ids to distinguish DM and stars

  
  propsLeafD.setWork(tree.n_leafs, 128);
  LOG("PropsLeaf: \n"); propsLeafD.printWorkSize();
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
      propsNonLeafD.set_arg<int>(0,    &i);  
      {    
        vector<size_t> localWork(2), globalWork(2);
        int totalOnThisLevel;
      
        totalOnThisLevel = tree.node_level_list[i]-tree.node_level_list[i-1];
        
        propsNonLeafD.setWork(totalOnThisLevel, 128);
        
        LOG("PropsNonLeaf, nodes on level %d : %d (start: %d end: %d)\t\n",
            i, totalOnThisLevel,tree.node_level_list[i-1], tree.node_level_list[i]); 
//        propsNonLeafD.printWorkSize();
      }      
      propsNonLeafD.set_arg<int>(0,    &i); //set the level
      propsNonLeafD.execute(execStream->s());     
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
  LOG("propsScaling: \t \n"); propsScalingD.printWorkSize();
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
  copyNodeDataToGroupData.set_arg<cl_mem>(3, tree.group_list.p());
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
  
  
  
#if 1
  my_dev::dev_mem<float4>  output_min(devContext);
  my_dev::dev_mem<float4>  output_max(devContext); //Lower bounds used for scaling? TODO
  
  memBufOffset = output_min.cmalloc_copy(tree.generalBuffer1, tree.n_coarse_groups, 0);
  memBufOffset = output_max.cmalloc_copy(tree.generalBuffer1, tree.n_coarse_groups, memBufOffset);


  my_dev::dev_mem<uint>  atomicValues(devContext);
  atomicValues.cmalloc(256, false);
  atomicValues.zeroMem();

  segmentedCoarseGroupBoundary.set_arg<int>(0,    &tree.n_coarse_groups);
  segmentedCoarseGroupBoundary.set_arg<int>(1,    &tree.n_groups);
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(2, atomicValues.p());
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(3, tree.coarseGroupCompact.p());
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(4, tree.groupSizeInfo.p());
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(5, tree.groupCenterInfo.p());
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(6, output_min.p());
  segmentedCoarseGroupBoundary.set_arg<cl_mem>(7, output_max.p());
  segmentedCoarseGroupBoundary.setWork(-1, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);
  segmentedCoarseGroupBoundary.execute(execStream->s());
  
  execStream->sync();
  output_min.d2h();
  output_max.d2h();

  //TODO, summarize the coarse groups using the tree-structure information
  //and then send this to the other process, such that it can use the
  //tree-walk to process the data

#if 0

  //Write the tree structure to file
  tree.multipole.d2h();
  tree.boxSizeInfo.d2h();
  tree.boxCenterInfo.d2h();
  for(int i=0; i < 5; i++)
  {
    char fileName[256];
    sprintf(fileName, "fullTreeStructure-%d-level-%d.txt", mpiGetRank(), i);
    ofstream nodeFile;
    //nodeFile.open(nodeFileName.c_str());
    nodeFile.open(fileName);
    nodeFile << "NODES" << endl;

    for(int j=tree.level_list[i].x; j < tree.level_list[i].y; j++)
    {   //nodeFile << i << "\t" << tree.boxCenterInfo[i].x << "\t" << tree.boxCenterInfo[i].y;
        //nodeFile << "\t" << 2*tree.boxSizeInfo[i].x << "\t" << 2*tree.boxSizeInfo[i].y << "\t";

        nodeFile << tree.boxCenterInfo[j].x << "\t" << tree.boxCenterInfo[j].y << "\t" << tree.boxCenterInfo[j].z;
        nodeFile << "\t"  << tree.boxSizeInfo[j].x << "\t" << tree.boxSizeInfo[j].y << "\t" << tree.boxSizeInfo[j].z << "\n";

    }//for j
    nodeFile.close();
  }//for i

#endif


#if 0
  string nodeFileName = "fullTreeStructure.txt";
  char fileName[256];
  sprintf(fileName, "fullTreeStructure-%d.txt", mpiGetRank());
  ofstream nodeFile;
  //nodeFile.open(nodeFileName.c_str());
  nodeFile.open(fileName);

  nodeFile << "NODES" << endl;


  LOG("Found coarse group boundarys before increase, number of groups %d : \n", tree.n_coarse_groups);
  for(int i=0; i < tree.n_coarse_groups; i++)
  {

//	  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n",
//			  output_min[i].x,output_min[i].y,output_min[i].z,
//			  output_max[i].x,output_max[i].y,output_max[i].z);

	  float3 center = make_float3(
			  	  	  0.5*(output_max[i].x+output_min[i].x),
			  	  	  0.5*(output_max[i].y+output_min[i].y),
			  	  	  0.5*(output_max[i].z+output_min[i].z));

	  float3 size = make_float3(fmaxf(fabs(center.x-output_min[i].x), fabs(center.x-output_max[i].x)),
	                               fmaxf(fabs(center.y-output_min[i].y), fabs(center.y-output_max[i].y)),
	                               fmaxf(fabs(center.z-output_min[i].z), fabs(center.z-output_max[i].z)));
      nodeFile <<  center.x << "\t" << center.y << "\t" << center.z;
      nodeFile << "\t" << size.x << "\t" << size.y << "\t" << size.z << "\n";


   }
  {
	  //Get the local domain boundary based on group positions and sizes

	  real4 r_min, r_max;
	  getBoundariesGroups(tree, r_min, r_max);

	  float3 center = make_float3( 0.5*(r_max.x+ r_min.x),0.5*(r_max.y+ r_min.y),0.5*(r_max.z+ r_min.z));

	  float3 size = make_float3(fmaxf(fabs(center.x- r_min.x), fabs(center.x-r_max.x)),
	                            fmaxf(fabs(center.y- r_min.y), fabs(center.y-r_max.y)),
	                            fmaxf(fabs(center.z- r_min.z), fabs(center.z-r_max.z)));

      nodeFile <<  center.x << "\t" << center.y << "\t" << center.z;
      nodeFile << "\t" << size.x << "\t" << size.y << "\t" << size.z << "\n";

  }


   nodeFile.close();

   sprintf(fileName, "fullTreeStructureParticles-%d.txt", mpiGetRank());
   ofstream partFile;
   partFile.open(fileName);
   tree.bodies_Ppos.d2h();
   partFile << "POINTS\n";
   for(int i=0; i < tree.n; i++)
   {
     float4  pos =  tree.bodies_Ppos[i];
     //partFile << i << "\t" << pos.x << "\t" << pos.y << "\t" << pos.z << endl;
     partFile << pos.x << "\t" << pos.y << "\t" << pos.z << endl;
   }
   partFile.close();


#endif

   sendCurrentRadiusInfoCoarse(&output_min[0], &output_max[0], tree.n_coarse_groups);

#endif  


}
