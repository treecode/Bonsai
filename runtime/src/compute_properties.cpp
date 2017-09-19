#include "octree.h"
                                                                                                    
inline int host_float_as_int(float val)                                                             
{                                                                                                   
  union{float f; int i;} u; //__float_as_int                                                        
  u.f           = val;                                                                              
  return u.i;                                                                                       
}   


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
    gpuCompact(tree.activeGrpList, tree.active_group_list, tree.n_groups, &tree.n_active_groups);

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


    bool smoothingOnly = false; //We want the full tree at this point
    gpuBoundaryTreeExtract.set_args(0, &smoothingOnly,
                                   tree.boxSizeInfo.p(), tree.boxCenterInfo.p(), tree.boxSmoothing.p(), tree.multipole.p(),
                                   tree.bodies_Ppos.p(), tree.bodies_Pvel.p(), tree.bodies_dens.p(), tree.bodies_hydro.p(),
                                   tree.smallBoundaryTreeIndices.p(), tree.smallBoundaryTree.p());
    gpuBoundaryTreeExtract.setWork(32, 32, 1);
    gpuBoundaryTreeExtract.execute2(execStream->s());
    gpuBoundaryTreeExtract.set_args(0, &smoothingOnly,
                                   tree.boxSizeInfo.p(), tree.boxCenterInfo.p(), tree.boxSmoothing.p(), tree.multipole.p(),
                                   tree.bodies_Ppos.p(), tree.bodies_Pvel.p(), tree.bodies_dens.p(), tree.bodies_hydro.p(),
                                   tree.fullBoundaryTreeIndices.p(), tree.fullBoundaryTree.p());
    gpuBoundaryTreeExtract.execute2(execStream->s());

    //Keep this sync for now since otherwise we run the risk that memory objects are destroyed
    //while still being in use (like multipoleD).
    double t1 = get_time();
    execStream->sync();
    LOGF(stderr, "Compute properties took: %lg  wait: %lg \n", get_time()-t0, get_time()-t1);

    devContext->stopTiming("Compute-properties", 3, execStream->s());

    if(nProcs > 1)
    {
       //Only start the copy after the execStream has been completed, otherwise the buffers aint filled yet
       tree.smallBoundaryTree.d2h(boundaryTreeDimensions.x, false, LETDataToHostStream->s());
       tree.fullBoundaryTree. d2h(boundaryTreeDimensions.y, false, LETDataToHostStream->s());
    }


#if 0
    cudaDeviceSynchronize();

    {
        //Compute the startOffset and number of items for the memory copies
        int2 nInfoSmall = make_int2(host_float_as_int(localTree.smallBoundaryTree[0].x), host_float_as_int(localTree.smallBoundaryTree[0].y));
        int2 nInfoFull  = make_int2(host_float_as_int(localTree.fullBoundaryTree[0].x),  host_float_as_int(localTree.fullBoundaryTree[0].y));

        int offsetSmall = 1+nInfoSmall.x+2*nInfoSmall.y;
        int offsetFull  = 1+nInfoFull.x +2*nInfoFull.y;

        for(int i=-3; i < 2; i++)
        {
            fprintf(stderr,"SMOOTH: %d  is: %f \n", i, localTree.smallBoundaryTree[offsetSmall+i].x);
        }
        for(int i=nInfoSmall.y-3; i < nInfoSmall.y+2; i++)
        {
            fprintf(stderr,"SMOOTH: %d  is: %f \n", i, localTree.smallBoundaryTree[offsetSmall+i].x);
        }

        updateCurrentInfoGrpTree();

        for(int i=-3; i < 2; i++)
        {
            fprintf(stderr,"SMOOTH: %d  is: %f \n", i, localTree.smallBoundaryTree[offsetSmall+i].x);
        }
        for(int i=nInfoSmall.y-3; i < nInfoSmall.y+2; i++)
        {
            fprintf(stderr,"SMOOTH: %d  is: %f \n", i, localTree.smallBoundaryTree[offsetSmall+i].x);
        }
        exit(0);
    }

#endif



#if 0
   if(procId == 0){

   {
     FILE *f = fopen("grpNew.txt", "w");

   int validParticles = host_float_as_int(localTree.smallBoundaryTree[0].x);
   int validNodes     = host_float_as_int(localTree.smallBoundaryTree[0].y);

   int sizeStart  = 1+validParticles;
   int cntrStart  = sizeStart+validNodes;
   int smthStart  = sizeStart+validNodes*2;
   int multiStart = sizeStart+validNodes*3;


   for(int i=0; i < validNodes; i++)
   {
       fprintf(f,"Node: %d Size\t%f %f %f %d\t| Cntr\t%f %f %f %f Smth\t%f Mult\t%f %f %f\n",
               i,
               tree.smallBoundaryTree[sizeStart+i].x,
               tree.smallBoundaryTree[sizeStart+i].y,
               tree.smallBoundaryTree[sizeStart+i].z,
               host_float_as_int(tree.smallBoundaryTree[sizeStart+i].w),

               tree.smallBoundaryTree[cntrStart+i].x,
               tree.smallBoundaryTree[cntrStart+i].y,
               tree.smallBoundaryTree[cntrStart+i].z,
               tree.smallBoundaryTree[cntrStart+i].w,

               tree.smallBoundaryTree[smthStart+i].x,

               tree.smallBoundaryTree[multiStart+(i*3)+0].x,
               tree.smallBoundaryTree[multiStart+(i*3)+1].x,
               tree.smallBoundaryTree[multiStart+(i*3)+2].x);
   }
   for(int i=0; i < validParticles; i++)
       fprintf(f,"Body: %d Cntr\t%f %f %f %f\n",
               i,
               tree.smallBoundaryTree[1 + i].x,
               tree.smallBoundaryTree[1 + i].y,
               tree.smallBoundaryTree[1 + i].z,
               tree.smallBoundaryTree[1 + i].w);
   fclose(f);
   }

   {
   FILE *f = fopen("grpNewF.txt", "w");

   int validParticles = host_float_as_int(localTree.fullBoundaryTree[0].x);
   int validNodes     = host_float_as_int(localTree.fullBoundaryTree[0].y);

   int sizeStart  = 1+validParticles;
   int cntrStart  = sizeStart+validNodes;
   int smthStart  = sizeStart+validNodes*2;
   int multiStart = sizeStart+validNodes*3;


   for(int i=0; i < validNodes; i++)
   {
       fprintf(f,"Node: %d Size\t%f %f %f %d\t| Cntr\t%f %f %f %f Smth\t%f Mult\t%f %f %f\n",
               i,
               tree.fullBoundaryTree[sizeStart+i].x,
               tree.fullBoundaryTree[sizeStart+i].y,
               tree.fullBoundaryTree[sizeStart+i].z,
               host_float_as_int(tree.fullBoundaryTree[sizeStart+i].w),

               tree.fullBoundaryTree[cntrStart+i].x,
               tree.fullBoundaryTree[cntrStart+i].y,
               tree.fullBoundaryTree[cntrStart+i].z,
               tree.fullBoundaryTree[cntrStart+i].w,

               tree.fullBoundaryTree[smthStart+i].x,

               tree.fullBoundaryTree[multiStart+(i*3)+0].x,
               tree.fullBoundaryTree[multiStart+(i*3)+1].x,
               tree.fullBoundaryTree[multiStart+(i*3)+2].x);
   }
   for(int i=0; i < validParticles; i++)
       fprintf(f,"Body: %d Cntr\t%f %f %f %f\n",
               i,
               tree.fullBoundaryTree[1 + i].x,
               tree.fullBoundaryTree[1 + i].y,
               tree.fullBoundaryTree[1 + i].z,
               tree.fullBoundaryTree[1 + i].w);
   fclose(f);
   }
}

#endif

// tree.boxSizeInfo.d2h(3);
// tree.boxCenterInfo.d2h(3);
// tree.boxSmoothing.d2h();
// const float maxSmth = tree.boxSmoothing[0];
// char fileName[128];
// sprintf(fileName, "searchBoundary-%d.txt", mpiGetRank());
// ofstream nodeFile;
// nodeFile.open(fileName);
// nodeFile << "NODES" << endl;
// nodeFile << tree.boxCenterInfo[0].x << "\t" << tree.boxCenterInfo[0].y << "\t" << tree.boxCenterInfo[0].z;
// nodeFile << "\t"  << maxSmth+tree.boxSizeInfo[0].x << "\t" << maxSmth+tree.boxSizeInfo[0].y << "\t" << maxSmth+tree.boxSizeInfo[0].z << "\n";
// nodeFile.close();
//
//
//fprintf(stderr,"TOP LEVEL BOX: %f %f %f \t %f %f %f \t SMTH: %f\n",
//        tree.boxCenterInfo[0].x,
//        tree.boxCenterInfo[0].y,
//        tree.boxCenterInfo[0].z,
//        tree.boxSizeInfo[0].x,
//        tree.boxSizeInfo[0].y,
//        tree.boxSizeInfo[0].z,
//        tree.boxSmoothing[0]);
//
//  this->dumpTreeStructureToFile(tree);


} //compute_propertiesD


