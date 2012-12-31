#include "octree.h"


static uint4 host_get_key(int4 crd)
{
  const int bits = 30;  //20 to make it same number as morton order
  int i,xi, yi, zi;
  int mask;
  int key;

  //0= 000, 1=001, 2=011, 3=010, 4=110, 5=111, 6=101, 7=100
  //000=0=0, 001=1=1, 011=3=2, 010=2=3, 110=6=4, 111=7=5, 101=5=6, 100=4=7
  const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};

  int temp;

  mask = 1 << (bits - 1);
  key  = 0;

  uint4 key_new;

  for(i = 0; i < bits; i++, mask >>= 1)
  {
    xi = (crd.x & mask) ? 1 : 0;
    yi = (crd.y & mask) ? 1 : 0;
    zi = (crd.z & mask) ? 1 : 0;

    int index = (xi << 2) + (yi << 1) + zi;

    if(index == 0)
    {
      temp = crd.z; crd.z = crd.y; crd.y = temp;
    }
    else  if(index == 1 || index == 5)
    {
      temp = crd.x; crd.x = crd.y; crd.y = temp;
    }
    else  if(index == 4 || index == 6)
    {
      crd.x = (crd.x) ^ (-1);
      crd.z = (crd.z) ^ (-1);
    }
    else  if(index == 7 || index == 3)
    {
      temp = (crd.x) ^ (-1);
      crd.x = (crd.y) ^ (-1);
      crd.y = temp;
    }
    else
    {
      temp = (crd.z) ^ (-1);
      crd.z = (crd.y) ^ (-1);
      crd.y = temp;
    }

    key = (key << 3) + C[index];

    if(i == 19)
    {
      key_new.y = key;
      key = 0;
    }
    if(i == 9)
    {
      key_new.x = key;
      key = 0;
    }
  } //end for

   key_new.z = key;

  return key_new;
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
  
  if(10*tree.n_nodes > 3*tree.n)
  {
    LOG("Resize generalBuffer1 in compute_properties\n");
    tree.generalBuffer1.cresize(10*tree.n_nodes*4, false);
  }
  
  my_dev::dev_mem<double4> multipoleD(devContext);      //Double precision buffer to store temp results
  my_dev::dev_mem<real4>   nodeLowerBounds(devContext); //Lower bounds used for computing box sizes
  my_dev::dev_mem<real4>   nodeUpperBounds(devContext); //Upper bounds used for computing box sizes
  
  int memBufOffset = multipoleD.cmalloc_copy          (tree.generalBuffer1, 3*tree.n_nodes, 0);
      memBufOffset = nodeLowerBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);
      memBufOffset = nodeUpperBounds.cmalloc_copy(tree.generalBuffer1, tree.n_nodes, memBufOffset);

  my_dev::dev_mem<uint4>  grpKeys(devContext);
  int grpKeysOffset = grpKeys.cmalloc_copy(tree.generalBuffer1, tree.n_groups, 0); //Overlaps with multipoleD
  
  double t0 = get_time();

  this->resetCompact(); //Make sure compact has been reset, for setActiveGrp later on

    //Get the keys of the groups. This is the key of the first body in the group. This is used over
    //the group center since now we do not require a sort operation to construct the groups. Since
    //the particles on one process are in the correct order
    //TODO test overall performance (build and tree-walk) when using center+sort instead of first + nosort

  if(nProcs > 1)
  {
    //Quickly compute the keys of the groups. Done in a separate kernel since it was (much)
    //faster than combining it with the summary function
    setPHGroupDataGetKey2.set_arg<int>(0,    &tree.n_groups);
    setPHGroupDataGetKey2.set_arg<cl_mem>(1, tree.bodies_Ppos.p());
    setPHGroupDataGetKey2.set_arg<cl_mem>(2, tree.group_list.p());
    setPHGroupDataGetKey2.set_arg<cl_mem>(3, grpKeys.p());
    setPHGroupDataGetKey2.set_arg<float4>(4, &tree.corner);
    setPHGroupDataGetKey2.setWork(tree.n_groups, 128);
    setPHGroupDataGetKey2.execute(LETDataToHostStream->s());
    //Copy the data back to the host when their compute-streams are complete
    grpKeys.d2h(false, LETDataToHostStream->s());
  }

  //Set the group properties
  setPHGroupData.set_arg<int>(0,    &tree.n_groups);
  setPHGroupData.set_arg<int>(1,    &tree.n);
  setPHGroupData.set_arg<cl_mem>(2, tree.bodies_Ppos.p());
  setPHGroupData.set_arg<cl_mem>(3, tree.group_list.p());
  setPHGroupData.set_arg<cl_mem>(4, tree.groupCenterInfo.p());
  setPHGroupData.set_arg<cl_mem>(5, tree.groupSizeInfo.p());
  setPHGroupData.setWork(-1, NCRIT, tree.n_groups);
  setPHGroupData.execute(copyStream->s());

  //Set valid list to zero
  //Reset the active particles
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


  //This overlaps with setPHGroupData and is async, we can safely use the same memory
  //as the multipoleD buffer because of the sync afterwards.
  if(nProcs > 1)
  {
    tree.groupCenterInfo.d2h(false, copyStream->s());
    tree.groupSizeInfo.d2h  (false, copyStream->s());

    //have to wait till this copy is complete until props can be computed
    grpKeys.waitForCopyEvent();
  }

  double tA = get_time();

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
  propsLeafD.setWork(tree.n_leafs, 128);
  LOG("PropsLeaf: on number of leaves: %d \n", tree.n_leafs); //propsLeafD.printWorkSize();
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
  LOG("propsScaling: on number of nodes: %d \n", tree.n_nodes); // propsScalingD.printWorkSize();
  propsScalingD.execute(execStream->s());   

  #ifdef INDSOFT
    //If we use individual softening we need to get the max softening value
    //to be broadcasted during the exchange of the LET boundaries.
    //Only copy the root node that contains the max value
    my_dev::dev_stream memCpyStream;
    tree.multipole.d2h(3, false, memCpyStream.s());
  #endif


    if(nProcs > 1)
    {
      LOGF(stderr, "Starting all compute-properties kernels took; %lg  start: %lg \n", get_time()-tA, get_time()-t0);

      //Build the group tree
      double tlocal = get_time();
      //Reuse some already allocated memory
      uint2 *nodes    = (uint2*)&tree.generalBuffer1[grpKeysOffset];
      uint4 *nodeKeys = (uint4*)&tree.generalBuffer1[grpKeysOffset+2*tree.n_groups]; //2* since generalBuffer is uint

      uint node_levels[MAXLEVELS];
      int grpTree_n_levels;
      int grpTree_startGrp;
      int grpTree_endGrp;
      build_GroupTree(tree.n_groups, &grpKeys[0], nodes,
                      nodeKeys, node_levels, grpTree_n_levels,
                      grpTree_n_nodes, grpTree_startGrp, grpTree_endGrp);

      //Send the number of nodes other processes so they can allocate memory
      //this can't be done yet since we can not do async communication

      tree.groupSizeInfo.waitForCopyEvent(); //Make sure data is on the host

      //Start copying the particle positions to the host, will overlap with compute properties
      localTree.bodies_Ppos.d2h(tree.n, false, LETDataToHostStream->s());


      if(localGrpTreeCntSize)
      {
        localGrpTreeCntSize  = (real4*)realloc(localGrpTreeCntSize,  2*grpTree_n_nodes*sizeof(real4));
      }
      else
      {
        localGrpTreeCntSize =  (real4*) malloc(2*grpTree_n_nodes*sizeof(real4));
      }

      //Compute groupTree properties
      computeProps_GroupTree(&tree.groupCenterInfo[0], &tree.groupSizeInfo[0],
                             localGrpTreeCntSize, &localGrpTreeCntSize[grpTree_n_nodes],
                             nodes, node_levels, grpTree_n_levels);

      //Store information required for the tree-walk in the top node
      union{int i; float f;} itof; //__int_as_float

      itof.i                = grpTree_startGrp;
      localGrpTreeCntSize[0].x = itof.f;

      itof.i                = grpTree_endGrp;
      localGrpTreeCntSize[0].y = itof.f;

      itof.i                = grpTree_n_nodes;
      localGrpTreeCntSize[0].z = itof.f;

      double t1 = get_time();
      LOGF(stderr, "Build local tree; %lg Since start compProps: %lg\n", t1-tlocal, t1-t0);

      //Now if the GPU is not done yet with computing properties we could start sending around
      //the grpTree properties. However we can't use async communication for now. So postpone
      //that method.
    }

  //Keep this sync for now since otherwise we run the risk that memory objects are destroyed
  //while still being in use (like multipoleD).
  double t1 = get_time();
  execStream->sync();
  LOGF(stderr, "Compute properties took: %lg  wait: %lg \n", get_time()-t0, get_time()-t1);



#if 1
  //Test code to merge top nodes into a new tree-structure
  {
    uint2 node_begend;
    int level_start = tree.startLevelMin;
    node_begend.x   = tree.level_list[level_start].x;
    node_begend.y   = tree.level_list[level_start].y;

    int nTopNodes = node_begend.y-node_begend.x;

    //Retrieve properties
    tree.multipole.d2h();
    tree.boxSizeInfo.d2h();
    tree.boxCenterInfo.d2h();

    float4 *topTreeCenters    = new float4[nTopNodes];
    float4 *topTreeSizes      = new float4[nTopNodes];
    float4 *topTreeMultipole  = new float4[3*nTopNodes];

    double4 *tempMultipoleRes = new double4[3*nTopNodes];



    uint4 *keys = new uint4[nTopNodes];

    //Compute the keys based on the center of the node
    for(int i=node_begend.x; i< node_begend.y; i++)
    {
      int4 crd;
      crd.x = (int)((tree.boxCenterInfo[i].x - tree.corner.x) / tree.corner.w);
      crd.y = (int)((tree.boxCenterInfo[i].y - tree.corner.y) / tree.corner.w);
      crd.z = (int)((tree.boxCenterInfo[i].z - tree.corner.z) / tree.corner.w);

      keys[i-node_begend.x]   = host_get_key(crd);
      keys[i-node_begend.x].w = i-node_begend.x;

//      fprintf(stderr, "%d\t%d Key:\t %d \t %d \t %d \n", i-node_begend.x,i, keys[i-node_begend.x].x,keys[i-node_begend.x].y,keys[i-node_begend.x].z);
    }

    std::sort(keys, keys+nTopNodes, cmp_ph_key());

//    for(int i=0; i < nTopNodes; i++)
//      fprintf(stderr, "Sorted\t\t %d\t%d Key:\t %d \t %d \t %d \t %d\n", i,i, keys[i].x,keys[i].y,keys[i].z,keys[i].w);


    uint2 *nodes    = (uint2*)&tree.generalBuffer1[0];
    //2* since generalBuffer is uint, assuming we wont create more new nodes than
    //we have top-nodes. TODO verify if this is actually true
    uint4 *nodeKeys = (uint4*)&tree.generalBuffer1[0+2*nTopNodes];

    uint node_levels[MAXLEVELS];
    int topTree_n_levels;
    int topTree_startNode;
    int topTree_endNode;
    int topTree_n_nodes;
    build_NewTopLevels(nTopNodes, &keys[0], nodes,
                    nodeKeys, node_levels, topTree_n_levels,
                    topTree_n_nodes, topTree_startNode, topTree_endNode);


    //Now we have to compute the properties, do this from bottom up, as in the GPU case
    for(int i=topTree_n_levels;i > 0; i--)
    {
      int startNode = node_levels[i-1];
      int endNode   = node_levels[i];
      fprintf(stderr, "Working on level: %d Start: %d  End: %d \n", i, startNode, endNode);

      for(int j=startNode; j < endNode; j++)
      {
        //Extract child information
        int child    =    nodes[j].x & 0x0FFFFFFF;//Index to the first child of the node
        int nchild   = (((nodes[j].x & 0xF0000000) >> 28)) + 1;

        fprintf(stderr, "Level info: %d  \t %d : Bottom: %d Child: %d  nChild: %d\n", j, nodes[j].y, (nodes[j].y == 1), child, nchild);

        float4 *sourceCenter = NULL;
        float4 *sourceSize   = NULL;
        float4 *multipole    = NULL;

        if(nodes[j].y == 1)
        {
          //This is an end-node, read from original received data-array
          sourceCenter = &tree.boxCenterInfo[node_begend.x];
          sourceSize   = &tree.boxSizeInfo  [node_begend.x];
        }
        else
        {
          //This is a newly created node, read from new array
          sourceCenter = &topTreeCenters[0];
          sourceSize   = &topTreeSizes[0];
        }

        double3 r_min = {+1e10f, +1e10f, +1e10f};
        double3 r_max = {-1e10f, -1e10f, -1e10f};

        double mass, posx, posy, posz;
        mass = posx = posy = posz = 0.0;

        double oct_q11, oct_q22, oct_q33;
        double oct_q12, oct_q13, oct_q23;

        oct_q11 = oct_q22 = oct_q33 = 0.0;
        oct_q12 = oct_q13 = oct_q23 = 0.0;

        for(int k=child; k < child+nchild; k++) //NOTE <= otherwise we miss the last child
        {
          double4 pos;
          double4 Q0, Q1;
          //Process/merge the children into this node

          //The center, compute the center+size back to a min/max
          double3 curRmin = {sourceCenter[k].x - sourceSize[k].x,
                             sourceCenter[k].y - sourceSize[k].y,
                             sourceCenter[k].z - sourceSize[k].z};
          double3 curRmax = {sourceCenter[k].x + sourceSize[k].x,
                             sourceCenter[k].y + sourceSize[k].y,
                             sourceCenter[k].z + sourceSize[k].z};

          //Compute the new min/max
          r_min.x = min(curRmin.x, r_min.x);
          r_min.y = min(curRmin.y, r_min.y);
          r_min.z = min(curRmin.z, r_min.z);
          r_max.x = max(curRmax.x, r_max.x);
          r_max.y = max(curRmax.y, r_max.y);
          r_max.z = max(curRmax.z, r_max.z);

          //Compute monopole and quadrupole
          if(nodes[j].y == 1)
          {
            pos = make_double4(tree.multipole[3*(node_begend.x+k)+0].x,
                               tree.multipole[3*(node_begend.x+k)+0].y,
                               tree.multipole[3*(node_begend.x+k)+0].z,
                               tree.multipole[3*(node_begend.x+k)+0].w);
            Q0  = make_double4(tree.multipole[3*(node_begend.x+k)+1].x,
                               tree.multipole[3*(node_begend.x+k)+1].y,
                               tree.multipole[3*(node_begend.x+k)+1].z,
                               tree.multipole[3*(node_begend.x+k)+1].w);
            Q1  = make_double4(tree.multipole[3*(node_begend.x+k)+2].x,
                               tree.multipole[3*(node_begend.x+k)+2].y,
                               tree.multipole[3*(node_begend.x+k)+2].z,
                               tree.multipole[3*(node_begend.x+k)+2].w);
            double temp = Q1.y;
            Q1.y = Q1.z; Q1.z = temp;
            //Scale back to original order
            double im = 1.0 / pos.w;
            Q0.x = Q0.x + pos.x*pos.x; Q0.x = Q0.x / im;
            Q0.y = Q0.y + pos.y*pos.y; Q0.y = Q0.y / im;
            Q0.z = Q0.z + pos.z*pos.z; Q0.z = Q0.z / im;
            Q1.x = Q1.x + pos.x*pos.y; Q1.x = Q1.x / im;
            Q1.y = Q1.y + pos.y*pos.z; Q1.y = Q1.y / im;
            Q1.z = Q1.z + pos.x*pos.z; Q1.z = Q1.z / im;
          }
          else
          {
            pos = tempMultipoleRes[3*k+0];
            Q0  = tempMultipoleRes[3*k+1];
            Q1  = tempMultipoleRes[3*k+2];
          }

          mass += pos.w;
          posx += pos.w*pos.x;
          posy += pos.w*pos.y;
          posz += pos.w*pos.z;

          //Quadrupole
          oct_q11 += Q0.x;
          oct_q22 += Q0.y;
          oct_q33 += Q0.z;
          oct_q12 += Q1.x;
          oct_q13 += Q1.y;
          oct_q23 += Q1.z;
        }

        double4 mon = {posx, posy, posz, mass};
        double im = 1.0/mon.w;
        if(mon.w == 0) im = 0; //Allow tracer/mass-less particles

        mon.x *= im;
        mon.y *= im;
        mon.z *= im;

        tempMultipoleRes[j*3+0] = mon;
        tempMultipoleRes[j*3+1] = make_double4(oct_q11,oct_q22,oct_q33,0);
        tempMultipoleRes[j*3+2] = make_double4(oct_q12,oct_q13,oct_q23,0);
        //Store float4 results right away, so we do not have to do an extra loop
        //Scale the quadropole
        double4 Q0, Q1;
        Q0.x = oct_q11*im - mon.x*mon.x;
        Q0.y = oct_q22*im - mon.y*mon.y;
        Q0.z = oct_q33*im - mon.z*mon.z;
        Q1.x = oct_q12*im - mon.x*mon.y;
        Q1.y = oct_q13*im - mon.y*mon.z;
        Q1.z = oct_q23*im - mon.x*mon.z;

        //Switch the y and z parameter
        double temp = Q1.y;
        Q1.y = Q1.z; Q1.z = temp;


        topTreeMultipole[j*3+0] = make_float4(mon.x,mon.y,mon.z,mon.w);
        topTreeMultipole[j*3+1] = make_float4(Q0.x,Q0.y,Q0.z,0);
        topTreeMultipole[j*3+2] = make_float4(Q1.x,Q1.y,Q1.z,0);

        //All intermediate steps are done in full-double precision to prevent round-off
        //errors. Note that there is still a chance of round-off errors, because we start
        //with float data, while on the GPU we start/keep full precision data
        double4 boxCenterD;
        boxCenterD.x = 0.5*((double)r_min.x + (double)r_max.x);
        boxCenterD.y = 0.5*((double)r_min.y + (double)r_max.y);
        boxCenterD.z = 0.5*((double)r_min.z + (double)r_max.z);

        double4 boxSizeD = make_double4(std::max(abs(boxCenterD.x-r_min.x), abs(boxCenterD.x-r_max.x)),
                                        std::max(abs(boxCenterD.y-r_min.y), abs(boxCenterD.y-r_max.y)),
                                        std::max(abs(boxCenterD.z-r_min.z), abs(boxCenterD.z-r_max.z)), 0);

        //Compute distance between center box and center of mass
        double3 s3     = make_double3((boxCenterD.x - mon.x), (boxCenterD.y - mon.y), (boxCenterD.z -     mon.z));

        double s      = sqrt((s3.x*s3.x) + (s3.y*s3.y) + (s3.z*s3.z));
        //If mass-less particles form a node, the s would be huge in opening angle, make it 0
        if(fabs(mon.w) < 1e-10) s = 0;

        //Length of the box, note times 2 since we only computed half the distance before
        double l = 2*fmax(boxSizeD.x, fmax(boxSizeD.y, boxSizeD.z));

        //Extra check, shouldn't be necessary, probably it is otherwise the test for leaf can fail
        //This actually IS important Otherwise 0.0 < 0 can fail, now it will be: -1e-12 < 0
        if(l < 0.000001)
          l = 0.000001;

        #ifdef IMPBH
          double cellOp = (l/theta) + s;
        #else
          //Minimum distance method
          float cellOp = (l/theta);
        #endif

        boxCenterD.w       = cellOp*cellOp;
        float4 boxCenter   = make_float4(boxCenterD.x,boxCenterD.y, boxCenterD.z, boxCenterD.w);
        topTreeCenters[j]  = boxCenter;

        //Encode the child information, the leaf offsets are changed
        //such that they point to the correct starting offsets
        //in the final array, which starts after the 'topTree_n_nodes'
        //items.
        if(nodes[j].y == 1)
        { //Leaf
          child += topTree_n_nodes;
        }
        
        int childInfo = child | (nchild << 28);
        
        union{float f; int i;} u; //__float_as_int
        u.i           = childInfo;

        float4 boxSize   = make_float4(boxSizeD.x, boxSizeD.y, boxSizeD.z, 0);
        boxSize.w        = u.f; //int_as_float

        topTreeSizes[j] = boxSize;
      }//for startNode < endNode
    }//for each topTree level

    //Compare the results
    for(int i=0; i < node_begend.x; i++)
    {
      fprintf(stderr, "XNode: %d \tSource size: %f %f %f %f Source center: %f %f %f %f \n",i,
          tree.boxSizeInfo[i].x,tree.boxSizeInfo[i].y,tree.boxSizeInfo[i].z,tree.boxSizeInfo[i].w,
          tree.boxCenterInfo[i].x,tree.boxCenterInfo[i].y,tree.boxCenterInfo[i].z, 
          tree.boxCenterInfo[i].w);

      fprintf(stderr, "XNode: %d \tNew    Size: %f %f %f %f  New    center: %f %f %f %f\n",i,
          topTreeSizes[i].x,  topTreeSizes[i].y,  topTreeSizes[i].z,   topTreeSizes[i].w,
          topTreeCenters[i].x,topTreeCenters[i].y,topTreeCenters[i].z, topTreeCenters[i].w);


      fprintf(stderr, "XOri-Node: %d \tMono: %f %f %f %f \tQ0: %f %f %f \tQ1: %f %f %f\n",i,
          tree.multipole[3*i+0].x,tree.multipole[3*i+0].y,tree.multipole[3*i+0].z,tree.multipole[3*i+0].w,
          tree.multipole[3*i+1].x,tree.multipole[3*i+1].y,tree.multipole[3*i+1].z,
          tree.multipole[3*i+2].x,tree.multipole[3*i+2].y,tree.multipole[3*i+2].z);

      fprintf(stderr, "XNew-Node: %d \tMono: %f %f %f %f \tQ0: %f %f %f \tQ1: %f %f %f\n\n\n",i,
          topTreeMultipole[3*i+0].x,topTreeMultipole[3*i+0].y,topTreeMultipole[3*i+0].z,topTreeMultipole[3*i+0].w,
          topTreeMultipole[3*i+1].x,topTreeMultipole[3*i+1].y,topTreeMultipole[3*i+1].z,
          topTreeMultipole[3*i+2].x,topTreeMultipole[3*i+2].y,topTreeMultipole[3*i+2].z);
    }


  }//end function/section
#endif








}










#if 0

  //Get the local domain boundary based on group positions and sizes
  if(nProcs > 1)
  {
    real4 r_min, r_max;
    getBoundariesGroups(tree, r_min, r_max); 
  }

  
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

//    LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n",
//        output_min[i].x,output_min[i].y,output_min[i].z,
//        output_max[i].x,output_max[i].y,output_max[i].z);

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

//   sendCurrentRadiusInfoCoarse(&output_min[0], &output_max[0], tree.n_coarse_groups);

#endif  

//  for(int i=0; i < grpTree_n_levels; i++)
//  {
//    char fileName[256];
//    sprintf(fileName, "/tmp/XgrpTest-%d.txt", i);
//
//    FILE *out = fopen(fileName, "w");
//    fprintf(out,"NODES\n");
//    for(int j= node_levels[i]; j < node_levels[i+1]; j++)
//    {
//      fprintf(out, "%f %f %f %f %f %f \n",
//          treeCnt[j].x,treeCnt[j].y,treeCnt[j].z,
//          treeSize[j].x,treeSize[j].y,treeSize[j].z);
//    }
//    fclose(out);
//  }
//
//
//  {
//    char fileName[256];
//    sprintf(fileName, "/tmp/XgrpTest-%d.txt", grpTree_n_levels);
//
//    FILE *out = fopen(fileName, "w");
//    fprintf(out,"NODES\n");
//    for(int j= 0; j < tree.n_groups; j++)
//    {
//      fprintf(out, "%f %f %f %f %f %f \n",
//          tree.groupCenterInfo[j].x,tree.groupCenterInfo[j].y,tree.groupCenterInfo[j].z,
//          tree.groupSizeInfo[j].x,tree.groupSizeInfo[j].y,tree.groupSizeInfo[j].z);
//    }
//    fclose(out);
//  }
