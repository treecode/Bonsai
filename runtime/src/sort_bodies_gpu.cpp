#include "octree.h"

void octree::getBoundaries(tree_structure &tree, real4 &r_min, real4 &r_max)
{

  //Start reduction to get the boundary's of the system
  boundaryReduction.set_arg<int>(0, &tree.n);
  boundaryReduction.set_arg<cl_mem>(1, tree.bodies_Ppos.p());
  boundaryReduction.set_arg<cl_mem>(2, devMemRMIN.p());
  boundaryReduction.set_arg<cl_mem>(3, devMemRMAX.p());

  boundaryReduction.setWork(tree.n, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);  //256 threads and 120 blocks in total
  boundaryReduction.execute(execStream->s());
  
   
  devMemRMIN.d2h();     //Need to be defined and initialized somewhere outside this function
  devMemRMAX.d2h();     //Need to be defined and initialized somewhere outside this function
  r_min = make_real4(+1e10, +1e10, +1e10, +1e10); 
  r_max = make_real4(-1e10, -1e10, -1e10, -1e10);   
  
  //Reduce the blocks, done on host since its
  //A faster and B we need the results anyway
  for (int i = 0; i < 120; i++) {    
    r_min.x = std::min(r_min.x, devMemRMIN[i].x);
    r_min.y = std::min(r_min.y, devMemRMIN[i].y);
    r_min.z = std::min(r_min.z, devMemRMIN[i].z);
    
    r_max.x = std::max(r_max.x, devMemRMAX[i].x);
    r_max.y = std::max(r_max.y, devMemRMAX[i].y);
    r_max.z = std::max(r_max.z, devMemRMAX[i].z);    
  }
  
  rMinLocalTree = r_min;
  rMaxLocalTree = r_max;
  
  LOG("Found boundarys, number of particles %d : \n", tree.n);
  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);
}

void octree::getBoundariesGroups(tree_structure &tree, real4 &r_min, real4 &r_max)
{
  //Start reduction to get the boundary's of the system
  boundaryReductionGroups.set_arg<int>(0, &tree.n_groups);
  boundaryReductionGroups.set_arg<cl_mem>(1, tree.groupCenterInfo.p());
  boundaryReductionGroups.set_arg<cl_mem>(2, tree.groupSizeInfo.p());
  boundaryReductionGroups.set_arg<cl_mem>(3, devMemRMIN.p());
  boundaryReductionGroups.set_arg<cl_mem>(4, devMemRMAX.p());

  boundaryReductionGroups.setWork(tree.n_groups, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);  //256 threads and 120 blocks in total
  boundaryReductionGroups.execute(execStream->s());

   
  devMemRMIN.d2h();     //Need to be defined and initialized somewhere outside this function
  devMemRMAX.d2h();     //Need to be defined and initialized somewhere outside this function
  r_min = make_real4(+1e10f, +1e10f, +1e10f, +1e10f); 
  r_max = make_real4(-1e10f, -1e10f, -1e10f, -1e10f);   
  
  //Reduce the blocks, done on host since its
  //A faster and B we need the results anyway
  for (int i = 0; i < 120; i++) {    
    r_min.x = std::min(r_min.x, devMemRMIN[i].x);
    r_min.y = std::min(r_min.y, devMemRMIN[i].y);
    r_min.z = std::min(r_min.z, devMemRMIN[i].z);
    
    r_max.x = std::max(r_max.x, devMemRMAX[i].x);
    r_max.y = std::max(r_max.y, devMemRMAX[i].y);
    r_max.z = std::max(r_max.z, devMemRMAX[i].z);    
  }
  
  LOG("Found group boundarys before increase, number of groups %d : \n", tree.n_groups);
  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);
  
  //Prevent small-numerical differences by making the group/box slightly bigger
  
  double smallFac1 = 0.99;
  double smallFac2 = 1.01;
  
  //Note that we have to check the sign to move the border in the right
  //direction
  r_min.x = (float)((r_min.x < 0) ? r_min.x * smallFac2 : r_min.x * smallFac1);
  r_min.y = (float)((r_min.y < 0) ? r_min.y * smallFac2 : r_min.y * smallFac1);
  r_min.z = (float)((r_min.z < 0) ? r_min.z * smallFac2 : r_min.z * smallFac1);

  r_max.x = (float)((r_max.x < 0) ? r_max.x * smallFac1 : r_max.x * smallFac2);
  r_max.y = (float)((r_max.y < 0) ? r_max.y * smallFac1 : r_max.y * smallFac2);
  r_max.z = (float)((r_max.z < 0) ? r_max.z * smallFac1 : r_max.z * smallFac2);
  
  
  LOG("Found group boundarys after increase, number of groups %d : \n", tree.n_groups);
  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);
}



void octree::sort_bodies(tree_structure &tree, bool doDomainUpdate, bool doFullShuffle) {

  //We assume the bodies are already onthe GPU
  devContext.startTiming(execStream->s());
  real4 r_min = {+1e10, +1e10, +1e10, +1e10}; 
  real4 r_max = {-1e10, -1e10, -1e10, -1e10};   
  
  if(doDomainUpdate)
  {
    getBoundaries(tree, r_min, r_max);  
    //Sync the boundary over the various processes
    if(this->mpiGetNProcs() > 1)
    {
      this->sendCurrentRadiusInfo(r_min, r_max);
    }
    rMinGlobal = r_min;    rMaxGlobal = r_max;
  }
  
  r_min = rMinGlobal;
  r_max = rMaxGlobal;
  
  //Compute the boundarys of the tree  
  real size     = 1.001f*std::max(r_max.z - r_min.z,
                         std::max(r_max.y - r_min.y, r_max.x - r_min.x));
  
  tree.corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                             0.5f*(r_min.y + r_max.y) - 0.5f*size,
                             0.5f*(r_min.z + r_max.z) - 0.5f*size, size); 
       
  tree.domain_fac   = size/(1 << MAXLEVELS);
  
  
  float idomain_fac = 1.0f/tree.domain_fac;
  float domain_fac  = tree.domain_fac;
  
  tree.corner.w = domain_fac;  
  
  LOG("Corner: %f %f %f idomain fac: %f domain_fac: %f\n", 
         tree.corner.x, tree.corner.y, tree.corner.z, idomain_fac, domain_fac);
  LOG("domain fac: %f idomain_fac: %f size: %f MAXLEVELS: %d \n", domain_fac, idomain_fac, size, MAXLEVELS);

  //Call the GPUSort function, since we made it general 
  //into a uint4 so we can extend the tree to 96bit key
  //we have to convert to 64bit key to a 96bit for sorting
  //and back from 96 to 64    
  my_dev::dev_mem<uint4>  srcValues(devContext);  
  
  //The generalBuffer1 has size uint*4*N*3
  //this buffer gets part: 0-uint*4*N
  srcValues.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
  
  //Compute the keys directly into srcValues 
  // will be sorted into tree.bodies_key below
  build_key_list.set_arg<cl_mem>(0,   srcValues.p());
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<int>(2,      &tree.n);
  build_key_list.set_arg<real4>(3,    &tree.corner);
  
  build_key_list.setWork(tree.n, 128); //128 threads per block
  build_key_list.execute(execStream->s());  
  
  // If srcValues and buffer are different, then the original values
  // are preserved, if they are the same srcValues will be overwritten
  if(tree.n > 0)
    gpuSort(devContext, srcValues, tree.bodies_key,srcValues, tree.n, 32, 3, tree);

  devContext.stopTiming("Sorting", 0, execStream->s());  

  //Call the reorder data functions
  devContext.startTiming(execStream->s());
  
  static int oneRunFull = 0;

  //JB this if statement is required untill I fix the order 
  //of functions in main.cpp  
  //if(oneRunFull == 1)
  if(!doFullShuffle)
  {
    my_dev::dev_mem<real4>  real4Buffer1(devContext);
    my_dev::dev_mem<ullong> ullBuffer1(devContext);

    
    int genBufOffset = real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
        genBufOffset = ullBuffer1.cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset);

    
    dataReorderR4.set_arg<int>(0,      &tree.n);
    dataReorderR4.set_arg<cl_mem>(1,   tree.bodies_key.p());  
    dataReorderR4.setWork(tree.n, 512);   

    //Position, velocity and acc0
    dataReorderR4.set_arg<cl_mem>(2,   tree.bodies_Ppos.p());
    dataReorderR4.set_arg<cl_mem>(3,   real4Buffer1.p()); 
    dataReorderR4.set_arg<cl_mem>(4,   tree.bodies_ids.p()); 
    dataReorderR4.set_arg<cl_mem>(5,   ullBuffer1.p());
    dataReorderR4.set_arg<cl_mem>(6,   tree.oriParticleOrder.p()); 
    dataReorderR4.execute(execStream->s());
    
//    tree.bodies_Ppos.copy(real4Buffer1,  tree.n);
//    tree.bodies_ids.copy (intBuffer1,    tree.n);
    tree.bodies_Ppos.copy_devonly(real4Buffer1,  tree.n);
    tree.bodies_ids.copy_devonly (ullBuffer1,    tree.n);


    //Shuffle the density values
    //NOTE I reused this kernel for the gpu_dataReorderF1 function
    my_dev::dev_mem<float>  realBuffer(devContext);
    genBufOffset = realBuffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    dataReorderI1.set_arg<int   >(0,      &tree.n);
    dataReorderI1.set_arg<cl_mem>(1,      tree.bodies_key.p());
    dataReorderI1.set_arg<cl_mem>(2,      tree.bodies_h.p());
    dataReorderI1.set_arg<cl_mem>(3,      realBuffer.p());
    dataReorderI1.setWork(tree.n, 512);   
    dataReorderI1.execute(execStream->s());
    tree.bodies_h.copy(realBuffer, realBuffer.get_size()); 

  }
  else
  {
    oneRunFull = 1;
    //Call the reorder data functions
    //First generate some memory buffers
    //generalBuffer is always at least 3xfloat4*N
    my_dev::dev_mem<real4>  real4Buffer1(devContext);
    my_dev::dev_mem<real4>  real4Buffer2(devContext);
    my_dev::dev_mem<real4>  real4Buffer3(devContext);
    
    int genBufOffset1 = real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
        genBufOffset1 = real4Buffer2.cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset1);    
        genBufOffset1 = real4Buffer3.cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset1);         
    

    
    dataReorderCombined.set_arg<int>(0,      &tree.n);
    dataReorderCombined.set_arg<cl_mem>(1,   tree.bodies_key.p());  
    dataReorderCombined.setWork(tree.n, 512);   
  //   dataReorderCombined.setWork(tree.n, 512, 240);  //256 threads and 120 blocks in total
    
    
    //Position, velocity and acc0
    dataReorderCombined.set_arg<cl_mem>(2,   tree.bodies_pos.p());
    dataReorderCombined.set_arg<cl_mem>(3,   real4Buffer1.p()); 
    dataReorderCombined.set_arg<cl_mem>(4,   tree.bodies_vel.p()); 
    dataReorderCombined.set_arg<cl_mem>(5,   real4Buffer2.p()); 
    dataReorderCombined.set_arg<cl_mem>(6,   tree.bodies_acc0.p()); 
    dataReorderCombined.set_arg<cl_mem>(7,   real4Buffer3.p()); 
    dataReorderCombined.execute(execStream->s());
    tree.bodies_pos.copy(real4Buffer1,  tree.n);
    tree.bodies_vel.copy(real4Buffer2,  tree.n);
    tree.bodies_acc0.copy(real4Buffer3, tree.n);
    
    //Acc1, Predicted position and velocity
    dataReorderCombined.set_arg<cl_mem>(2,   tree.bodies_acc1.p()); 
    dataReorderCombined.set_arg<cl_mem>(3,   real4Buffer1.p()); 
    dataReorderCombined.set_arg<cl_mem>(4,   tree.bodies_Ppos.p());
    dataReorderCombined.set_arg<cl_mem>(5,   real4Buffer2.p()); 
    dataReorderCombined.set_arg<cl_mem>(6,   tree.bodies_Pvel.p()); 
    dataReorderCombined.set_arg<cl_mem>(7,   real4Buffer3.p());   
    dataReorderCombined.execute(execStream->s());

    tree.bodies_acc1.copy(real4Buffer1, tree.n);
    tree.bodies_Ppos.copy(real4Buffer2,  tree.n);
    tree.bodies_Pvel.copy(real4Buffer3, tree.n);   


    //These can reuse the real4Buffer1 and 2 space :-)
    my_dev::dev_mem<float2>  float2Buffer(devContext);
    my_dev::dev_mem<ullong> sortPermutation(devContext);
    genBufOffset1 = float2Buffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    genBufOffset1 = sortPermutation.cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset1);
    
    
    dataReorderF2.set_arg<int>(0,      &tree.n);
    dataReorderF2.set_arg<cl_mem>(1,   tree.bodies_key.p());  
    
    dataReorderF2.set_arg<cl_mem>(2,   tree.bodies_time.p());
    dataReorderF2.set_arg<cl_mem>(3,   float2Buffer.p()); //Reuse as destination1
    dataReorderF2.set_arg<cl_mem>(4,   tree.bodies_ids.p()); 
    dataReorderF2.set_arg<cl_mem>(5,   sortPermutation.p()); //Reuse as destination2  
    dataReorderF2.setWork(tree.n, 512);   
    dataReorderF2.execute(execStream->s());
    
    
    tree.bodies_time.copy(float2Buffer, float2Buffer.get_size()); 
    tree.bodies_ids.copy(sortPermutation, sortPermutation.get_size());  
    
    //Shuffle the density values
    //NOTE I reused this kernel for the gpu_dataReorderF1 function
    my_dev::dev_mem<float>  realBuffer(devContext);
    genBufOffset1 = realBuffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    dataReorderI1.set_arg<int>(0,      &tree.n);
    dataReorderI1.set_arg<cl_mem>(1,      tree.bodies_key.p());
    dataReorderI1.set_arg<cl_mem>(2,      tree.bodies_h.p());
    dataReorderI1.set_arg<cl_mem>(3,      realBuffer.p());
    dataReorderI1.setWork(tree.n, 512);   
    dataReorderI1.execute(execStream->s());
    tree.bodies_h.copy(realBuffer, realBuffer.get_size()); 

  } //end if
  
  devContext.stopTiming("Data-reordering", 1, execStream->s());   


#if 0

  //Items = 11078474
  //Valid = 6558113
  //Buff = 236720128

   if(procId == 0)
   {
     const int nExportParticles = 6558113;

     bool doInOneGo              = true;
     bodyStruct *extraBodyBuffer = NULL;

     localTree.generalBuffer1.cresize(236720128, true);


    my_dev::dev_mem<uint2>  validList2(devContext);
    my_dev::dev_mem<uint2>  validList3(devContext);
    int tempOffset1 = validList2.  cmalloc_copy(localTree.generalBuffer1, localTree.n, 0);
        tempOffset1 = validList3.  cmalloc_copy(localTree.generalBuffer1, localTree.n, tempOffset1);


    //Check if the memory size, of the generalBuffer is large enough to store the exported particles
    //if not allocate more but make sure that the copy of compactList survives
    int validCount = nExportParticles;
    //int tempSize   = localTree.generalBuffer1.get_size() - (4*localTree.n); //4* = 2x uint2 validList2/3
    int tempSize   = localTree.generalBuffer1.get_size() - tempOffset1; //4* = 2x uint2 validList2/3
    int stepSize   = (tempSize / (sizeof(bodyStruct) / sizeof(int)))-512; //Available space in # of bodyStructs

    if(stepSize > nExportParticles)
    {
      doInOneGo = true; //We can do it in one go
    }
    else
    {
      doInOneGo       = false; //We need an extra CPU buffer
      extraBodyBuffer = new bodyStruct[validCount];
      assert(extraBodyBuffer != NULL);
    }


    my_dev::dev_mem<bodyStruct>  bodyBuffer(devContext);
    int memOffset1 = bodyBuffer.cmalloc_copy(localTree.generalBuffer1, stepSize, tempOffset1);

    for(int i=0; i < validCount; i++)
      validList2[i] = make_uint2(i,i);
    validList2.h2d();

    FILE *out = fopen("temp-1.txt", "w");


    int extractOffset = 0;
    for(unsigned int i=0; i < validCount; i+= stepSize)
    {
      int items = min(stepSize, (int)(validCount-i));

      if(items > 0)
      {

        LOGF(stderr, "extractOffset: %d items: %d stepSize: %d validCount: %d validList2.size: %d tempSize: %d buff: %d  i %d onego: %d tempOffset1: %d memOffset1: %d\n",
                     extractOffset, items, stepSize, validCount, validList2.get_size(),
                     tempSize, localTree.generalBuffer1.get_size(), i, doInOneGo,
                     tempOffset1, memOffset1);

        double t1 = get_time();
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<int>(0,    &extractOffset);
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<int>(1,    &items);
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(2, validList2.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(3, localTree.bodies_Ppos.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(4, localTree.bodies_Pvel.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(5, localTree.bodies_pos.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(6, localTree.bodies_vel.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(7, localTree.bodies_acc0.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(8, localTree.bodies_acc1.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(9, localTree.bodies_time.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(10, localTree.bodies_ids.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(11, localTree.bodies_key.p());
        extractOutOfDomainParticlesAdvancedSFC2.set_arg<cl_mem>(12, bodyBuffer.p());
        extractOutOfDomainParticlesAdvancedSFC2.setWork(items, 128);
        extractOutOfDomainParticlesAdvancedSFC2.printWorkSize();
        extractOutOfDomainParticlesAdvancedSFC2.execute(execStream->s());
        execStream->sync();
        double t2 = get_time();

        LOGF(stderr,"EXTR: %d \t %lg  size of body buf: %d Step: %d \n", items, t2-t1, sizeof(bodyStruct), i);

        bodyBuffer.d2h(items); // validCount);
        for(int i=0; i < items; i++)
        {
          fprintf(out,"%d\tPos: %f %f %f %f\tVel: %f %f %f %f\tKey: %d %d %d %d \n",
              i,
              bodyBuffer[i].pos.x,bodyBuffer[i].pos.y,bodyBuffer[i].pos.z,bodyBuffer[i].pos.w,
              bodyBuffer[i].vel.x,bodyBuffer[i].vel.y,bodyBuffer[i].vel.z,bodyBuffer[i].vel.w,
              bodyBuffer[i].key.x,bodyBuffer[i].key.y,bodyBuffer[i].key.z,bodyBuffer[i].key.w);
        }

      }
    }
    fclose(out);
   }
   
   MPI_Finalize(); exit(0);
#endif
}

