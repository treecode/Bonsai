#include "octree.h"

void octree::getBoundaries(tree_structure &tree, real4 &r_min, real4 &r_max)
{

  //Start reduction to get the boundary's of the system
  boundaryReduction.set_arg<int>(0, &tree.n);
  boundaryReduction.set_arg<cl_mem>(1, tree.bodies_Ppos.p());
  boundaryReduction.set_arg<cl_mem>(2, devMemRMIN.p());
  boundaryReduction.set_arg<cl_mem>(3, devMemRMAX.p());

  boundaryReduction.setWork(tree.n, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);  //256 threads and 120 blocks in total
  boundaryReduction.execute();
  
   
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
//     LOG("%f\t%f\t%f\t || \t%f\t%f\t%f\n", rMIN[i].x,rMIN[i].y,rMIN[i].z,rMAX[i].x,rMAX[i].y,rMAX[i].z);    
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
  boundaryReductionGroups.execute();

   
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
  

  rMinLocalTreeGroups = r_min;
  rMaxLocalTreeGroups = r_max;
  
  
  LOG("Found group boundarys after increase, number of groups %d : \n", tree.n_groups);
  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);
}



void octree::sort_bodies(tree_structure &tree, bool doDomainUpdate) {

  //We assume the bodies are already onthe GPU
  devContext.startTiming();
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
  srcValues.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[0], 0,  
                         tree.n, getAllignmentOffset(0));  
  
  //Compute the keys directly into srcValues 
  // will be sorted into tree.bodies_key below
  build_key_list.set_arg<cl_mem>(0,   srcValues.p());
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<int>(2,      &tree.n);
  build_key_list.set_arg<real4>(3,    &tree.corner);
  
  build_key_list.setWork(tree.n, 128); //128 threads per block
  build_key_list.execute();  
  
  // If srcValues and buffer are different, then the original values
  // are preserved, if they are the same srcValues will be overwritten  
  gpuSort(devContext, srcValues, tree.bodies_key,srcValues, tree.n, 32, 3, tree);

  devContext.stopTiming("Sorting", 0);  

  //Call the reorder data functions
  //First generate some memory buffers
  //generalBuffer is always at least 3xfloat4*N
  my_dev::dev_mem<real4>  real4Buffer1(devContext);
  my_dev::dev_mem<real4>  real4Buffer2(devContext);
  my_dev::dev_mem<real4>  real4Buffer3(devContext);
  
  real4Buffer1.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[0], 0,  
                         tree.n, getAllignmentOffset(0));  
    
  real4Buffer2.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[4*tree.n], 4*tree.n, 
                         tree.n, getAllignmentOffset(4*tree.n));   
  int prevOffset = getAllignmentOffset(4*tree.n);
  
  real4Buffer3.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[8*tree.n], 8*tree.n, 
                         tree.n, prevOffset + getAllignmentOffset(8*tree.n+prevOffset));   
  
  
  devContext.startTiming();
  
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
  dataReorderCombined.execute();
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
  dataReorderCombined.execute();

  tree.bodies_acc1.copy(real4Buffer1, tree.n);
  tree.bodies_Ppos.copy(real4Buffer2,  tree.n);
  tree.bodies_Pvel.copy(real4Buffer3, tree.n);   

  
  my_dev::dev_mem<int>  intBuffer(devContext);
  intBuffer.cmalloc_copy(tree.generalBuffer1.get_pinned(),   
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[4*tree.n], 4*tree.n,
                         tree.n, getAllignmentOffset(4*tree.n));  
  
  
  my_dev::dev_mem<float2>  float2Buffer(devContext);
  my_dev::dev_mem<int> sortPermutation(devContext);
  float2Buffer.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[0], 0,  
                         tree.n, getAllignmentOffset(0)); 
  sortPermutation.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[2*tree.n], 2*tree.n, 
                         tree.n, getAllignmentOffset(2*tree.n)); 
  
  dataReorderF2.set_arg<int>(0,      &tree.n);
  dataReorderF2.set_arg<cl_mem>(1,   tree.bodies_key.p());  
  
  dataReorderF2.set_arg<cl_mem>(2,   tree.bodies_time.p());
  dataReorderF2.set_arg<cl_mem>(3,   float2Buffer.p()); //Reuse as destination1
  dataReorderF2.set_arg<cl_mem>(4,   tree.bodies_ids.p()); 
  dataReorderF2.set_arg<cl_mem>(5,   sortPermutation.p()); //Reuse as destination2  
  dataReorderF2.setWork(tree.n, 512);   
  dataReorderF2.execute();
  
  
  tree.bodies_time.copy(float2Buffer, float2Buffer.get_size()); 
  tree.bodies_ids.copy(sortPermutation, sortPermutation.get_size());  



  devContext.stopTiming("Data-reordering", 1);    
}

