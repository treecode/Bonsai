#include "octree.h"


//External imports in order to call thrust or cub functions which have been compiled by nvcc
extern "C" void thrustDataReorderU4 (const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<uint4>  &dIn, my_dev::dev_mem<uint4>  &dOut);
extern "C" void thrustDataReorderF4 (const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float4> &dIn, my_dev::dev_mem<float4> &dOut);
extern "C" void thrustDataReorderF2 (const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float2> &dIn, my_dev::dev_mem<float2> &dOut);
extern "C" void thrustDataReorderULL(const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<ullong> &dIn, my_dev::dev_mem<ullong> &dOut);
extern "C" void thrustDataReorderF1 (const int N, my_dev::dev_mem<uint> &permutation, my_dev::dev_mem<float>  &dIn, my_dev::dev_mem<float>  &dOut);

extern "C" void thrustSort(my_dev::dev_mem<uint4> &srcKeys,
                           my_dev::dev_mem<uint>  &permutation_buffer,
                           my_dev::dev_mem<uint>  &temp_buffer,
                           int N);
extern "C" void  cubSort(my_dev::dev_mem<uint4>  &srcKeys,
                         my_dev::dev_mem<uint>   &outPermutation,
                         my_dev::dev_mem<char>   &tempBuffer,
                         my_dev::dev_mem<uint>   &tempB,
                         my_dev::dev_mem<uint>   &tempC,
                         my_dev::dev_mem<uint>   &tempD,
                         int  N) ;


void octree::getBoundaries(tree_structure &tree, real4 &r_min, real4 &r_max)
{
  //Start reduction to get the boundary's of the system
  boundaryReduction.set_arg<int>(0, &tree.n);
  boundaryReduction.set_arg<cl_mem>(1, tree.bodies_Ppos.p());
  boundaryReduction.set_arg<cl_mem>(2, devMemRMIN.p());
  boundaryReduction.set_arg<cl_mem>(3, devMemRMAX.p());

  //devMemRMIN.zeroMem(); devMemRMAX.zeroMem();

  boundaryReduction.setWork(tree.n, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);  //256 threads and 120 blocks in total
  boundaryReduction.execute(execStream->s());
  

  devMemRMIN.d2h();     //Need to be defined and initialized somewhere outside this function
  devMemRMAX.d2h();     //Need to be defined and initialized somewhere outside this function
//  r_min = make_real4(devMemRMIN[0].x, devMemRMIN[0].y, devMemRMIN[0].z, +1e10f);
//  r_max = make_real4(devMemRMAX[0].x, devMemRMAX[0].y, devMemRMAX[0].z, -1e10f);
  
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
//  LOG("2min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", devMemRMIN[120].x, devMemRMIN[120].y, devMemRMIN[120].z,devMemRMAX[120].x, devMemRMAX[120].y,devMemRMAX[120].z);


  //  FILE *fout = fopen("boundaries.txt","w");
  //  tree.bodies_Ppos.d2h();
  //  fprintf(fout,"#items %d\n", tree.n);
  //  fprintf(fout,"#idx\tX\tY\tZ\n");
  //  for(int i=0; i < tree.n; i++)
  //  {
  //    fprintf(fout,"%f\t%f\t%f\n", tree.bodies_Ppos[i].x, tree.bodies_Ppos[i].y, tree.bodies_Ppos[i].z);
  //  }
  //  fprintf(fout,"#results minx miny minz maxx maxy maxz\n");
  //  fprintf(fout,"%f\t%f\t%f\t%f\t%f\t%f\n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);
  //  fclose(fout);
//    exit(0);
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
//
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

  //We assume the bodies are already on the GPU
  devContext.startTiming(execStream->s());
  real4 r_min = {+1e10, +1e10, +1e10, +1e10}; 
  real4 r_max = {-1e10, -1e10, -1e10, -1e10};   
  
  if(doDomainUpdate)
  {
    getBoundaries(tree, r_min, r_max);  
    //Sync the boundary over the various processes
    if(this->mpiGetNProcs() > 1) { this->sendCurrentRadiusInfo(r_min, r_max); }
    rMinGlobal = r_min;    rMaxGlobal = r_max;
  }
  
  r_min = rMinGlobal;
  r_max = rMaxGlobal;
  
  //Compute the boundary's of the tree
  real size     = 1.001f*std::max(r_max.z - r_min.z,
                         std::max(r_max.y - r_min.y, r_max.x - r_min.x));
  
  tree.corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                             0.5f*(r_min.y + r_max.y) - 0.5f*size,
                             0.5f*(r_min.z + r_max.z) - 0.5f*size, size); 
       
  tree.domain_fac = size/(1 << MAXLEVELS);
  tree.corner.w   = tree.domain_fac;


  LOG("Corner: %f %f %f idomain fac: %f domain_fac: %f\n", 
         tree.corner.x, tree.corner.y, tree.corner.z, 1.0f/tree.domain_fac, tree.domain_fac);
  LOG("size: %f MAXLEVELS: %d \n", size, MAXLEVELS);

  //Call the GPUSort function, and give it the to be sorted arrays and scratch space
  my_dev::dev_mem<uint4>  srcValues(devContext);

  my_dev::dev_mem<uint> tempB(devContext), tempC(devContext),  tempD(devContext);
  my_dev::dev_mem<char> tempE(devContext);
  //The generalBuffer1 has size uint*4*N*3 = uint*12*N
  int genBufOffset2 = 0;

  genBufOffset2 = srcValues.cmalloc_copy(tree.generalBuffer1, tree.n, 0);  //uint*N -uint5*N
  genBufOffset2 = tempB    .cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset2); //uint5*N-uint6*N
  genBufOffset2 = tempC    .cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset2); //uint6*N-uint7*N
  genBufOffset2 = tempD    .cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset2); //uint7*N-uint8*N
  genBufOffset2 = tempE    .cmalloc_copy(tree.generalBuffer1, tree.n, genBufOffset2); //uint8*N-uint9*N

  //Compute the keys directly into srcValues will be sorted into tree.bodies_key below
  build_key_list.set_arg<cl_mem>(0,   srcValues.p());
  build_key_list.set_arg<cl_mem>(1,   tree.bodies_Ppos.p());
  build_key_list.set_arg<int>(2,      &tree.n);
  build_key_list.set_arg<real4>(3,    &tree.corner);
  build_key_list.setWork(tree.n, 128); //128 threads per block
  build_key_list.execute(execStream->s());

//  execStream->sync();
  
#if 0
  srcValues.d2h();
  for(int i=0; i < tree.n; i++)
  {
      fprintf(stderr,"PRE: %d\t\t%d\t%d\t%d\t%d\n",
          i, srcValues[i].x,srcValues[i].y, srcValues[i].z, srcValues[i].w);
      if(i > 10) break;
  }

#endif

  // If srcValues and buffer are different, then the original values
  // are preserved, if they are the same srcValues will be overwritten
  if(tree.n > 0) gpuSort(srcValues, tree.oriParticleOrder, tempB, tempC, tempD, tempE, tree.n);
  this->dataReorder(tree.n, tree.oriParticleOrder, srcValues, tree.bodies_key);

#if 0
  tree.bodies_key.d2h();
  for(int i=0; i < tree.n; i++)
  {
      fprintf(stderr,"Out-ori: %d\t\t%d\t%d\t%d\t%d\n",
          i, tree.bodies_key[i].x,tree.bodies_key[i].y, tree.bodies_key[i].z, tree.bodies_key[i].w);
      if(i > 10) break;
  }
//  exit(0);
#endif


  devContext.stopTiming("Sorting", 0, execStream->s());  

  //Call the reorder data functions
  devContext.startTiming(execStream->s());

  //JB this if statement is required until I fix the order
  //of functions in main.cpp  

  if(!doFullShuffle)
  {
    my_dev::dev_mem<real4>  real4Buffer1(devContext);
    my_dev::dev_mem<ullong> ullBuffer(devContext);
    my_dev::dev_mem<float>  realBuffer(devContext);

    real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    ullBuffer.   cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    realBuffer.  cmalloc_copy(tree.generalBuffer1, tree.n, 0);

    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_Ppos, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_ids, ullBuffer);

    //Density values
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_h, realBuffer);
  }
  else
  {
    //Call the reorder data functions
    //generalBuffer is always at least 3xfloat4*N
    my_dev::dev_mem<real4>    real4Buffer1(devContext);
    my_dev::dev_mem<float2>   float2Buffer(devContext);
    my_dev::dev_mem<ullong>   ullBuffer(devContext);
    my_dev::dev_mem<float>    realBuffer(devContext);
    real4Buffer1.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    float2Buffer.cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    ullBuffer.   cmalloc_copy(tree.generalBuffer1, tree.n, 0);
    realBuffer.  cmalloc_copy(tree.generalBuffer1, tree.n, 0);

    //Position, velocity and acc0
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_pos, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_vel, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_acc0, real4Buffer1);

    //Acc1, Predicted position and velocity
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_acc1, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_Ppos, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_Pvel, real4Buffer1);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_time, float2Buffer);
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_ids, ullBuffer);

    //Density values
    dataReorder(tree.n, tree.oriParticleOrder, tree.bodies_h, realBuffer);

  } //end if
  
  devContext.stopTiming("Data-reordering", 1, execStream->s());

//  exit(0);
}
//iter=15 : time= 1  Etot= -0.2453192142  Ekin= 0.242805   Epot= -0.488124 : de= -6.43185e-05 ( 6.43185e-05 ) d(de)= -0 ( 6.76109e-06 ) t_sim=  2.52183 sec



/*
Sort an array of int4, the idea is that the key is somehow moved into x/y/z and the
value is put in w...
Sorts values based on the last item so order becomes something like:
z y x
2 2 1
2 1 2
2 3 3
2 5 3

*/


//Input keys, output a permutation that presents the new order
void octree::gpuSort(my_dev::dev_mem<uint4> &srcKeys,
                     my_dev::dev_mem<uint>  &permutation, //For 32bit values
                     my_dev::dev_mem<uint>  &tempB,       //For 32bit values
                     my_dev::dev_mem<uint>  &tempC,       //For 32bit keys
                     my_dev::dev_mem<uint>  &tempD,       //For 32bit keys
                     my_dev::dev_mem<char>  &tempE,       //For sorting space
                     int N)
{
//#define USE_CUB
  #ifdef USE_CUB
    cubSort(srcKeys, permutation, tempE, tempB, tempC, tempD, N);
  #else
    thrustSort(srcKeys,permutation, tempB, N);
  #endif
}


//Pass the buffers on to the thrust::gather functions
template<typename T> void octree::dataReorder(const int              N,
                                              my_dev::dev_mem<uint> &permutation,
                                              my_dev::dev_mem<T>    &dIn,
                                              my_dev::dev_mem<T>    &scratch,
                                              bool                   overwrite)
{
  dataReorder2(N, permutation, dIn, scratch);
  if(overwrite)  dIn.copy(scratch,  N);
}

//Predefined templates to point to the correct external functions
template<> void octree::dataReorder2<uint4>(const int N, my_dev::dev_mem<uint> &permutation,
                                 my_dev::dev_mem<uint4>  &dIn, my_dev::dev_mem<uint4>  &dOut) {
  thrustDataReorderU4(N, permutation, dIn, dOut);
}
template<> void octree::dataReorder2<float4>(const int N, my_dev::dev_mem<uint> &permutation,
                                  my_dev::dev_mem<float4>  &dIn, my_dev::dev_mem<float4>  &dOut) {
  thrustDataReorderF4(N, permutation, dIn, dOut);
}

template<> void octree::dataReorder2<float2>(const int N, my_dev::dev_mem<uint> &permutation,
                                  my_dev::dev_mem<float2>  &dIn, my_dev::dev_mem<float2>  &dOut) {
  thrustDataReorderF2(N, permutation, dIn, dOut);
}
template<> void octree::dataReorder2<float>(const int N, my_dev::dev_mem<uint> &permutation,
                                 my_dev::dev_mem<float>  &dIn, my_dev::dev_mem<float>  &dOut) {
  thrustDataReorderF1(N, permutation, dIn, dOut);
}

template<> void octree::dataReorder2<ullong>(const int N, my_dev::dev_mem<uint> &permutation,
                                  my_dev::dev_mem<ullong>  &dIn, my_dev::dev_mem<ullong>  &dOut) {
  thrustDataReorderULL(N, permutation, dIn, dOut);
}

