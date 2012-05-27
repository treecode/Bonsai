#include "octree.h"

#if defined(USE_B40C)
#include "sort.h"
#elif defined(USE_THRUST)
#define USE_THRUST_96
  extern "C" void thrust_sort_32b(my_dev::context &devContext, 
                                  my_dev::dev_mem<uint> &srcKeys,     my_dev::dev_mem<uint> &srcValues,
                                  my_dev::dev_mem<int>  &keysOutput,  my_dev::dev_mem<uint> &keysAPing,
                                  my_dev::dev_mem<uint> &valuesOutput,my_dev::dev_mem<uint> &valuesAPing,
                                  int N, int numberOfBits);
  extern "C" void thrust_sort_96b(my_dev::dev_mem<uint4> &srcKeys, 
                                  my_dev::dev_mem<uint4> &sortedKeys,
                                  my_dev::dev_mem<uint>  &temp_buffer,
                                  my_dev::dev_mem<uint>  &permutation_buffer,
                                  int N);
  extern "C" void thrust_gpuCompact(my_dev::context &devContext, 
                                    my_dev::dev_mem<uint> &srcValues,
                                    my_dev::dev_mem<uint> &output,                        
                                    int N, int *validCount);
#endif

void octree::set_context( bool disable_timing) {
  
  devContext.create(disable_timing);
  set_context2();
}


void octree::set_context(std::ofstream &log, bool disable_timing) {  
  
  devContext.create(log, disable_timing);
  set_context2();
}

void octree::set_context2()
{
  devContext.createQueue(devID);
  
  devContext_flag = true;
  
  //Assign the context to the local tree
  this->localTree.setContext(devContext);
  
  //And for the remote tree
  this->remoteTree.setContext(devContext);

}

void octree::load_kernels() {

  if (!devContext_flag) set_context();
  
  //If we arive here we have aquired a device, configure parts of the code
  
  //Get the number of multiprocessors and compute number of 
  //blocks to be used during the tree-walk
  nMultiProcessors      = devContext.multiProcessorCount;
  const int blocksPerSM = getTreeWalkBlocksPerSM(
                          this->getDevContext()->getComputeCapabilityMajor(),
                          this->getDevContext()->getComputeCapabilityMinor());
 
  nBlocksForTreeWalk = nMultiProcessors*blocksPerSM;
  

  std::string pathName;


  //AMUSE specific
  if(this->src_directory != NULL)
  {
    pathName.assign(this->src_directory);
  }
  else
  {  
    //Strip the executable name, to get the path name
    std::string temp(execPath);
    int idx = (int)temp.find_last_of("/\\");
    pathName.assign(temp.substr(0, idx+1));
  }
  

  // load scan & sort kernels

  compactCount.setContext(devContext);
  exScanBlock.setContext(devContext);
  compactMove.setContext(devContext);
  splitMove.setContext(devContext);
  sortCount.setContext(devContext);
  sortMove.setContext(devContext);
  extractInt.setContext(devContext);
  reOrderKeysValues.setContext(devContext);
  convertKey64to96.setContext(devContext);
  extractKeyAndPerm.setContext(devContext);
  dataReorderR4.setContext(devContext);
  dataReorderF2.setContext(devContext);
  dataReorderI1.setContext(devContext);
  dataReorderCombined.setContext(devContext);
  
#ifdef USE_CUDA
  compactCount.load_source("./scanKernels.ptx", pathName.c_str());
  compactCount.create("compact_count");
  
  exScanBlock.load_source("./scanKernels.ptx", pathName.c_str());
  exScanBlock.create("exclusive_scan_block");
  
  compactMove.load_source("./scanKernels.ptx", pathName.c_str());
  compactMove.create("compact_move");
  
  splitMove.load_source("./scanKernels.ptx", pathName.c_str());
  splitMove.create("split_move");
  
  sortCount.load_source("./sortKernels.ptx", pathName.c_str());
  sortCount.create("sort_count");

  sortMove.load_source("./sortKernels.ptx", pathName.c_str());
  sortMove.create("sort_move_stage_key_value");  

  extractInt.load_source("./sortKernels.ptx", pathName.c_str());
  extractInt.create("extractInt");  
  
  reOrderKeysValues.load_source("./sortKernels.ptx", pathName.c_str());
  reOrderKeysValues.create("reOrderKeysValues");    
  
  extractKeyAndPerm.load_source("./sortKernels.ptx", pathName.c_str());
  extractKeyAndPerm.create("extractKeyAndPerm");  
  
  convertKey64to96.load_source("./sortKernels.ptx", pathName.c_str());
  convertKey64to96.create("convertKey64to96");
  
  dataReorderR4.load_source("./sortKernels.ptx", pathName.c_str());
//  dataReorderR4.create("dataReorderR4");  
  dataReorderR4.create("dataReorderCombined4");
  
  dataReorderF2.load_source("./sortKernels.ptx", pathName.c_str());
  dataReorderF2.create("dataReorderF2");  

  dataReorderI1.load_source("./sortKernels.ptx", pathName.c_str());
  dataReorderI1.create("dataReorderI1");        
  
  dataReorderCombined.load_source("./sortKernels.ptx", pathName.c_str());
  dataReorderCombined.create("dataReorderCombined");
//  dataReorderCombined.create("dataReorderCombined4");
  
  
#else
  compactCount.load_source("scanKernels.cl", "OpenCLKernels");
  compactCount.create("compact_count");
  
  exScanBlock.load_source("scanKernels.cl", "OpenCLKernels");
  exScanBlock.create("exclusive_scan_block");
  
  compactMove.load_source("scanKernels.cl", "OpenCLKernels");
  compactMove.create("compact_move");
  
  splitMove.load_source("scanKernels.cl", "OpenCLKernels");
  splitMove.create("split_move");
#endif


  // load tree-build kernels
  
  /* set context */
  
  build_key_list.setContext(devContext);
  build_valid_list.setContext(devContext);
  build_nodes.setContext(devContext);
  link_tree.setContext(devContext);
  define_groups.setContext(devContext);
  build_level_list.setContext(devContext);
  boundaryReduction.setContext(devContext);
  boundaryReductionGroups.setContext(devContext);  
  build_body2group_list.setContext(devContext);
  store_groups.setContext(devContext);
  
  /* load kernels tree properties */
  
#ifdef USE_CUDA
  build_key_list.load_source("./build_tree.ptx", pathName.c_str());
  build_valid_list.load_source("./build_tree.ptx", pathName.c_str());
  build_nodes.load_source("./build_tree.ptx", pathName.c_str());
  link_tree.load_source("./build_tree.ptx", pathName.c_str());
  define_groups.load_source("./build_tree.ptx", pathName.c_str());
  build_level_list.load_source("./build_tree.ptx", pathName.c_str());
  boundaryReduction.load_source("./build_tree.ptx", pathName.c_str());
  boundaryReductionGroups.load_source("./build_tree.ptx", pathName.c_str());
  build_body2group_list.load_source("./build_tree.ptx", pathName.c_str());
  store_groups.load_source("./build_tree.ptx", pathName.c_str());
  
  /* create kernels */

  build_key_list.create("cl_build_key_list");  
  build_valid_list.create("cl_build_valid_list");
  build_nodes.create("cl_build_nodes");
  link_tree.create("cl_link_tree");
  define_groups.create("build_group_list2");
  build_level_list.create("build_level_list");
  boundaryReduction.create("boundaryReduction");
  boundaryReductionGroups.create("boundaryReductionGroups");
  build_body2group_list.create("build_body2group_list");
  store_groups.create("store_group_list");

#else
  build_key_list.load_source("build_tree.cl", "");
  build_valid_list.load_source("build_tree.cl", "");
  build_nodes.load_source("build_tree.cl", "");
  link_tree.load_source("build_tree.cl", "");
  
  /* create kernels */

  build_key_list.create("cl_build_key_list");
  build_valid_list.create("cl_build_valid_list");
  build_nodes.create("cl_build_nodes");
  link_tree.create("cl_link_tree");
#endif

  // load tree-props kernels
  propsNonLeafD.setContext(devContext);
  propsLeafD.setContext(devContext);
  propsScalingD.setContext(devContext);
  
  copyNodeDataToGroupData.setContext(devContext);
 
  /* load kernels */
  
#ifdef USE_CUDA

  propsNonLeafD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "", -1);
  propsLeafD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "", -1);
  propsScalingD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "",-1);
  
  copyNodeDataToGroupData.load_source("./compute_propertiesD.ptx", pathName.c_str());

  /* create kernels */
  
  propsNonLeafD.create("compute_non_leaf"); 
  propsLeafD.create("compute_leaf");
  propsScalingD.create("compute_scaling");

  copyNodeDataToGroupData.create("setPHGroupData");
  
#else
  propsNonLeaf.load_source("compProps.cl", "");
  propsLeaf.load_source("compProps.cl", "");
  propsScaling.load_source("compProps.cl", ""); 
  
  /* create kernels */

  propsNonLeaf.create("compute_non_leaf");
  propsLeaf.create("compute_leaf");
  propsScaling.create("compute_scaling");  
#endif

  /* Tree iteration */
  getTNext.setContext(devContext);
  predictParticles.setContext(devContext);
  getNActive.setContext(devContext);
  approxGrav.setContext(devContext);
  directGrav.setContext(devContext);
  correctParticles.setContext(devContext);
  computeDt.setContext(devContext);
  computeEnergy.setContext(devContext);
  setActiveGrps.setContext(devContext);
  distanceCheck.setContext(devContext);  


  approxGravLET.setContext(devContext);

#ifdef USE_CUDA
  getTNext.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  predictParticles.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  getNActive.load_source("./timestep.ptx", pathName.c_str(), "", -1);
	approxGrav.load_source("./dev_approximate_gravity.ptx", pathName.c_str(), "", 64);
  directGrav.load_source("./dev_direct_gravity.ptx", pathName.c_str(), "", 64);
  correctParticles.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  computeDt.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  computeEnergy.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  setActiveGrps.load_source("./timestep.ptx", pathName.c_str(), "", -1);
  distanceCheck.load_source("./timestep.ptx", pathName.c_str(), "", -1);  
  
  approxGravLET.load_source("./dev_approximate_gravity_let.ptx", pathName.c_str(), "", 64);  
  /* create kernels */

  getTNext.create("get_Tnext"); 
  predictParticles.create("predict_particles");   
  getNActive.create("get_nactive");
  approxGrav.create("dev_approximate_gravity");
  directGrav.create("dev_direct_gravity");
  correctParticles.create("correct_particles");
  computeDt.create("compute_dt");  
  setActiveGrps.create("setActiveGroups");

  computeEnergy.create("compute_energy_double");  
  distanceCheck.create("distanceCheck");  
  
  approxGravLET.create("dev_approximate_gravity_let");

#else
  getTNext.load_source("", "");
  
  /* create kernels */

  getTNext.create("");
  
#endif

  //Parallel kernels
  domainCheck.setContext(devContext);  
  extractSampleParticles.setContext(devContext);  
  extractOutOfDomainR4.setContext(devContext);  
  extractOutOfDomainBody.setContext(devContext);  
  insertNewParticles.setContext(devContext);  
  internalMove.setContext(devContext);  

#ifdef USE_CUDA
  domainCheck.load_source("./parallel.ptx", pathName.c_str());
  extractSampleParticles.load_source("./parallel.ptx", pathName.c_str());
  extractOutOfDomainR4.load_source("./parallel.ptx", pathName.c_str());
  extractOutOfDomainBody.load_source("./parallel.ptx", pathName.c_str());
  insertNewParticles.load_source("./parallel.ptx", pathName.c_str());
  internalMove.load_source("./parallel.ptx", pathName.c_str());
  
  domainCheck.create("doDomainCheck");
  extractSampleParticles.create("extractSampleParticles");
  extractOutOfDomainR4.create("extractOutOfDomainParticlesR4");
  extractOutOfDomainBody.create("extractOutOfDomainParticlesAdvanced");
  insertNewParticles.create("insertNewParticles");
  internalMove.create("internalMove");

#else


#endif
  
#ifdef USE_DUST
   define_dust_groups.setContext(devContext);
   define_dust_groups.load_source("./build_tree.ptx", pathName.c_str());
   define_dust_groups.create("define_dust_groups");
   
   store_dust_groups.setContext(devContext);
   store_dust_groups.load_source("./build_tree.ptx", pathName.c_str());
   store_dust_groups.create("store_dust_groups");
   
   predictDust.setContext(devContext);
   predictDust.load_source("./build_tree.ptx", pathName.c_str());
   predictDust.create("predict_dust_particles");   
   
   correctDust.setContext(devContext);
   correctDust.load_source("./build_tree.ptx", pathName.c_str());
   correctDust.create("correct_dust_particles");     
   
   
   
#endif  
  
  

}

void octree::resetCompact()
{
  // reset counts to 1 so next compact proceeds...
  cudaStreamSynchronize(execStream->s()); // must make sure any outstanding compact is finished  
  this->devMemCountsx[0] = 1;
  this->devMemCountsx.h2d(1, false, copyStream->s());
}

//Compacts an array of integers, the values in srcValid indicate if a
//value is valid (1 == valid anything else is UNvalid) returns the 
//compacted values in the output array and the total 
//number of valid items is stored in 'count' 
void octree::gpuCompact(my_dev::context &devContext, 
                        my_dev::dev_mem<uint> &srcValues,
                        my_dev::dev_mem<uint> &output,                        
                        int N, 
                        int *validCount) // if validCount NULL leave count on device
{

// thrust_gpuCompact(devContext, 
//                   srcValues,
//                   output,                        
//                   N, validCount);
  
  //Memory that should be alloced outside the function:
  //devMemCounts and devMemCountsx 

  // make sure previous reset has finished.
  this->devMemCountsx.waitForCopyEvent();

  //Kernel configuration parameters
  setupParams sParam;
  sParam.jobs = (N / 64) / 480  ; //64=32*2 2 items per look, 480 is 120*4, number of procs
  sParam.blocksWithExtraJobs = (N / 64) % 480; 
  sParam.extraElements = N % 64;
  sParam.extraOffset = N - sParam.extraElements;

  compactCount.set_arg<cl_mem>(0, srcValues.p());
  compactCount.set_arg<cl_mem>(1, this->devMemCounts.p());
  compactCount.set_arg<uint>(2, &N);
  compactCount.set_arg<int>(3, NULL, 128);
  compactCount.set_arg<setupParams>(4, &sParam);
  compactCount.set_arg<cl_mem>(5, this->devMemCountsx.p());

  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 32*120;   globalWork[1] = 4;
  localWork [0] = 32;       localWork[1] = 4;   
  compactCount.setWork(globalWork, localWork);

  ///////////////

  exScanBlock.set_arg<cl_mem>(0, this->devMemCounts.p());  
  int blocks = 120*4;
  exScanBlock.set_arg<int>(1, &blocks);
  exScanBlock.set_arg<cl_mem>(2, this->devMemCountsx.p());
  exScanBlock.set_arg<int>(3, NULL, 512); //shared memory allocation
  

  globalWork[0] = 512; globalWork[1] = 1;
  localWork [0] = 512; localWork [1] = 1;

  exScanBlock.setWork(globalWork, localWork);

  //////////////

  compactMove.set_arg<cl_mem>(0, srcValues.p());
  compactMove.set_arg<cl_mem>(1, output.p());
  compactMove.set_arg<cl_mem>(2, this->devMemCounts.p());
  compactMove.set_arg<uint>(3, &N);
  compactMove.set_arg<uint>(4, NULL, 192); //Dynamic shared memory
  compactMove.set_arg<setupParams>(5, &sParam);
  compactMove.set_arg<cl_mem>(6, this->devMemCountsx.p());

  globalWork[0] = 120*32;  globalWork[1] = 4;
  localWork [0] = 32;      localWork [1] = 4;

  compactMove.setWork(globalWork, localWork);

  ////////////////////

  compactCount.execute(execStream->s());
  exScanBlock.execute(execStream->s());
  compactMove.execute(execStream->s());
  
  if (validCount)
  {
    this->devMemCountsx.d2h();
    *validCount = this->devMemCountsx[0];
    //printf("Total number of valid items: %d \n", countx[0]);
  }
}

//Splits an array of integers, the values in srcValid indicate if a
//value is valid (1 == valid anything else is UNvalid) returns the 
//splitted values in the output array (first all valid 
//number and then the invalid ones) and the total
//number of valid items is stored in 'count' 
void octree::gpuSplit(my_dev::context &devContext, 
                      my_dev::dev_mem<uint> &srcValues,
                      my_dev::dev_mem<uint> &output,                        
                      int N, 
                      int *validCount)  // if validCount NULL leave count on device
{

  //In the next step we associate the GPU memory with the Kernel arguments
  //my_dev::dev_mem<uint> counts(devContext, 512), countx(devContext, 512);
  //Memory that should be alloced outside the function:
  //devMemCounts and devMemCountsx 
  
  // make sure previous reset has finished.
  this->devMemCountsx.waitForCopyEvent();

  //Kernel configuration parameters
  setupParams sParam;
  sParam.jobs = (N / 64) / 480  ; //64=32*2 2 items per look, 480 is 120*4, number of procs
  sParam.blocksWithExtraJobs = (N / 64) % 480; 
  sParam.extraElements = N % 64;
  sParam.extraOffset = N - sParam.extraElements;
  
  compactCount.set_arg<cl_mem>(0, srcValues.p());
  compactCount.set_arg<cl_mem>(1, this->devMemCounts.p());
  compactCount.set_arg<uint>(2, &N);
  compactCount.set_arg<int>(3, NULL, 128);
  compactCount.set_arg<setupParams>(4, &sParam);
  compactCount.set_arg<cl_mem>(5, this->devMemCountsx.p());
  
  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 32*120;   globalWork[1] = 4;
  localWork [0] = 32;       localWork[1] = 4;   
  compactCount.setWork(globalWork, localWork);

  ///////////////

  exScanBlock.set_arg<cl_mem>(0, this->devMemCounts.p());  
  int blocks = 120*4;
  exScanBlock.set_arg<int>(1, &blocks);
  exScanBlock.set_arg<cl_mem>(2, this->devMemCountsx.p());
  exScanBlock.set_arg<int>(3, NULL, 512); //shared memory allocation

  globalWork[0] = 512; globalWork[1] = 1;
  localWork [0] = 512; localWork [1] = 1;

  exScanBlock.setWork(globalWork, localWork);

  //////////////

  splitMove.set_arg<cl_mem>(0, srcValues.p());
  splitMove.set_arg<cl_mem>(1, output.p());
  splitMove.set_arg<cl_mem>(2, this->devMemCounts.p());
  splitMove.set_arg<uint>(3, &N);
  splitMove.set_arg<uint>(4, NULL, 192); //Dynamic shared memory
  splitMove.set_arg<setupParams>(5, &sParam);
  
  globalWork[0] = 120*32;  globalWork[1] = 4;
  localWork [0] = 32;      localWork [1] = 4;

  splitMove.setWork(globalWork, localWork);

  ////////////////////
  compactCount.execute(execStream->s());
  exScanBlock.execute(execStream->s());
  splitMove.execute(execStream->s());

  if (validCount) {
    this->devMemCountsx.d2h();
    *validCount = this->devMemCountsx[0];
  }
}


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

// If srcValues and buffer are different, then the original values
// are preserved, if they are the same srcValues will be overwritten
void  octree::gpuSort(my_dev::context &devContext,
                      my_dev::dev_mem<uint4> &srcValues,
                      my_dev::dev_mem<uint4> &output,
                      my_dev::dev_mem<uint4> &buffer,
                      int N, int numberOfBits, int subItems,
                      tree_structure &tree) {

#if defined (USE_B40C)
  sorter->sort(srcValues, output, N);

#elif defined(USE_THRUST) && defined(USE_THRUST_96)
  //Extra buffer values
  my_dev::dev_mem<uint> permutation(devContext);   // Permutation values, for sorting the int4 data
  my_dev::dev_mem<uint> temp_buffer(devContext);  // temporary uint buffer
  
  //Permutation has to be allocated after the two previous
  //allocated buffers, get the right offset
  int memOffset  = permutation.getGlobalMemAllignmentPadding(8*N);
      memOffset += 8*N; 

      memOffset = permutation.cmalloc_copy(tree.generalBuffer1, N, memOffset);
      memOffset = temp_buffer.cmalloc_copy(tree.generalBuffer1, N, memOffset);      
      
  thrust_sort_96b(srcValues, output, temp_buffer, permutation, N);
  
#else
  //Extra buffer values
  my_dev::dev_mem<uint> simpleKeys(devContext);    //Int keys,
  my_dev::dev_mem<uint> permutation(devContext);   //Permutation values, for sorting the int4 data
  my_dev::dev_mem<int>  output32b(devContext);       //Permutation values, for sorting the int4 data
  my_dev::dev_mem<uint> valuesOutput(devContext);  //Buffers for the values which are the indexes
  
  //Permutation has to be allocated after the two previous
  //allocated buffers, get the right offset
  int memOffset = simpleKeys.getGlobalMemAllignmentPadding(8*N);
      memOffset += 8*N; 
      memOffset = simpleKeys.cmalloc_copy(tree.generalBuffer1, N, memOffset);
      memOffset = permutation.cmalloc_copy(tree.generalBuffer1, N, memOffset);   
      memOffset = output32b.cmalloc_copy(tree.generalBuffer1, N, memOffset); 
      memOffset = valuesOutput.cmalloc_copy(tree.generalBuffer1, N, memOffset); 
  
    
  //Dimensions for the kernels that shuffle and extract data
  const int blockSize = 256;
  
  extractInt.setWork(N, blockSize); 
  reOrderKeysValues.setWork(N, blockSize); 

  //Idx depends on subitems, z goes first, x last if subitems = 3
  //subitems = 3, than idx=2
  //subitems = 2, than idx=1
  //subitems = 1, than idx=0
  //intIdx = subItems-1   
  int intIdx = subItems-1;

  //Extracts a 32bit key and fills a sequence
  extractInt.set_arg<cl_mem>(0, srcValues.p());
  extractInt.set_arg<cl_mem>(1, simpleKeys.p());
  extractInt.set_arg<cl_mem>(2, permutation.p());
  extractInt.set_arg<uint>(3, &N);
  extractInt.set_arg<int>(4, &intIdx);//bit idx


  reOrderKeysValues.set_arg<cl_mem>(0, srcValues.p());
  reOrderKeysValues.set_arg<cl_mem>(1, output.p());
  reOrderKeysValues.set_arg<cl_mem>(2, valuesOutput.p());
  reOrderKeysValues.set_arg<uint>(3, &N);

  extractInt.execute(execStream->s());
  
  #ifdef USE_THRUST
  
  thrust_sort_32b(devContext, 
                   simpleKeys, permutation,
                   output32b, simpleKeys,
                   valuesOutput,permutation,
                   N, 32);
  
  #else
    //Now sort the first 32bit keys
    //Using 32bit sort with key and value seperated    
    gpuSort_32b(devContext, 
                    simpleKeys, permutation,
                    output32b, simpleKeys,
                    valuesOutput,permutation,
                    N, 32);
  #endif  

    
  //Now reorder the main keys
  //Use output as the new output/src value thing buffer
  reOrderKeysValues.execute(execStream->s());
  
  if(subItems == 1)
  {
    //Only doing one 32bit sort. Data is already in output so done
    return;
  }


  //2nd set of 32bit keys
  //Idx depends on subitems, z goes first, x last if subitems = 3  
  //subitems = 3, than idx=1
  //subitems = 2, than idx=0
  //subitems = 1, completed previous round
  //intIdx = subItems-2   
  intIdx = subItems-2;
  
  extractInt.set_arg<cl_mem>(0, output.p());
  extractInt.set_arg<int>(4, &intIdx);//smem size
  extractInt.execute(execStream->s());

  #ifdef USE_THRUST
  
    thrust_sort_32b(devContext, 
                    simpleKeys, permutation,
                    output32b, simpleKeys,
                    valuesOutput,permutation,
                    N, 32);
  
  #else
    //Now sort the 2nd 32bit keys
    //Using 32bit sort with key and value seperated    
    gpuSort_32b(devContext, 
                    simpleKeys, permutation,
                    output32b, simpleKeys,
                    valuesOutput,permutation,
                    N, 32);
  #endif   

  reOrderKeysValues.set_arg<cl_mem>(0, output.p());
  reOrderKeysValues.set_arg<cl_mem>(1, buffer.p());
  reOrderKeysValues.execute(execStream->s());

  if(subItems == 2)
  {
    //Doing two 32bit sorts. Data is in buffer
    //so move the data from buffer to output    
    output.copy(buffer, buffer.get_size());    
    return;
  }

  //3th set of 32bit keys
  //Idx depends on subitems, z goes first, x last if subitems = 3  
  //subitems = 3, than idx=0
  //subitems = 2, completed previous round
  //subitems = 1, completed previous round
  //intIdx = subItems-2     
  intIdx = 0;
 
  extractInt.set_arg<cl_mem>(0, buffer.p());
  extractInt.set_arg<int>(4, &intIdx);//integer idx
  extractInt.execute(execStream->s());


  //Now sort the final set of 32bit keys
  #ifdef USE_THRUST  
    thrust_sort_32b(devContext, 
                    simpleKeys, permutation,
                    output32b, simpleKeys,
                    valuesOutput,permutation,
                    N, 32);
  
  #else
    gpuSort_32b(devContext, 
                    simpleKeys, permutation,
                    output32b, simpleKeys,
                    valuesOutput,permutation,
                    N, 32);
  #endif   
  
  reOrderKeysValues.set_arg<cl_mem>(0, buffer.p());
  reOrderKeysValues.set_arg<cl_mem>(1, output.p());
  reOrderKeysValues.execute(execStream->s());  
#endif // USE_THRUST_96
}


void octree::gpuSort_32b(my_dev::context &devContext, 
                    my_dev::dev_mem<uint> &srcKeys,     my_dev::dev_mem<uint> &srcValues,
                    my_dev::dev_mem<int>  &keysOutput,  my_dev::dev_mem<uint> &keysAPing,
                    my_dev::dev_mem<uint> &valuesOutput,my_dev::dev_mem<uint> &valuesAPing,
                    int N, int numberOfBits)
{

  int bitIdx = 0;

  //Step 1, do the count
  //Memory that should be alloced outside the function:

  setupParams sParam;
  sParam.jobs = (N / 64) / 480  ; //64=32*2 2 items per look, 480 is 120*4, number of procs
  sParam.blocksWithExtraJobs = (N / 64) % 480;
  sParam.extraElements = N % 64;
  sParam.extraOffset = N - sParam.extraElements;

  sortCount.set_arg<cl_mem>(0, srcKeys.p());
  sortCount.set_arg<cl_mem>(1, this->devMemCounts.p());
  sortCount.set_arg<uint>(2, &N);
  sortCount.set_arg<int>(3, NULL, 128);//smem size
  sortCount.set_arg<setupParams>(4, &sParam);
  sortCount.set_arg<int>(5, &bitIdx);
  
  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 32*120;   globalWork[1] = 4;
  localWork [0] = 32;       localWork[1] = 4;
  sortCount.setWork(globalWork, localWork);

  ///////////////

  exScanBlock.set_arg<cl_mem>(0, this->devMemCounts.p());
  int blocks = 120*4;
  exScanBlock.set_arg<int>(1, &blocks);
  exScanBlock.set_arg<cl_mem>(2, this->devMemCountsx.p());
  exScanBlock.set_arg<int>(3, NULL, 512); //shared memory allocation

  globalWork[0] = 512; globalWork[1] = 1;
  localWork [0] = 512; localWork [1] = 1;

  exScanBlock.setWork(globalWork, localWork);

  //////////////

  sortMove.set_arg<cl_mem>(0, srcKeys.p());
  sortMove.set_arg<cl_mem>(1, keysOutput.p());
  sortMove.set_arg<cl_mem>(2, srcValues.p());
  sortMove.set_arg<cl_mem>(3, valuesOutput.p());
  sortMove.set_arg<cl_mem>(4, this->devMemCounts.p());
  sortMove.set_arg<uint>(5, &N);
  sortMove.set_arg<uint>(6, NULL, 192); //Dynamic shared memory 128+64 , prefux sum buffer
  sortMove.set_arg<uint>(7, NULL, 64*4); //Dynamic shared memory stage buffer
  sortMove.set_arg<uint>(8, NULL, 64*4); //Dynamic shared memory stage_values buffer
  sortMove.set_arg<setupParams>(9, &sParam);
  sortMove.set_arg<int>(10, &bitIdx);

  globalWork[0] = 120*32;  globalWork[1] = 4;
  localWork [0] = 32;      localWork [1] = 4;

  sortMove.setWork(globalWork, localWork);

  bool pingPong = false;

  //Execute bitIdx 0

  sortCount.execute(execStream->s());
  exScanBlock.execute(execStream->s());
  sortMove.execute(execStream->s());  

  //Swap buffers
  sortCount.set_arg<cl_mem>(0, keysOutput.p());
  sortMove.set_arg<cl_mem>(0, keysOutput.p());
  sortMove.set_arg<cl_mem>(1, keysAPing.p());
  sortMove.set_arg<cl_mem>(2, valuesOutput.p());
  sortMove.set_arg<cl_mem>(3, valuesAPing.p());

  //Remaining bits, ping ponging buffers
  for(int i=1; i < numberOfBits; i++)
  {
    bitIdx = i;
    sortCount.set_arg<int>(5, &bitIdx);
    sortMove.set_arg<int>(10, &bitIdx);

    sortCount.execute(execStream->s());
    exScanBlock.execute(execStream->s()); 
    
    sortMove.execute(execStream->s());

    //Switch buffers
    if(pingPong)
    {
      sortCount.set_arg<cl_mem>(0, keysOutput.p());

      sortMove.set_arg<cl_mem>(0, keysOutput.p());
      sortMove.set_arg<cl_mem>(1, keysAPing.p());

      sortMove.set_arg<cl_mem>(2, valuesOutput.p());
      sortMove.set_arg<cl_mem>(3, valuesAPing.p());

      pingPong = false;
    }
    else
    {
      sortCount.set_arg<cl_mem>(0, keysAPing.p());

      sortMove.set_arg<cl_mem>(0, keysAPing.p());
      sortMove.set_arg<cl_mem>(1, keysOutput.p());

      sortMove.set_arg<cl_mem>(2, valuesAPing.p());
      sortMove.set_arg<cl_mem>(3, valuesOutput.p());

      pingPong = true;
    }
  }
 

}

