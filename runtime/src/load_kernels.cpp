#include "octree.h"
#include "devFunctionDefinitions.h"



void octree::set_context( bool disable_timing) {
  
  devContext.create(disable_timing);
  set_context2();
}


void octree::set_context(std::ostream &log, bool disable_timing) {
  
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

void octree::set_logPreamble(std::string text)
{
  devContext.setLogPreamble(text);
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



#ifdef USE_CUDA
  compactCount.load_source("./scanKernels.ptx", pathName.c_str());
  compactCount.create("compact_count", (const void*)&compact_count);
  

  exScanBlock.load_source("./scanKernels.ptx", pathName.c_str());
  exScanBlock.create("exclusive_scan_block", (const void*)&exclusive_scan_block);
  
  compactMove.load_source("./scanKernels.ptx", pathName.c_str());
  compactMove.create("compact_move", (const void*)&compact_move);
  
  splitMove.load_source("./scanKernels.ptx", pathName.c_str());
  splitMove.create("split_move", (const void*)split_move);
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
  segmentedCoarseGroupBoundary.setContext(devContext);
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
  segmentedCoarseGroupBoundary.load_source("./build_tree.ptx", pathName.c_str());
  
  /* create kernels */

  build_key_list.create("cl_build_key_list", (const void*)&cl_build_key_list);
  build_valid_list.create("cl_build_valid_list", (const void*)&cl_build_valid_list);
  build_nodes.create("cl_build_nodes", (const void*)&cl_build_nodes);
  link_tree.create("cl_link_tree", (const void*)&cl_link_tree);
  define_groups.create("build_group_list2", (const void*)&build_group_list2);
  build_level_list.create("build_level_list", (const void*)&gpu_build_level_list);
  boundaryReduction.create("boundaryReduction", (const void*)&gpu_boundaryReduction);
  boundaryReductionGroups.create("boundaryReductionGroups", (const void*)&gpu_boundaryReductionGroups);
  store_groups.create("store_group_list", (const void*)&store_group_list);
  segmentedCoarseGroupBoundary.create("segmentedCoarseGroupBoundary", (const void*)&gpu_segmentedCoarseGroupBoundary);
#endif

  // load tree-props kernels
  propsNonLeafD.setContext(devContext);
  propsLeafD.setContext(devContext);
  propsScalingD.setContext(devContext);
  
  setPHGroupData.setContext(devContext);
  setPHGroupDataGetKey.setContext(devContext);
  setPHGroupDataGetKey2.setContext(devContext);
 
  /* load kernels */
  
#ifdef USE_CUDA

  propsNonLeafD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "", -1);
  propsLeafD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "", -1);
  propsScalingD.load_source("./compute_propertiesD.ptx", pathName.c_str(), "",-1);
  
  setPHGroupData.load_source("./compute_propertiesD.ptx", pathName.c_str());
  setPHGroupDataGetKey.load_source("./compute_propertiesD.ptx", pathName.c_str());
  setPHGroupDataGetKey2.load_source("./compute_propertiesD.ptx", pathName.c_str());
  /* create kernels */
  
  propsNonLeafD.create("compute_non_leaf", (const void*)&compute_non_leaf);
  propsLeafD.create("compute_leaf", (const void*)&compute_leaf);
  propsScalingD.create("compute_scaling", (const void*)&compute_scaling);

  setPHGroupData.create("setPHGroupData", (const void*)&gpu_setPHGroupData);
  setPHGroupDataGetKey.create("setPHGroupDataGetKey", (const void*)&gpu_setPHGroupDataGetKey);
  setPHGroupDataGetKey2.create("setPHGroupDataGetKey2", (const void*)&gpu_setPHGroupDataGetKey2);
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
  determineLET.setContext(devContext);

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
  
  approxGravLET.load_source("./dev_approximate_gravity.ptx", pathName.c_str(), "", 64);  
  determineLET.load_source("./dev_approximate_gravity.ptx", pathName.c_str(), "", 64);
  /* create kernels */

  getTNext.create("get_Tnext", (const void*)&get_Tnext);
  predictParticles.create("predict_particles", (const void*)&predict_particles);
  getNActive.create("get_nactive", (const void*)&get_nactive);
  approxGrav.create("dev_approximate_gravity", (const void*)&dev_approximate_gravity);

#ifdef KEPLER /* preferL1 equal egaburov */
  cudaFuncSetCacheConfig((const void*)&dev_approximate_gravity, cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig((const void*)&dev_approximate_gravity_let, cudaFuncCachePreferL1);
#if 0
#if 1
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
#endif
#endif


  directGrav.create("dev_direct_gravity", (const void*)&dev_direct_gravity);
  correctParticles.create("correct_particles", (const void*)&correct_particles);
  computeDt.create("compute_dt", (const void*)&compute_dt);
  setActiveGrps.create("setActiveGroups", (const void*)&setActiveGroups);

  computeEnergy.create("compute_energy_double", (const void*)&compute_energy_double);
  distanceCheck.create("distanceCheck", (const void*)&distanceCheck);
  
  approxGravLET.create("dev_approximate_gravity_let", (const void*)&dev_approximate_gravity_let);
#if 0  /* egaburov, doesn't compile with this */
  determineLET.create("dev_determineLET", (const void*)&dev_determineLET);
#endif
#endif

  //Parallel kernels
  domainCheck.setContext(devContext);  
  extractSampleParticles.setContext(devContext);  
  internalMove.setContext(devContext);  
  build_parallel_grps.setContext(devContext);
  segmentedSummaryBasic.setContext(devContext);
  domainCheckSFC.setContext(devContext);
  internalMoveSFC.setContext(devContext);
  internalMoveSFC2.setContext(devContext);
  extractOutOfDomainParticlesAdvancedSFC2.setContext(devContext);
  insertNewParticlesSFC.setContext(devContext);
  extractSampleParticlesSFC.setContext(devContext);
  domainCheckSFCAndAssign.setContext(devContext);



#ifdef USE_CUDA
  domainCheck.load_source("./parallel.ptx", pathName.c_str());
  extractSampleParticles.load_source("./parallel.ptx", pathName.c_str());
  internalMove.load_source("./parallel.ptx", pathName.c_str());

  
  build_parallel_grps.load_source("./build_tree.ptx", pathName.c_str());
  segmentedSummaryBasic.load_source("./build_tree.ptx", pathName.c_str());
  domainCheckSFC.load_source("./parallel.ptx", pathName.c_str());
  internalMoveSFC.load_source("./parallel.ptx", pathName.c_str());
  internalMoveSFC2.load_source("./parallel.ptx", pathName.c_str());
  extractOutOfDomainParticlesAdvancedSFC2.load_source("./parallel.ptx", pathName.c_str());
  insertNewParticlesSFC.load_source("./parallel.ptx", pathName.c_str());
  extractSampleParticlesSFC.load_source("./parallel.ptx", pathName.c_str());
  domainCheckSFCAndAssign.load_source("./parallel.ptx", pathName.c_str());

  domainCheck.create("doDomainCheck", (const void*)&doDomainCheck);
  extractSampleParticles.create("extractSampleParticles", (const void*)&gpu_extractSampleParticles);
  internalMove.create("internalMove", (const void*)&gpu_internalMove);

  extractSampleParticlesSFC.create("build_parallel_grps", (const void*)&gpu_extractSampleParticlesSFC);
  build_parallel_grps.create("build_parallel_grps", (const void*)&gpu_build_parallel_grps);
  segmentedSummaryBasic.create("segmentedSummaryBasic", (const void*)&gpu_segmentedSummaryBasic);
  domainCheckSFC.create("domainCheckSFC", (const void*)&gpu_domainCheckSFC);
  internalMoveSFC.create("internalMoveSFC", (const void*)&gpu_internalMoveSFC);
  internalMoveSFC2.create("internalMoveSFC2", (const void*)&gpu_internalMoveSFC2);
  extractOutOfDomainParticlesAdvancedSFC2.create("extractOutOfDomainParticlesAdvancedSFC2", (const void*)&gpu_extractOutOfDomainParticlesAdvancedSFC2);
  insertNewParticlesSFC.create("insertNewParticlesSFC", (const void*)&gpu_insertNewParticlesSFC);
  domainCheckSFCAndAssign.create("domainCheckSFCAndAssign", (const void*)&gpu_domainCheckSFCAndAssign);
#endif
  
#ifdef USE_DUST
   define_dust_groups.setContext(devContext);
   define_dust_groups.load_source("./build_tree.ptx", pathName.c_str());
   define_dust_groups.create("define_dust_groups",(const void*)&gpu_define_dust_groups);
   
   store_dust_groups.setContext(devContext);
   store_dust_groups.load_source("./build_tree.ptx", pathName.c_str());
   store_dust_groups.create("store_dust_groups",(const void*)&gpu_store_dust_groups);
   
   predictDust.setContext(devContext);
   predictDust.load_source("./build_tree.ptx", pathName.c_str());
   predictDust.create("predict_dust_particles",(const void*)&predict_dust_particles);
   
   correctDust.setContext(devContext);
   correctDust.load_source("./build_tree.ptx", pathName.c_str());
   correctDust.create("correct_dust_particles",(const void*)&correct_dust_particles);

   copyNodeDataToGroupData.setContext(devContext);
   copyNodeDataToGroupData.load_source("./compute_propertiesD.ptx", pathName.c_str());
   copyNodeDataToGroupData.create("setPHGroupData", (const void*)&gpu_setPHGroupData);
   
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





