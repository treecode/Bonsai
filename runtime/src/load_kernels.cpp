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
  
  //Assign the context to the local and remote tree structures (required for memory allocations)
  this->localTree.setContext(devContext);
  this->remoteTree.setContext(devContext);
}

void octree::set_logPreamble(std::string text)
{
  devContext.setLogPreamble(text);
}

void octree::load_kernels() {

  if (!devContext_flag) set_context();
  
  //If we arrive here we have acquired a device, configure parts of the code
  
  //Get the number of multiprocessors and compute number of 
  //blocks to be used during the tree-walk
  nMultiProcessors      = devContext.multiProcessorCount;
  const int blocksPerSM = getTreeWalkBlocksPerSM(
                          this->getDevContext()->getComputeCapabilityMajor(),
                          this->getDevContext()->getComputeCapabilityMinor());
  nBlocksForTreeWalk 	= nMultiProcessors*blocksPerSM;
  

  //AMUSE specific
  std::string pathName;
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
  
  //Connect the compute kernels to the actual CUDA functions


  //Scan, compact, split kernels
  compactCount.create("compact_count"		, (const void*)&compact_count);
  exScanBlock. create("exclusive_scan_block", (const void*)&exclusive_scan_block);
  compactMove. create("compact_move"		, (const void*)&compact_move);
  splitMove.   create("split_move"			, (const void*)split_move);


  //Tree-build kernels
  build_key_list.		  create("cl_build_key_list", 		(const void*)&cl_build_key_list);
  build_valid_list.		  create("cl_build_valid_list", 	(const void*)&cl_build_valid_list);
  build_nodes.			  create("cl_build_nodes", 			(const void*)&cl_build_nodes);
  link_tree.			  create("cl_link_tree", 			(const void*)&cl_link_tree);
  define_groups.		  create("build_group_list2", 		(const void*)&build_group_list2);
  build_level_list.		  create("build_level_list", 		(const void*)&gpu_build_level_list);
  boundaryReduction.      create("boundaryReduction", 		(const void*)&gpu_boundaryReduction);
  boundaryReductionGroups.create("boundaryReductionGroups", (const void*)&gpu_boundaryReductionGroups);
  store_groups.			  create("store_group_list", 		(const void*)&store_group_list);

  // load tree-props kernels
  propsNonLeafD. create("compute_non_leaf", (const void*)&compute_non_leaf);
  propsLeafD.	 create("compute_leaf",     (const void*)&compute_leaf);
  propsScalingD. create("compute_scaling",  (const void*)&compute_scaling);
  setPHGroupData.create("setPHGroupData",   (const void*)&gpu_setPHGroupData);

  //Time integration kernels
  getTNext.		   create("get_Tnext", 			     (const void*)&get_Tnext);
  predictParticles.create("predict_particles", 	     (const void*)&predict_particles);
  getNActive.      create("get_nactive", 		     (const void*)&get_nactive);
  correctParticles.create("correct_particles", 	     (const void*)&correct_particles);
  computeDt.	   create("compute_dt", 		     (const void*)&compute_dt);
  setActiveGrps.   create("setActiveGroups", 	     (const void*)&setActiveGroups);
  computeEnergy.   create("compute_energy_double",   (const void*)&compute_energy_double);
  approxGrav.	   create("dev_approximate_gravity", (const void*)&dev_approximate_gravity);

  //Parallel kernels
  approxGravLET.						  create("dev_approximate_gravity_let", 			(const void*)&dev_approximate_gravity_let);
  internalMoveSFC2.						  create("internalMoveSFC2", 						(const void*)&gpu_internalMoveSFC2);
  extractOutOfDomainParticlesAdvancedSFC2.create("extractOutOfDomainParticlesAdvancedSFC2", (const void*)&gpu_extractOutOfDomainParticlesAdvancedSFC2);
  insertNewParticlesSFC.				  create("insertNewParticlesSFC", 					(const void*)&gpu_insertNewParticlesSFC);
  domainCheckSFCAndAssign.				  create("domainCheckSFCAndAssign", 				(const void*)&gpu_domainCheckSFCAndAssign);

  //Other
  directGrav.create("dev_direct_gravity", (const void*)&dev_direct_gravity);


#ifdef KEPLER /* preferL1 equal egaburov */
  cudaFuncSetCacheConfig((const void*)&dev_approximate_gravity, 	cudaFuncCachePreferL1);
  cudaFuncSetCacheConfig((const void*)&dev_approximate_gravity_let, cudaFuncCachePreferL1);
#if 0
#if 1
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
#else
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
#endif
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





