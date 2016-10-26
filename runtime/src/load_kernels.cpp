#include "octree.h"
#include "devFunctionDefinitions.h"


void octree::load_kernels() {
  
  //If we arrive here we have acquired a device, configure parts of the code
  
  //Get the number of multiprocessors and compute number of 
  //blocks to be used during the tree-walk
  nMultiProcessors      = devContext->multiProcessorCount;
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
  SPHDensity.      create("dev_sph_density", (const void*)&dev_sph_density);

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
void octree::gpuCompact(my_dev::dev_mem<uint> &srcValues,
                        my_dev::dev_mem<uint> &output,                        
                        int N, 
                        int *validCount) // if validCount NULL leave count on device
{

// thrust_gpuCompact(devContext, 
//                   srcValues,
//                   output,                        
//                   N, validCount);
  
  //Memory that should be allocated outside the function:
  //devMemCounts and devMemCountsx 

  // make sure previous reset has finished.
  this->devMemCountsx.waitForCopyEvent();

  //Kernel configuration parameters
  setupParams sParam;
  sParam.jobs                = (N / 64) / 480  ; //64=32*2 2 items per look, 480 is 120*4, number of procs
  sParam.blocksWithExtraJobs = (N / 64) % 480; 
  sParam.extraElements       = N % 64;
  sParam.extraOffset         = N - sParam.extraElements;

  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 32*120;   globalWork[1] = 4;
  localWork [0] = 32;       localWork[1]  = 4;

  compactCount.set_args(sizeof(int)*128, srcValues.p(), this->devMemCounts.p(), &N, &sParam, this->devMemCountsx.p());
  compactCount.setWork(globalWork, localWork);
  compactMove.set_args(sizeof(uint)*192, srcValues.p(), output.p(),this->devMemCounts.p(), &N, &sParam, this->devMemCountsx.p());
  compactMove.setWork(globalWork, localWork);


  globalWork[0] = 512; globalWork[1] = 1;
  localWork [0] = 512; localWork [1] = 1;
  int compactBlocks = 120*4;    //Number of blocks used for the compactCount and move calls
  exScanBlock.set_args(sizeof(int)*512, this->devMemCounts.p(), &compactBlocks, this->devMemCountsx.p());
  exScanBlock.setWork(globalWork, localWork);

  ////////////////////

  compactCount.execute2(execStream->s());
  exScanBlock .execute2(execStream->s());
  compactMove .execute2(execStream->s());
  
  if (validCount)
  {
    this->devMemCountsx.d2h();
    *validCount = this->devMemCountsx[0];
    //printf("Total number of valid items: %d \n", countx[0]);
  }
}

//Splits an array of integers, the values in srcValid indicate if a
//value is valid (1 == valid anything else is UNvalid) returns the 
//split values in the output array (first all valid
//number and then the invalid ones) and the total
//number of valid items is stored in 'count' 
void octree::gpuSplit(my_dev::dev_mem<uint> &srcValues,
                      my_dev::dev_mem<uint> &output,                        
                      int N, 
                      int *validCount)  // if validCount NULL leave count on device
{
  //In the next step we associate the GPU memory with the Kernel arguments
  //Memory that should be allocated outside the function:
  //devMemCounts and devMemCountsx 
  
  // make sure previous reset has finished.
  this->devMemCountsx.waitForCopyEvent();

  //Kernel configuration parameters
  setupParams sParam;
  sParam.jobs = (N / 64) / 480  ; //64=32*2 2 items per look, 480 is 120*4, number of procs
  sParam.blocksWithExtraJobs = (N / 64) % 480; 
  sParam.extraElements = N % 64;
  sParam.extraOffset = N - sParam.extraElements;
  

  vector<size_t> localWork(2), globalWork(2);
  globalWork[0] = 32*120;   globalWork[1] = 4;
  localWork [0] = 32;       localWork[1] = 4;

  compactCount.set_args(sizeof(int)*128, srcValues.p(), this->devMemCounts.p(), &N, &sParam, this->devMemCountsx.p());
  compactCount.setWork(globalWork, localWork);

  splitMove.set_args(sizeof(uint)*192, srcValues.p(), output.p(),this->devMemCounts.p(), &N, &sParam);
  splitMove.setWork(globalWork, localWork);

  globalWork[0] = 512; globalWork[1] = 1;
  localWork [0] = 512; localWork [1] = 1;
  int compactBlocks = 120*4;    //Number of blocks used for the compactCount and move calls
  exScanBlock.set_args(sizeof(int)*512, this->devMemCounts.p(), &compactBlocks, this->devMemCountsx.p());
  exScanBlock.setWork(globalWork, localWork);

  ////////////////////
  compactCount.execute2(execStream->s());
  exScanBlock. execute2(execStream->s());
  splitMove.   execute2(execStream->s());

  if (validCount) {
    this->devMemCountsx.d2h();
    *validCount = this->devMemCountsx[0];
  }
}


