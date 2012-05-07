#include "octree.h"

#ifdef USE_DUST

void octree::allocateDustMemory(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  
  if( tree.dust_pos.get_size() > 0)
  {
    //Dust buffers, resize only
    //resize since we are going to add an extra
    //galaxy, wooo!
    int n_dust = tree.n_dust;  
    tree.dust_pos.cresize(n_dust+1, false);     
    tree.dust_key.cresize(n_dust+1, false);     
    tree.dust_vel.cresize(n_dust, false);
    tree.dust_acc0.cresize(n_dust, false);     
    tree.dust_acc1.cresize(n_dust, false);     
    tree.dust_ids.cresize(n_dust+1, false);     
    
    tree.dust2group_list.cresize(n_dust, false);
    tree.active_dust_list.cresize(n_dust+10, false);      //Extra space for atomics
    tree.dust_interactions.cresize(n_dust, false);     
    
    tree.dust_ngb.cresize(n_dust, false);     
  }
  else
  {
    //Dust buffers
    int n_dust = tree.n_dust;  
    tree.dust_pos.cmalloc(n_dust+1, false);     
    tree.dust_key.cmalloc(n_dust+1, false);     
    tree.dust_vel.cmalloc(n_dust, false);
    tree.dust_acc0.cmalloc(n_dust, false);     
    tree.dust_acc1.cmalloc(n_dust, false);     
    tree.dust_ids.cmalloc(n_dust+1, false);     
    
    tree.dust2group_list.cmalloc(n_dust, false);
    tree.active_dust_list.cmalloc(n_dust+10, false);      //Extra space for atomics
    tree.dust_interactions.cmalloc(n_dust, false);     
    
    tree.dust_ngb.cmalloc(n_dust, false);       
  }

  //Increase the position buffer, we will add the dust behind
  //this when rendering
  tree.bodies_pos.cresize(tree.n+1+tree.n_dust, false); 
  tree.bodies_ids.cresize(tree.n+1+tree.n_dust, false); 
  tree.bodies_vel.cresize(tree.n+1+tree.n_dust, false); 
  tree.dust_acc0.zeroMem();

}




void octree::allocateDustGroupBuffers(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  
  bool reduce = false;
  if(tree.dust_group_list.get_size() > 0)
  {
    tree.dust_group_list.cresize(tree.n_dust_groups, reduce);
    tree.activeDustGrouplist.cresize(tree.n_dust_groups, reduce);
    tree.dust_groupSizeInfo.cresize(tree.n_dust_groups, reduce);
    tree.dust_groupCenterInfo.cresize(tree.n_dust_groups, reduce);
  }
  else
  {
    tree.dust_group_list.cmalloc(tree.n_dust_groups, false);
    tree.activeDustGrouplist.cmalloc(tree.n_dust_groups, false);
    tree.dust_groupSizeInfo.cmalloc(tree.n_dust_groups, false);
    tree.dust_groupCenterInfo.cmalloc(tree.n_dust_groups, false);    
  }
}

void octree::sort_dust(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  
  devContext.startTiming();

  //Start reduction to get the boundary's of the dust
  boundaryReduction.set_arg<int>(0, &tree.n_dust);
  boundaryReduction.set_arg<cl_mem>(1, tree.dust_pos.p());
  boundaryReduction.set_arg<cl_mem>(2, devMemRMIN.p());
  boundaryReduction.set_arg<cl_mem>(3, devMemRMAX.p());

  boundaryReduction.setWork(tree.n, NTHREAD_BOUNDARY, NBLOCK_BOUNDARY);  //256 threads and 120 blocks in total
  boundaryReduction.execute(execStream->s());
  
   
  devMemRMIN.d2h();     //Need to be defined and initialized somewhere outside this function
  devMemRMAX.d2h();     //Need to be defined and initialized somewhere outside this function
  real4 r_min = make_real4(+1e10, +1e10, +1e10, +1e10); 
  real4 r_max = make_real4(-1e10, -1e10, -1e10, -1e10);   
  
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
  
  
  LOG("Found dust boundarys, number of particles %d : \n", tree.n_dust);
  LOG("min: %f\t%f\t%f\tmax: %f\t%f\t%f \n", r_min.x,r_min.y,r_min.z,r_max.x,r_max.y,r_max.z);

  //Compute the boundarys of the dust, needed to get the PH key
  real size     = 1.001f*std::max(r_max.z - r_min.z,
                         std::max(r_max.y - r_min.y, r_max.x - r_min.x));
  
  float4 corner   = make_real4(0.5f*(r_min.x + r_max.x) - 0.5f*size,
                             0.5f*(r_min.y + r_max.y) - 0.5f*size,
                             0.5f*(r_min.z + r_max.z) - 0.5f*size, size); 
       
  float domain_fac   = size/(1 << MAXLEVELS);
  
  corner.w = domain_fac;  
  

  //Compute the keys
  my_dev::dev_mem<uint4>  srcValues(devContext);
  
  //The generalBuffer1 has size uint*4*N*3
  //this buffer gets part: 0-uint*4*N
  srcValues.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                         tree.generalBuffer1.get_flags(), 
                         tree.generalBuffer1.get_devMem(),
                         &tree.generalBuffer1[0], 0,  
                         tree.n, getAllignmentOffset(0));  
  
  //Compute the keys directly into srcValues   
  build_key_list.set_arg<cl_mem>(0,   srcValues.p());
  build_key_list.set_arg<cl_mem>(1,   tree.dust_pos.p());
  build_key_list.set_arg<int>(2,      &tree.n_dust);
  build_key_list.set_arg<real4>(3,    &corner);
  build_key_list.setWork(tree.n_dust, 128); //128 threads per block
  build_key_list.execute(execStream->s());  
  
  // If srcValues and buffer are different, then the original values
  // are preserved, if they are the same srcValues will be overwritten  
  gpuSort(devContext, srcValues, tree.dust_key,srcValues, tree.n_dust, 32, 3, tree);


  
  //Sort the relevant properties
  //Note we can optimize this further as done with 
  //the normal particles

  my_dev::dev_mem<real4>  real4Buffer1(devContext);
  my_dev::dev_mem<real4>  real4Buffer2(devContext);
  my_dev::dev_mem<real4>  real4Buffer3(devContext);
  
  real4Buffer1.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[0], 0,  
                        tree.n_dust, getAllignmentOffset(0));  
    
  real4Buffer2.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[4*tree.n_dust], 4*tree.n_dust, 
                        tree.n_dust, getAllignmentOffset(4*tree.n_dust));   
  int prevOffset = getAllignmentOffset(4*tree.n_dust);
  
  real4Buffer3.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[8*tree.n_dust], 8*tree.n_dust, 
                        tree.n_dust, prevOffset + getAllignmentOffset(8*tree.n_dust+prevOffset));   
  
  
  dataReorderCombined.set_arg<int>(0,      &tree.n_dust);
  dataReorderCombined.set_arg<cl_mem>(1,   tree.dust_key.p());  
  dataReorderCombined.setWork(tree.n_dust, 512);   
  
  
  //Position, velocity and acc0
  dataReorderCombined.set_arg<cl_mem>(2,   tree.dust_pos.p());
  dataReorderCombined.set_arg<cl_mem>(3,   real4Buffer1.p()); 
  dataReorderCombined.set_arg<cl_mem>(4,   tree.dust_vel.p()); 
  dataReorderCombined.set_arg<cl_mem>(5,   real4Buffer2.p()); 
  dataReorderCombined.set_arg<cl_mem>(6,   tree.dust_acc0.p()); 
  dataReorderCombined.set_arg<cl_mem>(7,   real4Buffer3.p()); 
  dataReorderCombined.execute(execStream->s());
  tree.dust_pos.copy(real4Buffer1,  tree.n_dust);
  tree.dust_vel.copy(real4Buffer2,  tree.n_dust);
  tree.dust_acc0.copy(real4Buffer3, tree.n_dust);
  

  my_dev::dev_mem<int>  intBuffer(devContext);
  intBuffer.cmalloc_copy(tree.generalBuffer1.get_pinned(),   
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[4*tree.n_dust], 4*tree.n_dust,
                        tree.n_dust, getAllignmentOffset(4*tree.n_dust));  
  
  
  my_dev::dev_mem<float2>  float2Buffer(devContext);
  my_dev::dev_mem<int> sortPermutation(devContext);
  float2Buffer.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[0], 0,  
                        tree.n_dust, getAllignmentOffset(0)); 
  sortPermutation.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                        tree.generalBuffer1.get_flags(), 
                        tree.generalBuffer1.get_devMem(),
                        &tree.generalBuffer1[2*tree.n_dust], 2*tree.n_dust, 
                        tree.n_dust, getAllignmentOffset(2*tree.n_dust)); 
  
  dataReorderF2.set_arg<int>(0,      &tree.n_dust);
  dataReorderF2.set_arg<cl_mem>(1,   tree.dust_key.p());  
  
  dataReorderF2.set_arg<cl_mem>(2,   float2Buffer.p()); //Plce holder, dust has no time
  dataReorderF2.set_arg<cl_mem>(3,   float2Buffer.p()); //Reuse as destination1
  dataReorderF2.set_arg<cl_mem>(4,   tree.dust_ids.p()); 
  dataReorderF2.set_arg<cl_mem>(5,   sortPermutation.p()); //Reuse as destination2  
  dataReorderF2.setWork(tree.n_dust, 512);   
  dataReorderF2.execute(execStream->s());

  tree.dust_ids.copy(sortPermutation, sortPermutation.get_size());  
  
  
  
  devContext.stopTiming("DustSortReorder", 0);  
  
}
  
void octree::make_dust_groups(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  //Split the dust particles into groups
  //This is done slightly different than with the normal 
  //particles. We do an extra split, similar to how we create
  //a tree-level. This to prevent that particles far away
  //get into the same box because of jumps into the PH 
  
  my_dev::dev_mem<uint>  validList(devContext);
  my_dev::dev_mem<uint>  compactList(devContext);
    
  int n_bodies = tree.n_dust;
  validList.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                                    tree.generalBuffer1.get_flags(), 
                                    tree.generalBuffer1.get_devMem(),
                                    &tree.generalBuffer1[0], 0,
                                    n_bodies*2, getAllignmentOffset(0));
  validList.zeroMem(); 
                                    
  compactList.cmalloc_copy(tree.generalBuffer1.get_pinned(), 
                                    tree.generalBuffer1.get_flags(), 
                                    tree.generalBuffer1.get_devMem(),
                                    &tree.generalBuffer1[n_bodies*2], n_bodies*2,
                                    n_bodies*2, getAllignmentOffset(n_bodies*2));  
   
  // set devMemCountsx to 1 because it is used to early out when it hits zero
  this->devMemCountsx[0] = 1;
  this->devMemCountsx.h2d(1);  
    
  int level =  5; //This level proofs to be sort of OK. We can tune it depending on the data-set
  build_valid_list.set_arg<int>(0,     &n_bodies);
  build_valid_list.set_arg<int>(1,     &level);
  build_valid_list.set_arg<cl_mem>(2,  tree.dust_key.p());
  build_valid_list.set_arg<cl_mem>(3,  validList.p());  
  build_valid_list.set_arg<cl_mem>(4,  this->devMemCountsx.p());  
  build_valid_list.setWork(tree.n, 128);
  build_valid_list.execute(execStream->s());
  
  
  //Now we reuse the results of the build_valid_list
  //it already has the breaks in the right places 
  //Just add breaks every NGROUP items
  
  //The newest group creation method!
  define_dust_groups.set_arg<int>(0, &n_bodies);  
  define_dust_groups.set_arg<cl_mem>(1, tree.dust_pos.p());    
  define_dust_groups.set_arg<cl_mem>(2, validList.p());    
  define_dust_groups.setWork(n_bodies, 256);  
  define_dust_groups.execute(execStream->s());  
  
  
  // reset counts to 1 so next compact proceeds...
  this->devMemCountsx[0] = 1;
  this->devMemCountsx.h2d(1); 
  
  int validCount;
  gpuCompact(devContext, validList, compactList, n_bodies*2, &validCount);
  tree.n_dust_groups = validCount / 2;
  LOGF(stderr, "Ngroups_dust: %d \n", tree.n_dust_groups);
  
  this->allocateDustGroupBuffers(tree);
  
  store_dust_groups.set_arg<int>(0,     &tree.n_dust_groups);  
  store_dust_groups.set_arg<cl_mem>(1,  compactList.p());    
  store_dust_groups.set_arg<cl_mem>(2,  tree.dust2group_list.p());     
  store_dust_groups.set_arg<cl_mem>(3,  tree.dust_group_list.p()); 
  store_dust_groups.set_arg<cl_mem>(4,  tree.activeDustGrouplist.p());  
  store_dust_groups.setWork(-1, NCRIT,  tree.n_dust_groups);  
  store_dust_groups.execute(execStream->s());  
  
}


void octree::setDustGroupProperties(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
   //Set the group properties, note that it is not based on the nodes anymore
  //but on self created groups based on particle order setPHGroupData    
  copyNodeDataToGroupData.set_arg<int>(0,    &tree.n_dust_groups);
  copyNodeDataToGroupData.set_arg<int>(1,    &tree.n_dust);
  copyNodeDataToGroupData.set_arg<cl_mem>(2, tree.dust_pos.p());  
  copyNodeDataToGroupData.set_arg<cl_mem>(3, tree.dust_group_list.p());
  copyNodeDataToGroupData.set_arg<cl_mem>(4, tree.dust_groupCenterInfo.p());  
  copyNodeDataToGroupData.set_arg<cl_mem>(5, tree.dust_groupSizeInfo.p());
 
  copyNodeDataToGroupData.setWork(-1, NCRIT, tree.n_dust_groups);    
  copyNodeDataToGroupData.execute(execStream->s());


  /*
  tree.dust_groupCenterInfo.d2h();  
  tree.dust_groupSizeInfo.d2h();
  
  float maxGrpSize = 0;
  float grpSizeSum = 0;
  int maxID = 0;
  for(int i=0; i < tree.n_dust_groups; i++)
  {
	  //if(i < 20)
	  if(tree.dust_groupCenterInfo[i].w > 12)
	  fprintf(stderr,"%d %f %f %f \t\t %f %f %f %f\n",
			  i,
			  tree.dust_groupCenterInfo[i].x, 
			  tree.dust_groupCenterInfo[i].y, 
			  tree.dust_groupCenterInfo[i].z, 
			  tree.dust_groupSizeInfo[i].x, 
			  tree.dust_groupSizeInfo[i].y, 
			  tree.dust_groupSizeInfo[i].z,
			  tree.dust_groupCenterInfo[i].w); 
    if(maxGrpSize < tree.dust_groupCenterInfo[i].w)
	    maxID = i;
    maxGrpSize =   std::max(maxGrpSize, tree.dust_groupCenterInfo[i].w);
    grpSizeSum += tree.dust_groupCenterInfo[i].w;
  }


  tree.dust2group_list.d2h();
  tree.dust_pos.d2h();
  for(int i=0; i < tree.n_dust; i++)
  {
	  if(tree.dust2group_list[i] == maxID)
	  {
		  float4 prevPos = tree.dust_pos[i-1];
		  float4 nexPos = tree.dust_pos[i+1];
		  float4 curPos = tree.dust_pos[i];


  float dsPlus = ((curPos.x-nexPos.x)*(curPos.x-nexPos.x)) + 
                 ((curPos.y-nexPos.y)*(curPos.y-nexPos.y)) + 
                 ((curPos.z-nexPos.z)*(curPos.z-nexPos.z));

  float dsMin = ((curPos.x-prevPos.x)*(curPos.x-prevPos.x)) + 
                ((curPos.y-prevPos.y)*(curPos.y-prevPos.y)) + 
                ((curPos.z-prevPos.z)*(curPos.z-prevPos.z));



		  fprintf(stderr,"POS BIG %d %f %f %f DP: %f DM: %f\n",
				  i,
				  tree.dust_pos[i].x,
				  tree.dust_pos[i].y,
				  tree.dust_pos[i].z,
				  dsPlus, dsMin);
	  }
  }




  fprintf(stderr, "Group stats Max: %f \t Sum: %f Avg: %f  #groups: %d\n",
		  maxGrpSize, grpSizeSum, grpSizeSum/tree.n_dust_groups, tree.n_dust_groups);
*/
/*
  tree.dust_pos.d2h();
  tree.dust_vel.d2h();
  for(int i=0; i < 20; i++)
  {
	  fprintf(stderr,"POS: %f %f %f \t\t %f %f %f \n",
			  tree.dust_pos[i].x, 
			  tree.dust_pos[i].y, 
			  tree.dust_pos[i].z, 
			  tree.dust_vel[i].x, 
			  tree.dust_vel[i].y, 
			  tree.dust_vel[i].z); 

  }*/



  
}


void octree::predictDustStep(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  int idx = 0;
  predictDust.set_arg<int>(idx++,    &tree.n_dust);
  predictDust.set_arg<float>(idx++,  &t_current);
  predictDust.set_arg<float>(idx++,  &t_previous);
  predictDust.set_arg<cl_mem>(idx++, tree.dust_pos.p());
  predictDust.set_arg<cl_mem>(idx++, tree.dust_vel.p());
  predictDust.set_arg<cl_mem>(idx++, tree.dust_acc0.p());
  predictDust.set_arg<cl_mem>(idx++, tree.dust2group_list.p());
  predictDust.set_arg<cl_mem>(idx++, tree.activeDustGrouplist.p());
  predictDust.setWork(tree.n_dust, 128);
  predictDust.execute(execStream->s());
} //End predict


void octree::correctDustStep(tree_structure &tree)
{
  if(tree.n_dust == 0) return;
  //Correct the dust particles
  int idx = 0;
  correctDust.set_arg<int   >(idx++, &tree.n_dust);
  correctDust.set_arg<float >(idx++, &timeStep);  
  correctDust.set_arg<cl_mem>(idx++, tree.active_dust_list.p());
  correctDust.set_arg<cl_mem>(idx++, tree.dust_vel.p());
  correctDust.set_arg<cl_mem>(idx++, tree.dust_acc0.p());
  correctDust.set_arg<cl_mem>(idx++, tree.dust_acc1.p());  
  correctDust.setWork(tree.n_dust, 128);
  correctDust.execute(execStream->s());
}

void octree::approximate_dust(tree_structure &tree)
{ 
  if(tree.n_dust == 0) return;
  
  uint2 node_begend;
  int level_start = 2;
  node_begend.x   = tree.level_list[level_start].x;
  node_begend.y   = tree.level_list[level_start].y;

  //Reset the active particles
  tree.active_dust_list.zeroMem();

  //Set the kernel parameters, many!
  approxGrav.set_arg<int>(0,     &tree.n_dust_groups);
  approxGrav.set_arg<int>(1,     &tree.n_dust);
  approxGrav.set_arg<float>(2,   &(this->eps2));
  approxGrav.set_arg<uint2>(3,   &node_begend);
  approxGrav.set_arg<cl_mem>(4,  tree.activeDustGrouplist.p());
  approxGrav.set_arg<cl_mem>(5,  tree.bodies_Ppos.p());  //Bodies from the tree
  approxGrav.set_arg<cl_mem>(6,  tree.multipole.p());
  approxGrav.set_arg<cl_mem>(7,  tree.dust_acc1.p());
  approxGrav.set_arg<cl_mem>(8,  tree.dust_pos.p());  //Dust bodies
  approxGrav.set_arg<cl_mem>(9,  tree.dust_ngb.p());
  approxGrav.set_arg<cl_mem>(10, tree.active_dust_list.p());
  approxGrav.set_arg<cl_mem>(11, tree.dust_interactions.p());
  approxGrav.set_arg<cl_mem>(12, tree.boxSizeInfo.p());
  approxGrav.set_arg<cl_mem>(13, tree.dust_groupSizeInfo.p());
  approxGrav.set_arg<cl_mem>(14, tree.boxCenterInfo.p());
  approxGrav.set_arg<cl_mem>(15, tree.dust_groupCenterInfo.p());
  approxGrav.set_arg<cl_mem>(16, tree.dust_vel.p());
  approxGrav.set_arg<cl_mem>(17,  tree.generalBuffer1.p()); //Instead of using Local memory
  
  
  approxGrav.set_arg<real4>(18, tree.boxSizeInfo, 4, "texNodeSize");
  approxGrav.set_arg<real4>(19, tree.boxCenterInfo, 4, "texNodeCenter");
  approxGrav.set_arg<real4>(20, tree.multipole, 4, "texMultipole");
  approxGrav.set_arg<real4>(21, tree.dust_pos, 4, "texBody");
    
 
  approxGrav.setWork(-1, NTHREAD, nBlocksForTreeWalk);
  approxGrav.execute(gravStream->s());  //First half

  //Print interaction statistics
  #if 0
  
    tree.dust_interactions.d2h();
    long long directSum = 0;
    long long apprSum = 0;
    long long directSum2 = 0;
    long long apprSum2 = 0;
    
    
    int maxDir = -1;
    int maxAppr = -1;

    for(int i=0; i < tree.n_dust; i++)
    {
      apprSum     += tree.dust_interactions[i].x;
      directSum   += tree.dust_interactions[i].y;
      
      maxAppr = max(maxAppr,tree.dust_interactions[i].x);
      maxDir  = max(maxDir,tree.dust_interactions[i].y);
      
      apprSum2     += tree.dust_interactions[i].x*tree.dust_interactions[i].x;
      directSum2   += tree.dust_interactions[i].y*tree.dust_interactions[i].y;      
    }
  

    cout << "DUST Interaction at (rank= " << mpiGetRank() << " ) iter: " << iter << "\tdirect: " << directSum << "\tappr: " << apprSum << "\t";
    cout << "avg dir: " << directSum / tree.n_dust << "\tavg appr: " << apprSum / tree.n_dust << "\tMaxdir: " << maxDir << "\tmaxAppr: " << maxAppr <<  endl;
    cout << "sigma dir: " << sqrt((directSum2  - directSum)/ tree.n_dust) << "\tsigma appr: " << std::sqrt((apprSum2 - apprSum) / tree.n_dust)  <<  endl;    
  #endif

/*
    tree.dust_acc1.d2h();
    tree.dust_pos.d2h();

    for(int i=0; i < tree.n_dust; i++)
    {
	   
	fprintf(stderr,"%d\t%f %f %f %f \t %f %f %f\n", i, 
			tree.dust_acc1[i].x,
			tree.dust_acc1[i].y,
			tree.dust_acc1[i].z,
			tree.dust_acc1[i].w,
			tree.dust_pos[i].x,
			tree.dust_pos[i].y,
			tree.dust_pos[i].z);
    }
*/
//exit(0);

}
//end approximate

#endif //USE_DUST

