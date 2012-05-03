#include "octree.h"


void octree::allocateDustMemory(tree_structure &tree)
{
  
  //Dust buffers
  tree.bodies_vel.cmalloc(n_dust, false);     
  tree.dust_pos.cmalloc(n_dust, false);     
  tree.dust_Ppos.cmalloc(n_dust, false);   
  tree.dust_key.cmalloc(n_dust, false);     
  tree.dust_vel.cmalloc(n_dust, false);
  tree.dust_Pvel.cmalloc(n_dust, false);     
  tree.dust_acc0.cmalloc(n_dust, false);     
  tree.dust_acc1.cmalloc(n_dust, false);     
  tree.dust_ids.cmalloc(n_dust, false);     
}