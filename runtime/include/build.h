#ifndef __BUILD_H__
#define __BUILD_H__

void build_tree_node_levels(octree &tree, 
                            my_dev::dev_mem<uint>  &validList,
                            my_dev::dev_mem<uint>  &compactList,
                            my_dev::dev_mem<uint>  &levelOffset,
                            my_dev::dev_mem<uint>  &maxLevel,
                            cudaStream_t           stream);

#endif