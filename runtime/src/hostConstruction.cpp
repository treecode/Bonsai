#include "octree.h"

#define LEVEL_MIN_GRP_TREE 2

#if 1



static inline uint4 get_mask2(int level) {
  int mask_levels = 3*std::max(MAXLEVELS - level, 0);
  uint4 mask = {0x3FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,0xFFFFFFFF};

  if (mask_levels > 60)
  {
    mask.z = 0;
    mask.y = 0;
    mask.x = (mask.x >> (mask_levels - 60)) << (mask_levels - 60);
  }
  else if (mask_levels > 30) {
    mask.z = 0;
    mask.y = (mask.y >> (mask_levels - 30)) << (mask_levels - 30);
  } else {
    mask.z = (mask.z >> mask_levels) << mask_levels;
  }

  return mask;
}

static inline int cmp_uint42(uint4 a, uint4 b) {
  if      (a.x < b.x) return -1;
  else if (a.x > b.x) return +1;
  else {
    if       (a.y < b.y) return -1;
    else  if (a.y > b.y) return +1;
    else {
      if       (a.z < b.z) return -1;
      else  if (a.z > b.z) return +1;
      return 0;
    } //end z
  }  //end y
} //end x, function

//Binary search of the key within certain bounds (cij.x, cij.y)
static inline int find_key2(uint4 key, uint2 cij, uint4 *keys) {
  int l = cij.x;
  int r = cij.y - 1;
  while (r - l > 1) {
    int m = (r + l) >> 1;
    int cmp = cmp_uint42(keys[m], key);
    if (cmp == -1) {
      l = m;
    } else {
      r = m;
    }
  }
  if (cmp_uint42(keys[l], key) >= 0) return l;

  return r;
}

void inline mergeBoxesForGrpTree(float4 cntA, float4 sizeA, float4 cntB, float4 sizeB,
                          float4 &tempCnt, float4 &tempSize)
{

  float minxA, minxB, minyA, minyB, minzA, minzB;
  float maxxA, maxxB, maxyA, maxyB, maxzA, maxzB;

  minxA = cntA.x - sizeA.x;  minxB = cntB.x - sizeB.x;
  minyA = cntA.y - sizeA.y;  minyB = cntB.y - sizeB.y;
  minzA = cntA.z - sizeA.z;  minzB = cntB.z - sizeB.z;

  maxxA = cntA.x + sizeA.x;  maxxB = cntB.x + sizeB.x;
  maxyA = cntA.y + sizeA.y;  maxyB = cntB.y + sizeB.y;
  maxzA = cntA.z + sizeA.z;  maxzB = cntB.z + sizeB.z;

  float newMinx = fmin(minxA, minxB);
  float newMiny = fmin(minyA, minyB);
  float newMinz = fmin(minzA, minzB);

  float newMaxx = fmax(maxxA, maxxB);
  float newMaxy = fmax(maxyA, maxyB);
  float newMaxz = fmax(maxzA, maxzB);

  tempCnt.x = 0.5*(newMinx + newMaxx);
  tempCnt.y = 0.5*(newMiny + newMaxy);
  tempCnt.z = 0.5*(newMinz + newMaxz);

  tempSize.x = fmax(fabs(tempCnt.x-newMinx), fabs(tempCnt.x-newMaxx));
  tempSize.y = fmax(fabs(tempCnt.y-newMiny), fabs(tempCnt.y-newMaxy));
  tempSize.z = fmax(fabs(tempCnt.z-newMinz), fabs(tempCnt.z-newMaxz));
}


void octree::build_GroupTree(int n_bodies,
                     uint4 *keys,
                     uint2 *nodes,
                     uint4 *node_keys,
                     uint  *node_levels,
                     int &n_levels,
                     int &n_nodes,
                     int &startGrp,
                     int &endGrp) {

  const int level_min = LEVEL_MIN_GRP_TREE;

  double t0 = get_time();

  /***
  ****  --> generating tree nodes
  ***/

  int nMasked = 0;
  n_nodes = 0;
  for (n_levels = 0; n_levels < MAXLEVELS; n_levels++) {
    node_levels[n_levels] = n_nodes;

    if(nMasked == n_bodies)
    { //Jump out when all bodies are processed
      break;
    }

    uint4 mask = get_mask2(n_levels);
    mask.x     = mask.x | ((unsigned int)1 << 30) | ((unsigned int)1 << 31);

    uint  i_body = 0;
    uint4 i_key  = keys[i_body];
    i_key.x = i_key.x & mask.x;
    i_key.y = i_key.y & mask.y;
    i_key.z = i_key.z & mask.z;

    for (int i = 0; i < n_bodies; )
    {
      uint4 key = keys[i];
      key.x = key.x & mask.x;
      key.y = key.y & mask.y;
      key.z = key.z & mask.z;

      //Gives no speed-up
      //       if(key.x == 0xFFFFFFFF && ((i_key.x & 0xC0000000) != 0))
      //       {
      //         i = key.w;
      //         continue;
      //       }


      if (cmp_uint42(key, i_key) != 0 || i == n_bodies - 1)
      {
        if ((i_key.x & 0xC0000000) == 0) //Check that top 2 bits are not set meaning
        {                                //meaning its a non-used particle
          int i1 = i;
          if (i1 == n_bodies - 1) i1++;
          uint n_node = i1 - i_body; //Number of particles in this node

          node_keys[n_nodes] = i_key; //Key to identify the node

          uint2 node; // node.nb = n_node; node.b  = i_body;

          if (n_node <= NLEAF && n_levels > level_min)
          { //Leaf node
            for (int k = i_body; k < i1; k++)
              keys[k] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

            nMasked += n_node;

            node.x    = i_body | ((uint)(n_node-1) << LEAFBIT);
            node.y = 1;
          }
          else
          { //Normal node
            node.x = 0;
            node.y = 0;
          }
          nodes[n_nodes++] = node;
        }
        i_body = i;
        i_key  = key;


      } //if we found a new / different masked key
      i++;
    } //for bodies
  } //for levels
  node_levels[n_levels] = n_nodes;

  startGrp = node_levels[level_min];
  endGrp   = node_levels[level_min+1];


  double tlink = get_time();
//   for(int i=0; i < n_levels; i++)
//     fprintf(stderr, "On level: %d : %d --> %d  \n", i, node_levels[i],node_levels[i+1]);



  /***
  ****  --> linking the tree
  ***/
  //Do not start at the root since the root has no parent :)
  for (int level = 1; level < n_levels; level++) {
    uint4 mask = get_mask2(level - 1);
    int n0 = node_levels[level-1];
    int n1 = node_levels[level  ];
    int n2 = node_levels[level+1];

    int beg = n0;
    for (int i = n1; i < n2; i++) {
      uint4 key  = node_keys[i];
      key.x      = key.x & mask.x;
      key.y      = key.y & mask.y;
      key.z      = key.z & mask.z;
      //uint2 cij; cij.x = n0; cij.y = n1;
      uint2 cij; cij.x = beg; cij.y = n1;  //Continue from last point
      beg    = find_key2(key, cij, &node_keys[0]);
      uint child = nodes[beg].x;

      if (child == 0) {
        child = i;
      } else {
        //Increase number of children by 1
        uint nc = (child & 0xF0000000) >> 28;
        child   = (child & 0x0FFFFFFF) | ((nc + 1) << 28);
      }

      nodes[beg].x = child; //set child of the parent
      //nodes[i  ].p = beg;   //set parent of the current node
    }
  }

  fprintf(stderr, "Building nodes: %lg Linking: %lg Total; %lg\n", tlink-t0, get_time()-tlink, get_time()-t0);

  /***
  ****  --> collecting tree leaves
  ***/



  //Not required just for stats
  int n_leaves0 = 0;
  for (int i = 0; i < n_nodes; i++)
    if (nodes[i].y) n_leaves0++;



// #ifdef PRINTERR
#if 1
  fprintf(stderr, "  n_levels= %d  n_nodes= %d [%d] n_leaves= %d\n",
          n_levels, n_nodes, node_levels[n_levels], n_leaves0);
#endif

}


void octree::computeProps_GroupTree(real4 *grpCenter,
                                    real4 *grpSize,
                                    real4 *treeCnt,
                                    real4 *treeSize,
                                    uint2 *nodes,
                                    uint  *node_levels,
                                    int    n_levels)
{
  //Compute the properties
  double t0 = get_time();

  union{int i; float f;} itof; //__int_as_float


  for(int i=n_levels-1; i >=  0; i--)
  {

    for(int j= node_levels[i]; j < node_levels[i+1]; j++)
    {
      float4 newCent, newSize;

      if(nodes[j].y)
      {
        //Leaf reads from the group data
        int startGroup = (nodes[j].x   & 0x0FFFFFFF);
        int nGroup     = ((nodes[j].x & 0xF0000000) >> 28)+1;
        newCent = grpCenter[startGroup];
        newSize = grpSize  [startGroup];

        for(int k=startGroup; k < startGroup+nGroup; k++)
        {
          mergeBoxesForGrpTree(newCent, newSize, grpCenter[k], grpSize[k], newCent, newSize);
        }
        newCent.w   = -1; //Mark as leaf
        treeCnt[j]  = newCent;
        treeSize[j] = newSize;
      }
      else
      {
        //Node reads from the tree data
        int child    =    nodes[j].x & 0x0FFFFFFF;                         //Index to the first child of the node
        int nchild   = (((nodes[j].x & 0xF0000000) >> 28)) ;


        newCent = treeCnt [child];
        newSize = treeSize[child];

        for(int k= child; k < child+nchild+1; k++)
        {
          mergeBoxesForGrpTree(newCent, newSize, treeCnt[k], treeSize[k], newCent, newSize);
        }

        itof.i           = nodes[j].x;
        newSize.w  = itof.f;  //Child info
        newCent.w = 1; //mark as normal node
        treeCnt[j]  = newCent;
        treeSize[j] = newSize;
      }//if leaf
    }//for all nodes on this level
  } //for each level

  fprintf(stderr, "Grp-tree Properties took: %lg \n", get_time()-t0);

}






#endif

