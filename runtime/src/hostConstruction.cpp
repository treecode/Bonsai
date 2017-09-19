#include "octree.h"

#ifndef WIN32
#include <sys/time.h>
#endif


#define USE_MPI

#ifdef USE_MPI


#ifdef __ALTIVEC__
    #include <altivec.h>
#else
    #include <xmmintrin.h>
#endif


typedef float  _v4sf  __attribute__((vector_size(16)));
typedef int    _v4si  __attribute__((vector_size(16)));

struct v4sf
{
  _v4sf data;
  v4sf() {}
  v4sf(const _v4sf _data) : data(_data) {}
  operator const _v4sf&() const {return data;}
  operator       _v4sf&()       {return data;}

};

//#endif

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

  float newMinx = std::min(minxA, minxB);
  float newMiny = std::min(minyA, minyB);
  float newMinz = std::min(minzA, minzB);

  float newMaxx = std::max(maxxA, maxxB);
  float newMaxy = std::max(maxyA, maxyB);
  float newMaxz = std::max(maxzA, maxzB);

  tempCnt.x = 0.5*(newMinx + newMaxx);
  tempCnt.y = 0.5*(newMiny + newMaxy);
  tempCnt.z = 0.5*(newMinz + newMaxz);

  tempSize.x = std::max(fabs(tempCnt.x-newMinx), fabs(tempCnt.x-newMaxx));
  tempSize.y = std::max(fabs(tempCnt.y-newMiny), fabs(tempCnt.y-newMaxy));
  tempSize.z = std::max(fabs(tempCnt.z-newMinz), fabs(tempCnt.z-newMaxz));

//  tempSize.x *= 1.10;
//  tempSize.y *= 1.10;
//  tempSize.z *= 1.10;
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

//  const int level_min = LEVEL_MIN_GRP_TREE;
//
  int level_min = -1;  

  double t0 = get_time();

  /***
  ****  --> generating tree nodes
  ***/
  bool minReached = false;
  int nMasked = 0;
  n_nodes = 0;
  for (n_levels = 0; n_levels < MAXLEVELS; n_levels++) {
    node_levels[n_levels] = n_nodes;

    if(n_nodes > 32 &&  !minReached)
    {
        //LOGF(stderr,"Min reached at: %d with %d \n", n_levels, n_nodes);
        minReached = true;
        level_min = n_levels-1;
    }

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

//          if (n_node <= NLEAF && n_levels > level_min)
          if (n_node <= 16 && minReached)
          { //Leaf node
            for (int k = i_body; k < i1; k++)
              keys[k] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, k); //We keep the w component for sorting the size and center arrays
              //keys[k] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

            nMasked += n_node;

//            node.x    = i_body | ((uint)(n_node-1) << LEAFBIT);
            node.x    = i_body | ((uint)(n_node-1) << 28);
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
  for(int i=0; i < n_levels; i++)
     LOGF(stderr, "On level: %d : %d --> %d  \n", i, node_levels[i],node_levels[i+1]);

//  mpiSync();
//exit(0);

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

//      if(procId == 1)	LOGF(stderr, "I iter: %d am ilevel: %d node: %d my parent is: %d \n", iter, level, i, beg);

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

  LOGF(stderr, "Building grp-tree took nodes: %lg Linking: %lg Total; %lg || n_levels= %d  n_nodes= %d [%d] start: %d end: %d\n",
                tlink-t0, get_time()-tlink, get_time()-t0,  n_levels, n_nodes, node_levels[n_levels], startGrp, endGrp);

  /***
  ****  --> collecting tree leaves
  ***/


// #ifdef PRINTERR
#if 0
  //Not required just for stats
  int n_leaves0 = 0;
  for (int i = 0; i < n_nodes; i++)
    if (nodes[i].y) n_leaves0++;

  LOGF(stderr, "  n_levels= %d  n_nodes= %d [%d] n_leaves= %d\n",
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

        for(int k= child; k < child+nchild+1; k++) //Note the +1
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

  LOGF(stderr, "Computing grp-tree Properties took: %lg \n", get_time()-t0);

}

void octree::build_NewTopLevels(int n_bodies,
                     uint4 *keys,
                     uint2 *nodes,
                     uint4 *node_keys,
                     uint  *node_levels,
                     int &n_levels,
                     int &n_nodes,
                     int &startNode,
                     int &endNode) {

  const int level_min = 1; //We just want a tree on top of our  trees, so no need for minimum


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

//          if (n_node <= NLEAF && n_levels > level_min)
          //NOTE: <= 8 since this won't be actual leaves but nodes
          if (n_node <= 8 && n_levels > level_min)
          { //Leaf node
            for (int k = i_body; k < i1; k++)
              keys[k] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);

            nMasked += n_node;

            node.x    = i_body | ((uint)(n_node-1) << 28);
            node.y    = 1; //1 indicate leaf
          }
          else
          { //Normal node
            node.x = 0;
            node.y = 0; //0 indicate node
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



  startNode = node_levels[level_min];
  endNode   = node_levels[level_min+1];


  double tlink = get_time();
  for(int i=0; i < n_levels; i++)
    LOGF(stderr, "On level: %d : %d --> %d  \n", i, node_levels[i],node_levels[i+1]);


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
    }
  }

  LOGF(stderr, "Building Top-nodes took nodes: %lg Linking: %lg Total; %lg || n_levels= %d  n_nodes= %d [%d]\n",
                tlink-t0, get_time()-tlink, get_time()-t0,  n_levels, n_nodes, node_levels[n_levels]);

  /***
  ****  --> collecting tree leaves
  ***/


// #ifdef PRINTERR
#if 0
  //Not required just for stats
  int n_leaves0 = 0;
  for (int i = 0; i < n_nodes; i++)
    if (nodes[i].y) n_leaves0++;

  LOGF(stderr, "  n_levels= %d  n_nodes= %d [%d] n_leaves= %d\n",
          n_levels, n_nodes, node_levels[n_levels], n_leaves0);
#endif
}

//Compute the properties of the newly build tree-nodes
void octree::computeProps_TopLevelTree(
                                      int topTree_n_nodes,
                                      int topTree_n_levels,
                                      uint* node_levels,
                                      uint2 *nodes,
                                      real4* topTreeCenters,
                                      real4* topTreeSizes,
                                      real4* topTreeMultipole,
                                      real4* nodeCenters,
                                      real4* nodeSizes,
                                      real4* multiPoles,
                                      double4* tempMultipoleRes)
  {
    //Now we have to compute the properties, do this from bottom up, as in the GPU case
    for(int i=topTree_n_levels;i > 0; i--)
    {
      int startNode = node_levels[i-1];
      int endNode   = node_levels[i];
//      LOGF(stderr, "Working on level: %d Start: %d  End: %d \n", i, startNode, endNode);

      for(int j=startNode; j < endNode; j++)
      {
        //Extract child information
        int child    =    nodes[j].x & 0x0FFFFFFF;//Index to the first child of the node
        int nchild   = (((nodes[j].x & 0xF0000000) >> 28)) + 1;

//        LOGF(stderr, "Level info node: %d  \tLeaf %d : Child: %d  nChild: %d\n",
//            j, nodes[j].y,  child, nchild);

        float4 *sourceCenter = NULL;
        float4 *sourceSize   = NULL;
        float4 *multipole    = NULL;

        if(nodes[j].y == 1)
        {
          //This is an end-node, read from original received data-array
          sourceCenter = &nodeCenters[0];
          sourceSize   = &nodeSizes  [0];
        }
        else
        {
          //This is a newly created node, read from new array
          sourceCenter = &topTreeCenters[0];
          sourceSize   = &topTreeSizes[0];
        }

        double3 r_min = {+1e10f, +1e10f, +1e10f};
        double3 r_max = {-1e10f, -1e10f, -1e10f};

        double3 r_minSPH = {+1e10f, +1e10f, +1e10f};
        double3 r_maxSPH = {-1e10f, -1e10f, -1e10f};

        double mass, posx, posy, posz;
        mass = posx = posy = posz = 0.0;

        double oct_q11, oct_q22, oct_q33;
        double oct_q12, oct_q13, oct_q23;

        oct_q11 = oct_q22 = oct_q33 = 0.0;
        oct_q12 = oct_q13 = oct_q23 = 0.0;

        float cellSmth = 0;

        for(int k=child; k < child+nchild; k++) //NOTE <= otherwise we miss the last child
        {
          double4 pos;
          double4 Q0, Q1;
          //Process/merge the children into this node

          //The center, compute the center+size back to a min/max
          double3 curRmin = {sourceCenter[k].x - sourceSize[k].x,
                             sourceCenter[k].y - sourceSize[k].y,
                             sourceCenter[k].z - sourceSize[k].z};
          double3 curRmax = {sourceCenter[k].x + sourceSize[k].x,
                             sourceCenter[k].y + sourceSize[k].y,
                             sourceCenter[k].z + sourceSize[k].z};

#if 0
          cellSmth = std::max(cellSmth , std::fabs(sourceCenter[k].w));
#else
          //Smoothing length is encoded in the higher bits of 32
          __half2 temp = *((__half2*)&sourceCenter[k].w);
          cellSmth = _cvtsh_ss(temp.y);
//          cellSmth = std::max(cellSmth , std::fabs(cellSmthx));
#endif



          //Compute the new min/max
          r_min.x = min(curRmin.x, r_min.x);
          r_min.y = min(curRmin.y, r_min.y);
          r_min.z = min(curRmin.z, r_min.z);
          r_max.x = max(curRmax.x, r_max.x);
          r_max.y = max(curRmax.y, r_max.y);
          r_max.z = max(curRmax.z, r_max.z);

          double smth = sqrt(abs(cellSmth));
          r_minSPH.x = min(curRmin.x-smth, r_min.x);
          r_minSPH.y = min(curRmin.y-smth, r_min.y);
          r_minSPH.z = min(curRmin.z-smth, r_min.z);
          r_maxSPH.x = max(curRmax.x+smth, r_max.x);
          r_maxSPH.y = max(curRmax.y+smth, r_max.y);
          r_maxSPH.z = max(curRmax.z+smth, r_max.z);

          //Compute monopole and quadrupole
          if(nodes[j].y == 1)
          {
            pos = make_double4(multiPoles[3*k+0].x,
                               multiPoles[3*k+0].y,
                               multiPoles[3*k+0].z,
                               multiPoles[3*k+0].w);
            Q0  = make_double4(multiPoles[3*k+1].x,
                               multiPoles[3*k+1].y,
                               multiPoles[3*k+1].z,
                               multiPoles[3*k+1].w);
            Q1  = make_double4(multiPoles[3*k+2].x,
                               multiPoles[3*k+2].y,
                               multiPoles[3*k+2].z,
                               multiPoles[3*k+2].w);
            double temp = Q1.y;
            Q1.y = Q1.z; Q1.z = temp;
            //Scale back to original order
            double im = 1.0 / pos.w;
            Q0.x = Q0.x + pos.x*pos.x; Q0.x = Q0.x / im;
            Q0.y = Q0.y + pos.y*pos.y; Q0.y = Q0.y / im;
            Q0.z = Q0.z + pos.z*pos.z; Q0.z = Q0.z / im;
            Q1.x = Q1.x + pos.x*pos.y; Q1.x = Q1.x / im;
            Q1.y = Q1.y + pos.y*pos.z; Q1.y = Q1.y / im;
            Q1.z = Q1.z + pos.x*pos.z; Q1.z = Q1.z / im;
          }
          else
          {
            pos = tempMultipoleRes[3*k+0];
            Q0  = tempMultipoleRes[3*k+1];
            Q1  = tempMultipoleRes[3*k+2];
          }

          mass += pos.w;
          posx += pos.w*pos.x;
          posy += pos.w*pos.y;
          posz += pos.w*pos.z;

          //Quadrupole
          oct_q11 += Q0.x;
          oct_q22 += Q0.y;
          oct_q33 += Q0.z;
          oct_q12 += Q1.x;
          oct_q13 += Q1.y;
          oct_q23 += Q1.z;
        }

        double4 mon = {posx, posy, posz, mass};
        double im = 1.0/mon.w;
        if(mon.w == 0) im = 0; //Allow tracer/mass-less particles

        mon.x *= im;
        mon.y *= im;
        mon.z *= im;

        tempMultipoleRes[j*3+0] = mon;
        tempMultipoleRes[j*3+1] = make_double4(oct_q11,oct_q22,oct_q33,0);
        tempMultipoleRes[j*3+2] = make_double4(oct_q12,oct_q13,oct_q23,0);
        //Store float4 results right away, so we do not have to do an extra loop
        //Scale the quadropole
        double4 Q0, Q1;
        Q0.x = oct_q11*im - mon.x*mon.x;
        Q0.y = oct_q22*im - mon.y*mon.y;
        Q0.z = oct_q33*im - mon.z*mon.z;
        Q1.x = oct_q12*im - mon.x*mon.y;
        Q1.y = oct_q13*im - mon.y*mon.z;
        Q1.z = oct_q23*im - mon.x*mon.z;

        //Switch the y and z parameter
        double temp = Q1.y;
        Q1.y = Q1.z; Q1.z = temp;


        topTreeMultipole[j*3+0] = make_float4(mon.x,mon.y,mon.z,mon.w);
        topTreeMultipole[j*3+1] = make_float4(Q0.x,Q0.y,Q0.z,0);
        topTreeMultipole[j*3+2] = make_float4(Q1.x,Q1.y,Q1.z,0);

        //All intermediate steps are done in full-double precision to prevent round-off
        //errors. Note that there is still a chance of round-off errors, because we start
        //with float data, while on the GPU we start/keep full precision data
        double4 boxCenterD;
        boxCenterD.x = 0.5*((double)r_min.x + (double)r_max.x);
        boxCenterD.y = 0.5*((double)r_min.y + (double)r_max.y);
        boxCenterD.z = 0.5*((double)r_min.z + (double)r_max.z);

        double4 boxSizeD = make_double4(std::max(abs(boxCenterD.x-r_min.x), abs(boxCenterD.x-r_max.x)),
                                        std::max(abs(boxCenterD.y-r_min.y), abs(boxCenterD.y-r_max.y)),
                                        std::max(abs(boxCenterD.z-r_min.z), abs(boxCenterD.z-r_max.z)), 0);

        //Compute distance between center box and center of mass
        double3 s3     = make_double3((boxCenterD.x - mon.x), (boxCenterD.y - mon.y), (boxCenterD.z -     mon.z));

        double s      = sqrt((s3.x*s3.x) + (s3.y*s3.y) + (s3.z*s3.z));
        //If mass-less particles form a node, the s would be huge in opening angle, make it 0
        if(fabs(mon.w) < 1e-10) s = 0;

        //Length of the box, note times 2 since we only computed half the distance before
        double l = 2*std::max(boxSizeD.x, std::max(boxSizeD.y, boxSizeD.z));

        //Extra check, shouldn't be necessary, probably it is otherwise the test for leaf can fail
        //This actually IS important Otherwise 0.0 < 0 can fail, now it will be: -1e-12 < 0
        if(l < 0.000001)
          l = 0.000001;

        #ifdef IMPBH
          double cellOp = (l/theta) + s;
        #else
          //Minimum distance method
          float cellOp = (l/theta);
        #endif


        float4 boxCenter   = make_float4(boxCenterD.x,boxCenterD.y, boxCenterD.z, boxCenterD.w);


        float maxSmth     = fmaxf(fmaxf(fmaxf(fabs(r_min.x-r_minSPH.x), fabs(r_max.x-r_maxSPH.x)),
                                  fmaxf(      fabs(r_min.y-r_minSPH.y), fabs(r_max.y-r_maxSPH.y))),
                                  fmaxf(      fabs(r_min.z-r_minSPH.z), fabs(r_max.z-r_maxSPH.z)));

        //Increase cellOp and maxSmth by 1% as there is no round-up mode on the host side.
        cellOp  *= 1.01;
        maxSmth *= 1.01;

        __half2 newCellOpSmth;
        newCellOpSmth.x = _cvtss_sh((float)(cellOp*cellOp),0); //GRAVITY
        newCellOpSmth.y = _cvtss_sh(maxSmth*maxSmth,0);        //SPH
        *((__half2*)&boxCenter.w) = newCellOpSmth;


        topTreeCenters[j]  = boxCenter;

        //Encode the child information, the leaf offsets are changed
        //such that they point to the correct starting offsets
        //in the final array, which starts after the 'topTree_n_nodes'
        //items.
        if(nodes[j].y == 1)
        { //Leaf
          child += topTree_n_nodes;
        }

        int childInfo = child | (nchild << 28);

        union{float f; int i;} u; //__float_as_int
        u.i           = childInfo;

        float4 boxSize   = make_float4(boxSizeD.x, boxSizeD.y, boxSizeD.z, 0);
        boxSize.w        = u.f; //int_as_float

        topTreeSizes[j] = boxSize;
      }//for startNode < endNode
    }//for each topTree level

#if 0
    //Compare the results
    for(int i=0; i < topTree_n_nodes; i++)
    {
      fprintf(stderr, "Node: %d \tSource size: %f %f %f %f Source center: %f %f %f %f \n",i,
          nodeSizes[i].x,nodeSizes[i].y,nodeSizes[i].z,nodeSizes[i].w,
          nodeCenters[i].x,nodeCenters[i].y,nodeCenters[i].z,
          nodeCenters[i].w);

      fprintf(stderr, "Node: %d \tNew    Size: %f %f %f %f  New    center: %f %f %f %f\n",i,
          topTreeSizes[i].x,  topTreeSizes[i].y,  topTreeSizes[i].z,   topTreeSizes[i].w,
          topTreeCenters[i].x,topTreeCenters[i].y,topTreeCenters[i].z, topTreeCenters[i].w);


      fprintf(stderr, "Ori-Node: %d \tMono: %f %f %f %f \tQ0: %f %f %f \tQ1: %f %f %f\n",i,
          multiPoles[3*i+0].x,multiPoles[3*i+0].y,multiPoles[3*i+0].z,multiPoles[3*i+0].w,
          multiPoles[3*i+1].x,multiPoles[3*i+1].y,multiPoles[3*i+1].z,
          multiPoles[3*i+2].x,multiPoles[3*i+2].y,multiPoles[3*i+2].z);

      fprintf(stderr, "New-Node: %d \tMono: %f %f %f %f \tQ0: %f %f %f \tQ1: %f %f %f\n\n\n",i,
          topTreeMultipole[3*i+0].x,topTreeMultipole[3*i+0].y,topTreeMultipole[3*i+0].z,topTreeMultipole[3*i+0].w,
          topTreeMultipole[3*i+1].x,topTreeMultipole[3*i+1].y,topTreeMultipole[3*i+1].z,
          topTreeMultipole[3*i+2].x,topTreeMultipole[3*i+2].y,topTreeMultipole[3*i+2].z);
    }
#endif

  }//end function/section


struct HostConstruction
{

#define MINNODES 8             //Minimum number of nodes required, before we create leaves
#define NLEAF_GROUP_TREE 16


private:


  double get_time() {
  #ifdef WIN32
    if (sysTimerFreq.QuadPart == 0)
    {
      return -1.0;
    }
    else
    {
      LARGE_INTEGER c;
      QueryPerformanceCounter(&c);
      return static_cast<double>( (double)(c.QuadPart - sysTimerAtStart.QuadPart) / sysTimerFreq.QuadPart );
    }
  #else
    struct timeval Tvalue;
    struct timezone dummy;

    gettimeofday(&Tvalue,&dummy);
    return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
  #endif
  }

  static uint4 host_get_key(uint4 crd)
  {
    const int bits = 30;  //20 to make it same number as morton order
    int i,xi, yi, zi;
    int mask;
    int key;

    mask = crd.y;
    crd.y = crd.z;
    crd.z = mask;

    //0= 000, 1=001, 2=011, 3=010, 4=110, 5=111, 6=101, 7=100
    //000=0=0, 001=1=1, 011=3=2, 010=2=3, 110=6=4, 111=7=5, 101=5=6, 100=4=7
    const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};

    int temp;

    mask = 1 << (bits - 1);
    key  = 0;

    uint4 key_new;

    for(i = 0; i < bits; i++, mask >>= 1)
    {
      xi = (crd.x & mask) ? 1 : 0;
      yi = (crd.y & mask) ? 1 : 0;
      zi = (crd.z & mask) ? 1 : 0;

      int index = (xi << 2) + (yi << 1) + zi;

      if(index == 0)
      {
        temp = crd.z; crd.z = crd.y; crd.y = temp;
      }
      else  if(index == 1 || index == 5)
      {
        temp = crd.x; crd.x = crd.y; crd.y = temp;
      }
      else  if(index == 4 || index == 6)
      {
        crd.x = (crd.x) ^ (-1);
        crd.z = (crd.z) ^ (-1);
      }
      else  if(index == 7 || index == 3)
      {
        temp = (crd.x) ^ (-1);
        crd.x = (crd.y) ^ (-1);
        crd.y = temp;
      }
      else
      {
        temp = (crd.z) ^ (-1);
        crd.z = (crd.y) ^ (-1);
        crd.y = temp;
      }

      key = (key << 3) + C[index];

      if(i == 19)
      {
        key_new.y = key;
        key = 0;
      }
      if(i == 9)
      {
        key_new.x = key;
        key = 0;
      }
    } //end for

    key_new.z = key;

    return key_new;
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

  //  tempSize.x *= 1.10;
  //  tempSize.y *= 1.10;
  //  tempSize.z *= 1.10;
  }




  void constructStructure(
                     int n_bodies,
                     vector<uint4> &keys,
                     vector<uint2> &nodes,
                     vector<uint4> &node_keys,
                     vector<uint>  &node_levels,
                     int &startGrp,
                     int &endGrp)
  {
    int level_min = -1;

    nodes.reserve(n_bodies);
    node_keys.reserve(n_bodies);
    node_levels.reserve(MAXLEVELS);

    //Generate the nodes
    bool minReached = false;
    int nMasked = 0;
    int n_nodes = 0;
    int n_levels = 0;
    for (n_levels = 0; n_levels < MAXLEVELS; n_levels++)
    {
      node_levels.push_back(n_nodes);

      if(n_nodes > MINNODES &&  !minReached)
      {
        minReached = true;
        level_min = n_levels-1;
      }

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

        if (cmp_uint42(key, i_key) != 0 || i == n_bodies - 1)
        {
          if ((i_key.x & 0xC0000000) == 0) //Check that top 2 bits are not set
          {                                //meaning its a non-used particle
            int i1 = i;
            if (i1 == n_bodies - 1) i1++;

            uint n_node = i1 - i_body; //Number of particles in this node
            node_keys.push_back(i_key); //Key to identify the node
            uint2 node;

            if (n_node <= NLEAF_GROUP_TREE && minReached)
            { //Leaf node
              for (int k = i_body; k < i1; k++)
                keys[k] = make_uint4(0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, k); //We keep the w component for sorting the size and center arrays

              nMasked += n_node;
              node.x    = i_body | ((uint)(n_node-1) << LEAFBIT);
//              node.x    = i_body | ((uint)(n_node-1) << 28);
              node.y = 1;
            }
            else
            { //Normal node
              node.x = 0;
              node.y = 0;
            }
            nodes.push_back(node);
            n_nodes++;
          }
          i_body = i;
          i_key  = key;
        } //if we found a new / different masked key
        i++;
      } //for bodies
    } //for levels
    node_levels.push_back(n_nodes);

    startGrp = node_levels[level_min];
    endGrp   = node_levels[level_min+1];


    double tlink = get_time();
    for(int i=0; i < n_levels; i++)
      LOGF(stderr, "On level: %d : %d --> %d  \n", i, node_levels[i],node_levels[i+1]);

    //Link the tree
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

        uint2 cij; cij.x = beg; cij.y = n1;  //Continue from last point
        beg        = find_key2(key, cij, &node_keys[0]);
        uint child = nodes[beg].x;

        if (child == 0) {
          child = i;
        } else {
          //Increase number of children by 1
          uint nc = (child & 0xF0000000) >> LEAFBIT;
          child   = (child & 0x0FFFFFFF) | ((nc + 1) << LEAFBIT);
        }

        nodes[beg].x = child; //set child of the parent
        //nodes[i  ].p = beg;   //set parent of the current node
      }
    }
    LOGF(stderr, "Building grp-tree took || n_levels= %d  n_nodes= %d [%d] start: %d end: %d\n",
                      n_levels, n_nodes, node_levels[n_levels], startGrp, endGrp);
  }



  void computeProperties(vector<float4> &cntrSizes,
                         vector<uint2>  &nodes,
                         vector<uint>   &node_levels,
                         int nGroups
                        )
  {
    //Compute the properties
    double t0 = get_time();

    real4 *grpCenter = &cntrSizes[nodes.size()];
    real4 *grpSizes  = &cntrSizes[2*nodes.size()+nGroups];
    real4 *treeCnt   = &cntrSizes[0];
    real4 *treeSize  = &cntrSizes[nodes.size()+nGroups];

    union{int i; float f;} itof; //__int_as_float

    int    n_levels = node_levels.size()-2; //-1 to start at correct lvl, -1 to ignore end
    for(int i=n_levels-1; i >=  0; i--)
    {
      LOGF(stderr,"On level: %d \n", i);
      for(int j= node_levels[i]; j < node_levels[i+1]; j++)
      {
        float4 newCent, newSize;

        if(nodes[j].y)
        {
          //Leaf reads from the group data
          int startGroup = (nodes[j].x  & 0x0FFFFFFF);
          int nGroup     = ((nodes[j].x & 0xF0000000) >> LEAFBIT)+1;
          newCent = grpCenter[startGroup];
          newSize = grpSizes [startGroup];

          for(int k=startGroup; k < startGroup+nGroup; k++)
          {
            mergeBoxesForGrpTree(newCent, newSize, grpCenter[k], grpSizes[k], newCent, newSize);
          }
          newCent.w   = -1; //Mark as a leaf

          //Modify the reading offset for the children/boundaries
          startGroup += nodes.size();
          itof.i      = startGroup | ((uint)(nGroup-1) << LEAFBIT);
          newSize.w   = itof.f;  //Child info

          treeCnt[j]  = newCent;
          treeSize[j] = newSize;
        }
        else
        {
          //Node reads from the tree data
          int child    =    nodes[j].x & 0x0FFFFFFF;                         //Index to the first child of the node
          int nchild   = (((nodes[j].x & 0xF0000000) >> LEAFBIT)) ;


          newCent = treeCnt [child];
          newSize = treeSize[child];

          for(int k= child; k < child+nchild+1; k++) //Note the +1
          {
            mergeBoxesForGrpTree(newCent, newSize, treeCnt[k], treeSize[k], newCent, newSize);
          }

          itof.i      = nodes[j].x;
          newSize.w   = itof.f;  //Child info
          newCent.w   = 1;       //mark as normal node
          treeCnt[j]  = newCent;
          treeSize[j] = newSize;
        }//if leaf
      }//for all nodes on this level
    } //for each level

    LOGF(stderr, "Computing grp-tree Properties took: %lg \n", get_time()-t0);
  }


public:
  HostConstruction(
      std::vector<real4> &groupCentre,
      std::vector<real4> &groupSize,
      std::vector<real4> &treeProperties,
      std::vector<int>   &originalOrder,
      const float4        corner)
  {
    double t10 = get_time();
    const int nGroups = groupCentre.size();
    std::vector<v4sf>   tempBuffer(2*nGroups);   //Used for reorder
    std::vector<int >   tempBufferInt(nGroups);  //Used for reorder
    std::vector<uint4> keys(nGroups);
    //Compute the keys for the boundary boxes based on their geometric centers
    for(int i=0; i < nGroups; i++)
    {
      real4 center = groupCentre[i];
      uint4 crd;
      crd.x = (int)((center.x - corner.x) / corner.w);
      crd.y = (int)((center.y - corner.y) / corner.w);
      crd.z = (int)((center.z - corner.z) / corner.w);

      keys[i]   = host_get_key(crd);
      keys[i].w = i;    //Store the original index to be used after sorting
    }//for i,

    //Sort the cells by their keys
    std::sort(keys.begin(), keys.end(), cmp_ph_key());

    //Reorder the groupCentre and groupSize arrays after the ordering of the keys
    for(int i=0; i < nGroups; i++)
    {
      tempBuffer[i]             = ((v4sf*)&groupCentre[0])[keys[i].w];
      tempBuffer[i+nGroups]     = ((v4sf*)&groupSize  [0])[keys[i].w];
      tempBufferInt[i]          = originalOrder[keys[i].w];
    }
    for(int i=0; i < nGroups; i++)
    {
      ((v4sf*)&groupCentre[0])[i] = tempBuffer[i];
      ((v4sf*)&groupSize[0])  [i] = tempBuffer[i+nGroups];
      originalOrder[i]            = tempBufferInt[i];
    }
    double t20 = get_time();

    vector<uint2> nodes;
    vector<uint4> node_keys;
    vector<uint>  node_levels;
    int startGrp, endGrp;

    constructStructure(nGroups,
                       keys,
                       nodes,
                       node_keys,
                       node_levels,
                       startGrp,
                       endGrp);
    double t30 = get_time();



    //Add the properties of the groups (size/center) after that of the tree. That way we
    //get a linear array containing all the available data in one consistent structure
    //Requires that we change offsets inside the 'computeProperties' function
    int nTreeNodes = nodes.size();

    treeProperties.resize(2*(nGroups+nTreeNodes)); //First centers then sizes
    for(int i=0; i < nGroups; i++)
    {
      treeProperties[nTreeNodes+i]           = groupCentre[i];
      treeProperties[nTreeNodes+i].w         = 0; //Mark as 0 to identify it as a group/box/particle
      treeProperties[2*nTreeNodes+nGroups+i] = groupSize  [i];
    }

    computeProperties(treeProperties,
                      nodes,
                      node_levels,
                      nGroups);
    double t40 = get_time();
    LOGF(stderr,"Building times: Sort: %lg Construct: %lg Props: %lg \n",t20-t10, t30-t20, t40-t30);

  }

};

#endif

#endif

