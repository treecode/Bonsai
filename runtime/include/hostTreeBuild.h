#ifndef WIN32
#include <sys/time.h>
#endif

//Next two lines are mainly for intellise sense of nsigh
#include <vector>
using namespace std;

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

#endif

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

    float4 *grpCenter = &cntrSizes[nodes.size()];
    float4 *grpSizes  = &cntrSizes[2*nodes.size()+nGroups];
    float4 *treeCnt   = &cntrSizes[0];
    float4 *treeSize  = &cntrSizes[nodes.size()+nGroups];

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
      std::vector<float4> &groupCentre,
      std::vector<float4> &groupSize,
      std::vector<float4> &treeProperties,
      std::vector<int>    &originalOrder,
      const float4         corner)
  {
    double t10 = get_time();
    const int nGroups = groupCentre.size();
    std::vector<v4sf>   tempBuffer(2*nGroups);   //Used for reorder
    std::vector<int >   tempBufferInt(nGroups);  //Used for reorder
    std::vector<uint4> keys(nGroups);
    //Compute the keys for the boundary boxes based on their geometric centers
    for(int i=0; i < nGroups; i++)
    {
      float4 center = groupCentre[i];
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


