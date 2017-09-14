#include "bonsai.h"
#include "support_kernels.cu"
#include <stdio.h>

#if 0
#include "cuda_fp16.h"
#else

#if defined(__CUDACC_RTC__)
#define __CUDA_FP16_DECL__ __host__ __device__
#else /* !__CUDACC_RTC__ */
#define __CUDA_FP16_DECL__ static __device__ __inline__
#endif /* __CUDACC_RTC__ */


typedef struct __align__(4) {
   unsigned int x;
} __half2;
typedef struct __align__(2) {
   unsigned short x;
} __half;

typedef __half2 half2;
__CUDA_FP16_DECL__ __half2 __halves2half2(const __half l, const __half h)
{
   __half2 val;
   asm("{  mov.b32 %0, {%1,%2};}\n"
       : "=r"(val.x) : "h"(l.x), "h"(h.x));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half(const float f)
{
   __half val;
   asm volatile("{  cvt.rn.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half_ru(const float f)
{
   __half val;
   asm volatile("{  cvt.rp.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}
__CUDA_FP16_DECL__ __half __float2half_rd(const float f)
{
   __half val;
   asm volatile("{  cvt.rm.f16.f32 %0, %1;}\n" : "=h"(val.x) : "f"(f));
   return val;
}

__CUDA_FP16_DECL__ float __half2float(const __half h)
{
   float val;
   asm volatile("{  cvt.f32.f16 %0, %1;}\n" : "=f"(val) : "h"(h.x));
   return val;
}


__CUDA_FP16_DECL__ float2 __half22float2(const __half2 l)
{
   float hi_float;
   float lo_float;
   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, low;}\n" : "=f"(lo_float) : "r"(l.x));

   asm("{.reg .f16 low,high;\n"
       "  mov.b32 {low,high},%1;\n"
       "  cvt.f32.f16 %0, high;}\n" : "=f"(hi_float) : "r"(l.x));

   return make_float2(lo_float, hi_float);
}
#endif



#include "../profiling/bonsai_timing.h"
PROF_MODULE(compute_propertiesD);

#include "node_specs.h"

static __device__ __forceinline__ void sh_MinMax2(int i, int j, float3 *r_min, float3 *r_max, volatile float3 *sh_rmin, volatile  float3 *sh_rmax)
{
  sh_rmin[i].x  = (*r_min).x = fminf((*r_min).x, sh_rmin[j].x);
  sh_rmin[i].y  = (*r_min).y = fminf((*r_min).y, sh_rmin[j].y);
  sh_rmin[i].z  = (*r_min).z = fminf((*r_min).z, sh_rmin[j].z);
  sh_rmax[i].x  = (*r_max).x = fmaxf((*r_max).x, sh_rmax[j].x);
  sh_rmax[i].y  = (*r_max).y = fmaxf((*r_max).y, sh_rmax[j].y);
  sh_rmax[i].z  = (*r_max).z = fmaxf((*r_max).z, sh_rmax[j].z);
}

//////////////////////////////
//////////////////////////////
//////////////////////////////

//Helper functions for leaf-nodes
static __device__ void compute_monopole(double &mass, double &posx,
                                 double &posy, double &posz,
                                 float4 pos)
{
  mass += pos.w;
  posx += pos.w*pos.x;
  posy += pos.w*pos.y;
  posz += pos.w*pos.z;
}

static __device__ void compute_quadropole(double &oct_q11, double &oct_q22, double &oct_q33,
                                   double &oct_q12, double &oct_q13, double &oct_q23,
                                   float4 pos)
{
  oct_q11 += pos.w * pos.x*pos.x;
  oct_q22 += pos.w * pos.y*pos.y;
  oct_q33 += pos.w * pos.z*pos.z;
  oct_q12 += pos.w * pos.x*pos.y;
  oct_q13 += pos.w * pos.y*pos.z;
  oct_q23 += pos.w * pos.z*pos.x;
}

static __device__ void compute_bounds(float3 &r_min, float3 &r_max,
                               float4 pos)
{
  r_min.x = fminf(r_min.x, pos.x);
  r_min.y = fminf(r_min.y, pos.y);
  r_min.z = fminf(r_min.z, pos.z);

  r_max.x = fmaxf(r_max.x, pos.x);
  r_max.y = fmaxf(r_max.y, pos.y);
  r_max.z = fmaxf(r_max.z, pos.z);
}

//Non-leaf node helper functions
static __device__ void compute_monopole_node(double &mass, double &posx,
                                 double &posy, double &posz,
                                 double4  pos)
{
  mass += pos.w;
  posx += pos.w*pos.x;
  posy += pos.w*pos.y;
  posz += pos.w*pos.z;
}


static __device__ void compute_quadropole_node(double &oct_q11, double &oct_q22, double &oct_q33,
                                        double &oct_q12, double &oct_q13, double &oct_q23,
                                        double4 Q0, double4 Q1)
{
  oct_q11 += Q0.x;
  oct_q22 += Q0.y;
  oct_q33 += Q0.z;
  oct_q12 += Q1.x;
  oct_q13 += Q1.y;
  oct_q23 += Q1.z;
}

static __device__ void compute_bounds_node(float3 &r_min, float3 &r_max,
                                    float4 node_min, float4 node_max)
{
  r_min.x = fminf(r_min.x, node_min.x);
  r_min.y = fminf(r_min.y, node_min.y);
  r_min.z = fminf(r_min.z, node_min.z);

  r_max.x = fmaxf(r_max.x, node_max.x);
  r_max.y = fmaxf(r_max.y, node_max.y);
  r_max.z = fmaxf(r_max.z, node_max.z);
}

KERNEL_DECLARE(compute_leaf_sph)(
                                 const int       n_leafs,
                                 uint           *leafsIdxs,
                                 uint2          *node_bodies,
                                 real4          *body_pos,
                                 double4        *multipole,
                                 real4          *nodeLowerBounds,
                                 real4          *nodeUpperBounds,
                                 real4          *body_vel,
                                 ulonglong1     *body_id,
                                 real           *body_h, //TODO REMOVE, not used anymore
                                 const float     h_min,  //TODO REMOVE, not used anymore
                                 const float2   *body_dens) {

  CUXTIMER("compute_sph_leaf");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint id  = bid * blockDim.x + tid;

  if (id >= n_leafs) return;


  //Since nodes are intermixed with non-leafs in the node_bodies array
  //we get a leaf-id from the leafsIdxs array
  int nodeID = leafsIdxs[id];

  const uint2 bij          =  node_bodies[nodeID];
  const uint firstChild    =  bij.x & ILEVELMASK;
  const uint lastChild     =  bij.y;  //TODO maybe have to increase it by 1

  //Variables holding properties and intermediate answers
  float4 p;

  double mass, posx, posy, posz;
  mass = posx = posy = posz = 0.0;

  double oct_q11, oct_q22, oct_q33;
  double oct_q12, oct_q13, oct_q23;
  oct_q11 = oct_q22 = oct_q33 = 0.0;
  oct_q12 = oct_q13 = oct_q23 = 0.0;

  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  //Loop over the children=>particles=>bodys
  //unroll increases register usage #pragma unroll 16
  float maxEps  = -100.0f;
  float maxSmth = -100.0f;
  for(int i=firstChild; i < lastChild; i++)
  {
    p      = body_pos[i];

    compute_monopole(mass, posx, posy, posz, p);
    compute_quadropole(oct_q11, oct_q22, oct_q33, oct_q12, oct_q13, oct_q23, p);
    //    compute_bounds(r_min, r_max, p);

    maxSmth = fmaxf(body_dens[i].y, maxSmth);

    //Change the particle into a box and use that as the boundaries
    float  smth  = body_dens[i].y*SPH_KERNEL_SIZE;
    compute_bounds_node(r_min, r_max,
                        make_float4(p.x-smth, p.y-smth, p.z-smth, 0),
                        make_float4(p.x+smth, p.y+smth, p.z+smth, 0));

  }

  double4 mon = {posx, posy, posz, mass};

  double im = 1.0/mon.w;
  if(mon.w == 0) im = 0;        //Allow tracer/massless particles
  mon.x *= im;
  mon.y *= im;
  mon.z *= im;


  double4 Q0, Q1;
  Q0 = make_double4(oct_q11, oct_q22, oct_q33, maxEps); //Store max softening
  Q1 = make_double4(oct_q12, oct_q13, oct_q23, 0.0f);

  //Store the leaf properties
  multipole[3*nodeID + 0] = mon;       //Monopole
  multipole[3*nodeID + 1] = Q0;        //Quadropole
  multipole[3*nodeID + 2] = Q1;        //Quadropole

  //Store the node boundaries
  nodeLowerBounds[nodeID] = make_float4(r_min.x, r_min.y, r_min.z, maxSmth); //4th parameter holds the maximum smoothing range of underlying particles
  nodeUpperBounds[nodeID] = make_float4(r_max.x, r_max.y, r_max.z, 1.0f);    //4th parameter is set to 1 to indicate this is a leaf

  return;
}


KERNEL_DECLARE(compute_leaf)
                            (const int         n_leafs,
                                   uint       *leafsIdxs,
                                   uint2      *node_bodies,
                                   real4      *body_pos,
                                   double4    *multipole,
                                   real4      *nodeLowerBounds,
                                   real4      *nodeUpperBounds,
                                   real4      *body_vel,
                                   ulonglong1 *body_id,
                                   real       *body_h,
			                 const float       h_min,
			                 const float2     *body_dens) {

  CUXTIMER("compute_leaf");
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint id  = bid * blockDim.x + tid;

  //Set the shared memory for these threads and exit the thread
  if (id >= n_leafs) return;


  //Since nodes are intermixes with non-leafs in the node_bodies array
  //we get a leaf-id from the leafsIdxs array
  int nodeID = leafsIdxs[id];

  const uint2 bij          =  node_bodies[nodeID];
  const uint firstChild    =  bij.x & ILEVELMASK;
  const uint lastChild     =  bij.y;  //TODO maybe have to increase it by 1

  //Variables holding properties and intermediate answers
  float4 p;

  double mass, posx, posy, posz;
  mass = posx = posy = posz = 0.0;

  double oct_q11, oct_q22, oct_q33;
  double oct_q12, oct_q13, oct_q23;

  oct_q11 = oct_q22 = oct_q33 = 0.0;
  oct_q12 = oct_q13 = oct_q23 = 0.0;

  float3  r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3  r_max = make_float3(-1e10f, -1e10f, -1e10f);

  float3 r_minSPH = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_maxSPH = make_float3(-1e10f, -1e10f, -1e10f);

  //Loop over the children=>particles=>bodys
  //unroll increases register usage #pragma unroll 16
  float maxEps  = -100.0f;
  float maxSmth = -100.0f;
  int count=0;
  for(int i=firstChild; i < lastChild; i++)
  {
    p      = body_pos[i];
    maxEps = fmaxf(body_vel[i].w, maxEps);      //Determine the max softening within this leaf
    count++;

    compute_monopole(mass, posx, posy, posz, p);
    compute_quadropole(oct_q11, oct_q22, oct_q33, oct_q12, oct_q13, oct_q23, p);
    compute_bounds(r_min, r_max, p);

//    maxSmth = fmaxf(body_dens[i].y, maxSmth);
    //Change the particle into a box and use that as the boundaries
    float  smth  = body_dens[i].y*SPH_KERNEL_SIZE;
    compute_bounds_node(r_minSPH, r_maxSPH,
                        make_float4(p.x-smth, p.y-smth, p.z-smth, 0),
                        make_float4(p.x+smth, p.y+smth, p.z+smth, 0));
  }

  //SPH max smoothing value for the cell
  //Uses the box-extends which makes the box tighter and hence more efficient. ( for example if (boxEdge-particle.pos) > particle-smoothing)

  //Compute the max smoothing range for this cell by determining the max distance between the tightbox and the smoothedbox
  maxSmth     = fmaxf(fmaxf(fmaxf(fabs(r_min.x-r_minSPH.x), fabs(r_max.x-r_maxSPH.x)),
                            fmaxf(fabs(r_min.y-r_minSPH.y), fabs(r_max.y-r_maxSPH.y))),
                            fmaxf(fabs(r_min.z-r_minSPH.z), fabs(r_max.z-r_maxSPH.z)));


  double4 mon = {posx, posy, posz, mass};

  double im = 1.0/mon.w;
  if(mon.w == 0) im = 0;        //Allow tracer/massless particles
  mon.x *= im;
  mon.y *= im;
  mon.z *= im;

  double4 Q0, Q1;
  Q0 = make_double4(oct_q11, oct_q22, oct_q33, maxEps); //Store max softening
  Q1 = make_double4(oct_q12, oct_q13, oct_q23, 0.0f);

  //Store the leaf properties
  multipole[3*nodeID + 0] = mon;       //Monopole
  multipole[3*nodeID + 1] = Q0;        //Quadropole
  multipole[3*nodeID + 2] = Q1;        //Quadropole

  //Store the node boundaries
  nodeLowerBounds[nodeID] = make_float4(r_min.x, r_min.y, r_min.z, maxSmth); //4th parameter holds the maximum smoothing range of underlying particles
  nodeUpperBounds[nodeID] = make_float4(r_max.x, r_max.y, r_max.z, 1.0f);    //4th parameter is set to 1 to indicate this is a leaf

//Compute the box physical center and size
//  float3 boxCenter,boxSize;
//  boxCenter.x   = make_float3(0.5*(r_min.x + r_max.x),
//                              0.5*(r_min.y + r_max.y),
//                              0.5*(r_min.z + r_max.z));
//  boxSize       = make_float3(fmaxf(fabs(boxCenter.x-r_min.x), fabs(boxCenter.x-r_max.x)),
//                              fmaxf(fabs(boxCenter.y-r_min.y), fabs(boxCenter.y-r_max.y)),
//                              fmaxf(fabs(boxCenter.z-r_min.z), fabs(boxCenter.z-r_max.z)));
//Do the same thing for the smoothed size





#if 1
    const float3 len = make_float3(r_max.x-r_min.x, r_max.y-r_min.y, r_max.z-r_min.z);
    const float  vol = cbrtf(len.x*len.y*len.z);
    float hp  = 0;
    if (vol > 0.0f)
    {
      const float nd  = float(lastChild - firstChild) / vol;
      hp  = cbrtf(42.0f / nd);
    }
    hp = max(hp, h_min);
    for(int i=firstChild; i < lastChild; i++)
      if(body_h[i] < 0)
        body_h[i] = hp;
#endif

  return;
}


//Function goes level by level (starting from deepest) and computes
//the properties of the non-leaf nodes
KERNEL_DECLARE(compute_non_leaf)(const int curLevel,         //Level for which we calc
                                 uint  *leafsIdxs,           //Conversion of ids
                                 uint  *node_level_list,     //Contains the start nodes of each lvl
                                 uint  *n_children,          //Reference from node to first child and number of childs
                                 double4 *multipole,
                                 real4 *nodeLowerBounds,
                                 real4 *nodeUpperBounds){

  CUXTIMER("compute_non_leaf");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const int idx = bid * (blockDim.x * blockDim.y) + tid;

  const int endNode   = node_level_list[curLevel];
  const int startNode = node_level_list[curLevel-1];


  if(idx >= (endNode-startNode))     return;

  const int nodeID = leafsIdxs[idx + startNode];

  //Get the children info
  const uint firstChild = n_children[nodeID] & 0x0FFFFFFF;                  //TODO make this name/define?
  const uint nChildren  = ((n_children[nodeID]  & 0xF0000000) >> 28); //TODO make this name/define?

  //Variables
  double mass, posx, posy, posz;
  mass = posx = posy = posz = 0.0;

  double oct_q11, oct_q22, oct_q33;
  double oct_q12, oct_q13, oct_q23;

  oct_q11 = oct_q22 = oct_q33 = 0.0;
  oct_q12 = oct_q13 = oct_q23 = 0.0;

  float3 r_min, r_max;
  r_min = make_float3(+1e10f, +1e10f, +1e10f);
  r_max = make_float3(-1e10f, -1e10f, -1e10f);

  float3 r_minSPH = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_maxSPH = make_float3(-1e10f, -1e10f, -1e10f);

  //Process the children (1 to 8)
  float maxEps  = -100.0f;
  float maxSmth = -100.0f;
  for(int i=firstChild; i < firstChild+nChildren; i++)
  {
    //Gogo process this data!
    double4 tmon = multipole[3*i + 0];

    maxEps  = max(multipole[3*i + 1].w, maxEps);


    compute_monopole_node(mass, posx, posy, posz, tmon);
    compute_quadropole_node(oct_q11, oct_q22, oct_q33, oct_q12, oct_q13, oct_q23,
                            multipole[3*i + 1], multipole[3*i + 2]);

    compute_bounds_node(r_min, r_max, nodeLowerBounds[i], nodeUpperBounds[i]);
    compute_bounds_node(r_minSPH, r_maxSPH,
                        make_float4(nodeLowerBounds[i].x-nodeLowerBounds[i].w, nodeLowerBounds[i].y-nodeLowerBounds[i].w, nodeLowerBounds[i].z-nodeLowerBounds[i].w, 0),
                        make_float4(nodeUpperBounds[i].x+nodeLowerBounds[i].w, nodeUpperBounds[i].y+nodeLowerBounds[i].w, nodeUpperBounds[i].z+nodeLowerBounds[i].w, 0));
  }

  maxSmth     = fmaxf(fmaxf(fmaxf(fabs(r_min.x-r_minSPH.x), fabs(r_max.x-r_maxSPH.x)),
                            fmaxf(fabs(r_min.y-r_minSPH.y), fabs(r_max.y-r_maxSPH.y))),
                            fmaxf(fabs(r_min.z-r_minSPH.z), fabs(r_max.z-r_maxSPH.z)));


  //Save the bounds
  nodeLowerBounds[nodeID] = make_float4(r_min.x, r_min.y, r_min.z, maxSmth);
  nodeUpperBounds[nodeID] = make_float4(r_max.x, r_max.y, r_max.z, 0.0f); //4th is set to 0 to indicate a non-leaf

  //Regularize and store the results
  double4 mon = {posx, posy, posz, mass};
  double im = 1.0/mon.w;
  if(mon.w == 0) im = 0; //Allow tracer/massless particles

  mon.x *= im;
  mon.y *= im;
  mon.z *= im;

  double4 Q0, Q1;
  Q0 = make_double4(oct_q11, oct_q22, oct_q33, maxEps); //store max Eps
  Q1 = make_double4(oct_q12, oct_q13, oct_q23, 0.0f);

  multipole[3*nodeID + 0] = mon;        //Monopole
  multipole[3*nodeID + 1] = Q0;         //Quadropole1
  multipole[3*nodeID + 2] = Q1;         //Quadropole2

  return;
}
KERNEL_DECLARE(compute_scaling)(const int      node_count,
                                      double4 *multipole,
                                      real4   *nodeLowerBounds,
                                      real4   *nodeUpperBounds,
                                      uint    *n_children,
                                      real4   *multipoleF,
                                      float    theta,
                                      real4   *boxSizeInfo,
                                      real4   *boxCenterInfo,
                                      float   *boxSmoothingInfo,
                                      uint2   *node_bodies){

  CUXTIMER("compute_scaling");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  const int idx = bid * (blockDim.x * blockDim.y) + tid;

  if(idx >= node_count)     return;

  double4 monD, Q0, Q1;

  monD = multipole[3*idx + 0];        //Monopole
  Q0   = multipole[3*idx + 1];        //Quadropole1
  Q1   = multipole[3*idx + 2];        //Quadropole2

  //Scale the quadropole
  double im = 1.0 / monD.w;
  if(monD.w == 0) im = 0;               //Allow tracer/massless particles
  Q0.x = Q0.x*im - monD.x*monD.x;
  Q0.y = Q0.y*im - monD.y*monD.y;
  Q0.z = Q0.z*im - monD.z*monD.z;
  Q1.x = Q1.x*im - monD.x*monD.y;
  Q1.y = Q1.y*im - monD.y*monD.z;
  Q1.z = Q1.z*im - monD.x*monD.z;

  //Switch the y and z parameter
  double temp = Q1.y;
  Q1.y = Q1.z; Q1.z = temp;

  //Convert the doubles to floats
  float4 mon            = make_float4(monD.x, monD.y, monD.z, monD.w);
  multipoleF[3*idx + 0] = mon;
  multipoleF[3*idx + 1] = make_float4(Q0.x, Q0.y, Q0.z, Q0.w);        //Quadropole1
  multipoleF[3*idx + 2] = make_float4(Q1.x, Q1.y, Q1.z, Q1.w);        //Quadropole2

  float4 r_min, r_max;
  r_min = nodeLowerBounds[idx];
  r_max = nodeUpperBounds[idx];

  //Compute center and size of the box

  float3 boxCenter;
  boxCenter.x = 0.5*(r_min.x + r_max.x);
  boxCenter.y = 0.5*(r_min.y + r_max.y);
  boxCenter.z = 0.5*(r_min.z + r_max.z);

  float3 boxSize = make_float3(fmaxf(fabs(boxCenter.x-r_min.x), fabs(boxCenter.x-r_max.x)),
                               fmaxf(fabs(boxCenter.y-r_min.y), fabs(boxCenter.y-r_max.y)),
                               fmaxf(fabs(boxCenter.z-r_min.z), fabs(boxCenter.z-r_max.z)));

  //Calculate distance between center of the box and the center of mass
  float3 s3     = make_float3((boxCenter.x - mon.x), (boxCenter.y - mon.y), (boxCenter.z - mon.z));
  double s      = sqrt((s3.x*s3.x) + (s3.y*s3.y) + (s3.z*s3.z));

  //If mass-less particles form a node, the s would be huge in opening angle, make it 0
  if(fabs(mon.w) < 1e-10) s = 0;

  //Length of the box, note times 2 since we only computed half the distance before
  float l = 2*fmaxf(boxSize.x, fmaxf(boxSize.y, boxSize.z));

  //Store the box size and opening criteria
  boxSizeInfo[idx].x = boxSize.x;
  boxSizeInfo[idx].y = boxSize.y;
  boxSizeInfo[idx].z = boxSize.z;
  boxSizeInfo[idx].w = __int_as_float(n_children[idx]);

#if 1
  boxCenterInfo[idx].x = boxCenter.x;
  boxCenterInfo[idx].y = boxCenter.y;
  boxCenterInfo[idx].z = boxCenter.z;
#else /* added by egaburov, see dev_approximate_gravity_warp.cu for matching code*/
  boxCenterInfo[idx].x = mon.x;
  boxCenterInfo[idx].y = mon.y;
  boxCenterInfo[idx].z = mon.z;
#endif

  uint2 bij     = node_bodies[idx];
  uint pfirst   = bij.x & ILEVELMASK;
  uint nchild   = bij.y - pfirst;

  //Change the indirections of the leaf nodes so
  //they point to the particle data
  bool leaf = (r_max.w > 0);
  if(leaf)
  {
    pfirst             = pfirst | ((nchild-1) << LEAFBIT);
    boxSizeInfo[idx].w = __int_as_float(pfirst);
  }



  //Extra check, shouldn't be necessary, probably it is otherwise the test for leaf can fail
  //So it IS important Otherwise 0.0 < 0 can fail, now it will be: -1e-12 < 0 
  if(l < 0.000001)
    l = 0.000001;

  #ifdef IMPBH
    float cellOp = (l/theta) + s;
  #else
    //Minimum distance method
    float cellOp = (l/theta); 
  #endif
    
  cellOp = cellOp*cellOp;


  //If this is (leaf)node with only 1 particle then we change 
  //the opening criteria to a large number to force that the 
  //leaf will be opened and the particle data is used 
  //instead of an approximation.
  //This because sometimes (mass*pos)*(1.0/mass) != pos
  //even in full double precision
  if(nchild == 1)
  {
    cellOp = 10e10; //Force this node to be opened
  }

  if(r_max.w > 0)
  {
    cellOp = -cellOp;       //This is a leaf node
  }

  boxCenterInfo[idx].w = cellOp;


  /* For SPH we re-use boxCenterInfo[idx].w  to hold the max smoothing range
   * of the particles within this cell. TODO use a better variable to hold this?
   */

  //TODO(jbedorf): Remove this after we removed the use of it in parallel.cpp
  //Store the search length squared. This eases the distance comparisons
  boxSmoothingInfo[idx] = 0.0;// (r_min.w)*(r_min.w);


  bool doSPH = true;
  if(doSPH)
  {
      //Make sure that value is non-zero
      if(r_min.w < 0.000001) r_min.w = 0.000001;
      if (r_max.w > 0){
          boxCenterInfo[idx].w    = -(r_min.w*r_min.w);
      }
      else {
          boxCenterInfo[idx].w    =  (r_min.w*r_min.w);
      }
  }

#if 1
      //Change it into half precision, allows storing gravity and sph criteria
      //into a single fp32 value

      __half sph_opening;
      __half tree_opening;

      //Round up and down to make sure we stay on the save side
      if (r_max.w > 0) {
          sph_opening  = __float2half_rd(-(r_min.w*r_min.w));
          tree_opening = __float2half_rd(cellOp);
      } else {
          sph_opening  = __float2half_ru(  r_min.w*r_min.w);
          tree_opening = __float2half_ru(cellOp);
      }

      half2 opening = __halves2half2(tree_opening, sph_opening);
      *((half2*)&boxCenterInfo[idx].w) = opening;
#endif

  return;
}


//Compute the properties for the groups
KERNEL_DECLARE(gpu_setPHGroupData)(const int n_groups,
                                          const int n_particles,   
                                          real4 *bodies_pos,
                                          int2  *group_list,                                                
                                          real4 *groupCenterInfo,
                                          real4 *groupSizeInfo){
  CUXTIMER("setPHGroupData");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if(bid >= n_groups)     return;

  //We set a default amount of shared memory to make sure that the reduction has enough memory
  //independent of the number of threads that is actually launched.
  #if NCRIT > 64
      #error "Fatal, NCRIT > 64 increase shared memory allocation below"
  #endif

  //Do a reduction on the particles assigned to this group
  const int NCRIT_TEMP = 64;
  volatile __shared__ float3 shmem[2*NCRIT_TEMP];
  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[NCRIT_TEMP];

  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  int start = group_list[bid].x;
  int end   = group_list[bid].y;
  
  int partIdx = start + threadIdx.x;

  //Set the shared memory with the data
  if (partIdx >= end)
  {
    sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
    sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;
  }
  else
  {
    sh_rmin[tid].x = r_min.x = bodies_pos[partIdx].x; sh_rmin[tid].y = r_min.y = bodies_pos[partIdx].y; sh_rmin[tid].z = r_min.z = bodies_pos[partIdx].z;
    sh_rmax[tid].x = r_max.x = bodies_pos[partIdx].x; sh_rmax[tid].y = r_max.y = bodies_pos[partIdx].y; sh_rmax[tid].z = r_max.z = bodies_pos[partIdx].z;
  }


  __syncthreads();
  // do reduction in shared mem  
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax2(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax2(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax2(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  if(blockDim.x >= 64) if (tid < 32)  {sh_MinMax2(tid, tid + 32, &r_min, &r_max, sh_rmin, sh_rmax); }
  if(blockDim.x >= 32) if (tid < 16) { sh_MinMax2(tid, tid + 16, &r_min, &r_max, sh_rmin, sh_rmax); }

  if(tid < 8)
  {
    sh_MinMax2(tid, tid +  8, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  4, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  2, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  1, &r_min, &r_max, sh_rmin, sh_rmax);
  }
  
  // write result for this block to global mem
  if (tid == 0)
  {

    //Compute the group center and size
    float3 grpCenter;
    grpCenter.x = 0.5*(r_min.x + r_max.x);
    grpCenter.y = 0.5*(r_min.y + r_max.y);
    grpCenter.z = 0.5*(r_min.z + r_max.z);

    float3 grpSize = make_float3(fmaxf(fabs(grpCenter.x-r_min.x), fabs(grpCenter.x-r_max.x)),
                                 fmaxf(fabs(grpCenter.y-r_min.y), fabs(grpCenter.y-r_max.y)),
                                 fmaxf(fabs(grpCenter.z-r_min.z), fabs(grpCenter.z-r_max.z)));

    //Store the box size and opening criteria
    groupSizeInfo[bid].x = grpSize.x;
    groupSizeInfo[bid].y = grpSize.y;
    groupSizeInfo[bid].z = grpSize.z;

    int nchild             = end-start;
    start                  = start | (nchild-1) << CRITBIT;
    groupSizeInfo[bid].w   = __int_as_float(start);  

    float l = max(grpSize.x, max(grpSize.y, grpSize.z));

    groupCenterInfo[bid].x = grpCenter.x;
    groupCenterInfo[bid].y = grpCenter.y;
    groupCenterInfo[bid].z = grpCenter.z;

    //Test stats for physical group size
    groupCenterInfo[bid].w = l;

  } //end tid == 0
}//end copyNode2grp



#if 0
//Compute the properties for the groups
KERNEL_DECLARE(gpu_setPHGroupDataGetKey)(const int n_groups,
                                          const int n_particles,
                                          real4 *bodies_pos,
                                          int2  *group_list,
                                          real4 *groupCenterInfo,
                                          real4 *groupSizeInfo,
                                          uint4  *body_key,
                                          float4 corner){
  CUXTIMER("setPHGroupDataGetKey");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;

  if(bid >= n_groups)     return;

  //Do a reduction on the particles assigned to this group

  volatile __shared__ float3 shmem[2*NCRIT];
  volatile float3 *sh_rmin = (float3*)&shmem [ 0];
  volatile float3 *sh_rmax = (float3*)&shmem[NCRIT];

  float3 r_min = make_float3(+1e10f, +1e10f, +1e10f);
  float3 r_max = make_float3(-1e10f, -1e10f, -1e10f);

  int start = group_list[bid].x;
  int end   = group_list[bid].y;

  int partIdx = start + threadIdx.x;

  //Set the shared memory with the data
  if (partIdx >= end)
  {
    sh_rmin[tid].x = r_min.x; sh_rmin[tid].y = r_min.y; sh_rmin[tid].z = r_min.z;
    sh_rmax[tid].x = r_max.x; sh_rmax[tid].y = r_max.y; sh_rmax[tid].z = r_max.z;
  }
  else
  {
    sh_rmin[tid].x = r_min.x = bodies_pos[partIdx].x; sh_rmin[tid].y = r_min.y = bodies_pos[partIdx].y; sh_rmin[tid].z = r_min.z = bodies_pos[partIdx].z;
    sh_rmax[tid].x = r_max.x = bodies_pos[partIdx].x; sh_rmax[tid].y = r_max.y = bodies_pos[partIdx].y; sh_rmax[tid].z = r_max.z = bodies_pos[partIdx].z;
  }


  __syncthreads();
  // do reduction in shared mem
  if(blockDim.x >= 512) if (tid < 256) {sh_MinMax2(tid, tid + 256, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 256) if (tid < 128) {sh_MinMax2(tid, tid + 128, &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();
  if(blockDim.x >= 128) if (tid < 64)  {sh_MinMax2(tid, tid + 64,  &r_min, &r_max, sh_rmin, sh_rmax);} __syncthreads();

  if(blockDim.x >= 64) if (tid < 32)  {sh_MinMax2(tid, tid + 32, &r_min, &r_max, sh_rmin, sh_rmax); }
  if(blockDim.x >= 32) if (tid < 16) { sh_MinMax2(tid, tid + 16, &r_min, &r_max, sh_rmin, sh_rmax); }

  if(tid < 8)
  {
    sh_MinMax2(tid, tid +  8, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  4, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  2, &r_min, &r_max, sh_rmin, sh_rmax);
    sh_MinMax2(tid, tid +  1, &r_min, &r_max, sh_rmin, sh_rmax);
  }

  // write result for this block to global mem
  if (tid == 0)
  {

    //Compute the group center and size
    float3 grpCenter;
    grpCenter.x = 0.5*(r_min.x + r_max.x);
    grpCenter.y = 0.5*(r_min.y + r_max.y);
    grpCenter.z = 0.5*(r_min.z + r_max.z);

    float3 grpSize = make_float3(fmaxf(fabs(grpCenter.x-r_min.x), fabs(grpCenter.x-r_max.x)),
                                 fmaxf(fabs(grpCenter.y-r_min.y), fabs(grpCenter.y-r_max.y)),
                                 fmaxf(fabs(grpCenter.z-r_min.z), fabs(grpCenter.z-r_max.z)));

    //Store the box size and opening criteria
    groupSizeInfo[bid].x = grpSize.x;
    groupSizeInfo[bid].y = grpSize.y;
    groupSizeInfo[bid].z = grpSize.z;

    real4 pos = bodies_pos[start];

    int nchild             = end-start;
    start                  = start | (nchild-1) << CRITBIT;
    groupSizeInfo[bid].w   = __int_as_float(start);

    float l = max(grpSize.x, max(grpSize.y, grpSize.z));

    groupCenterInfo[bid].x = grpCenter.x;
    groupCenterInfo[bid].y = grpCenter.y;
    groupCenterInfo[bid].z = grpCenter.z;

    //Test stats for physical group size
    groupCenterInfo[bid].w = l;

    int4 crd;

    real domain_fac = corner.w;

    #ifndef EXACT_KEY
       crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
       crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
       crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
    #else
       crd.x = (int)((pos.x - corner.x) / domain_fac);
       crd.y = (int)((pos.y - corner.y) / domain_fac);
       crd.z = (int)((pos.z - corner.z) / domain_fac);
    #endif

    body_key[bid] = get_key(crd);

  } //end tid == 0
}//end copyNode2grp

//Compute the key for the groups
KERNEL_DECLARE(gpu_setPHGroupDataGetKey2)(const int n_groups,
                                      real4 *bodies_pos,
                                      int2  *group_list,
                                      uint4  *body_key,
                                      float4 corner){
  CUXTIMER("setPHGroupDataGetKey2");
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int idx = bid * (blockDim.x * blockDim.y) + tid;

  if(idx >= n_groups)     return;


  int start = group_list[idx].x;
  real4 pos = bodies_pos[start];

//  int end   = group_list[idx].y-1;
//  real4 pos = bodies_pos[end];

//  int end   = group_list[idx].y-1;
//  int start = group_list[idx].x;
//  start     = (end+start) / 2;
//  real4 pos = bodies_pos[start];


  int4 crd;

  real domain_fac = corner.w;

  #ifndef EXACT_KEY
     crd.x = (int)roundf(__fdividef((pos.x - corner.x), domain_fac));
     crd.y = (int)roundf(__fdividef((pos.y - corner.y) , domain_fac));
     crd.z = (int)roundf(__fdividef((pos.z - corner.z) , domain_fac));
  #else
     crd.x = (int)((pos.x - corner.x) / domain_fac);
     crd.y = (int)((pos.y - corner.y) / domain_fac);
     crd.z = (int)((pos.z - corner.z) / domain_fac);
  #endif
    // uint2 key =  get_key_morton(crd);
    // body_key[idx] = make_uint4(key.x, key.y, 0,0);
    body_key[idx] = get_key(crd); //has to be PH key in order to prevent the need for sorting

}//end copyNode2grp

#endif

