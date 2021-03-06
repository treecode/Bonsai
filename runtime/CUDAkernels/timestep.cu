#include "bonsai.h"
#ifdef __DEVICE_EMULATION__
  #define EMUSYNC __syncthreads();
#else
  #define EMUSYNC
#endif
#include "../profiling/bonsai_timing.h"
PROF_MODULE(timestep);

#include "node_specs.h"


//Reduce function to get the minimum timestep
static __device__ __forceinline__ void get_TnextD(const int n_bodies,
                                     float2 *time,
                                     float *tnext, volatile float *sdata) {
  //float2 time : x is time begin, y is time end

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdata[tid] = 1.0e10f;
  float tmin = 1.0e10f;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies) tmin = fminf(tmin, time[i            ].y);
    if (i + blockSize < n_bodies) tmin = fminf(tmin, time[i + blockSize].y);

    i += gridSize;
  }

  sdata[tid] = tmin;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 256]); } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 128]); } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  64]); } __syncthreads(); }
#ifndef __DEVICE_EMULATION__
  if (tid < 32)
#endif
    {
      if (blockSize >=  64) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 32]); EMUSYNC; }
      if (blockSize >=  32) { sdata[tid] = tmin = fminf(tmin, sdata[tid + 16]); EMUSYNC; }
      if (blockSize >=  16) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  8]); EMUSYNC; }
      if (blockSize >=   8) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  4]); EMUSYNC; }
      if (blockSize >=   4) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  2]); EMUSYNC; }
      if (blockSize >=   2) { sdata[tid] = tmin = fminf(tmin, sdata[tid +  1]); EMUSYNC; }
  }

  // write result for this block to global mem
  if (tid == 0) tnext[blockIdx.x] = sdata[0];
}

KERNEL_DECLARE(get_Tnext)(const int n_bodies,
                                     float2 *time,
                                     float *tnext) {
  extern __shared__ float sdata[];
  get_TnextD(n_bodies, time, tnext, sdata);
}


//Reduce function to get the number of active particles
static __device__ void get_nactiveD(const int n_bodies,
                                       uint *valid,
                                       uint *tnact, volatile int *sdataInt) {
  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;
  sdataInt[tid] = 0;
  int sum       = 0;

  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies) sum = sum + valid[i            ];
    if (i + blockSize < n_bodies) sum = sum + valid[i + blockSize];

    i += gridSize;
  }
  sdataInt[tid] = sum;
  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) { sdataInt[tid] = sum = sum + sdataInt[tid + 256]; } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) { sdataInt[tid] = sum = sum + sdataInt[tid + 128]; } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) { sdataInt[tid] = sum = sum + sdataInt[tid +  64]; } __syncthreads(); }


#ifndef __DEVICE_EMULATION__
  if (tid < 32)
#endif
    {
      if (blockSize >=  64) { sdataInt[tid] = sum = sum + sdataInt[tid + 32]; EMUSYNC; }
      if (blockSize >=  32) { sdataInt[tid] = sum = sum + sdataInt[tid + 16]; EMUSYNC; }
      if (blockSize >=  16) { sdataInt[tid] = sum = sum + sdataInt[tid +  8]; EMUSYNC; }
      if (blockSize >=   8) { sdataInt[tid] = sum = sum + sdataInt[tid +  4]; EMUSYNC; }
      if (blockSize >=   4) { sdataInt[tid] = sum = sum + sdataInt[tid +  2]; EMUSYNC; }
      if (blockSize >=   2) { sdataInt[tid] = sum = sum + sdataInt[tid +  1]; EMUSYNC; }
  }

  // write result for this block to global mem
  if (tid == 0) tnact[blockIdx.x] = sdataInt[0];
}

//Reduce function to get the number of active particles
KERNEL_DECLARE(get_nactive)(const int n_bodies,
                                       uint *valid,
                                       uint *tnact) {
  extern __shared__ int sdataInt[];
  get_nactiveD(n_bodies, valid, tnact, sdataInt);
}

KERNEL_DECLARE(predict_particles)(const int 	n_bodies,
										float 	tc,
										float 	tp,
										real4 	*pos,
										real4 	*vel,
										real4 	*acc,
										float2 	*time,
										real4 	*pPos,
										real4 	*pVel){
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;


  if (idx >= n_bodies) return;

  float4 p = pos [idx];
  float4 v = vel [idx];
  float4 a = acc [idx];
  float tb = time[idx].x;

  #ifdef DO_BLOCK_TIMESTEP
    float dt_cb  = tc - tb;
  #else
    float dt_cb  = tc - tp;
    time[idx].x  = tp;
  #endif

//   float dt_pb  = tp - tb;

  p.x += v.x*dt_cb + a.x*dt_cb*dt_cb*0.5f;
  p.y += v.y*dt_cb + a.y*dt_cb*dt_cb*0.5f;
  p.z += v.z*dt_cb + a.z*dt_cb*dt_cb*0.5f;

  v.x += a.x*dt_cb;
  v.y += a.y*dt_cb;
  v.z += a.z*dt_cb;

  pPos[idx] = p;
  pVel[idx] = v;
}


KERNEL_DECLARE(setActiveGroups)(const int n_bodies,
                                            float tc,
                                            float2 *time,
                                            uint  *body2grouplist,
                                            uint  *valid_list){
  const uint bid = blockIdx.y * gridDim.x + blockIdx.x;
  const uint tid = threadIdx.x;
  const uint idx = bid * blockDim.x + tid;

  if (idx >= n_bodies) return;

  float te = time[idx].y;

  //Set the group to active if the time current = time end of
  //this particle. Can be that multiple particles write to the
  //same location but the net result is the same
  int grpID = body2grouplist[idx];

  //valid_list[grpID] = grpID | ((tc == te) << 31);

  if(tc == te)
  {
    valid_list[grpID] = grpID | (1 << 31);
  }
}


static __device__ __forceinline__ float adjustH(const float h_old, const float nnb)
{
	const float nbDesired 	= 32;
	const float f      	= 0.5f * (1.0f + cbrtf(nbDesired / nnb));
	const float fScale 	= max(min(f, 2.0), 0.5);
	return (h_old*fScale);
}
KERNEL_DECLARE(correct_particles)(const int n_bodies,
                                  /*  1 */   float tc,
                                  /*  2 */   float2 *time,
                                  /*  3 */   uint   *active_list,
                                  /*  4 */   real4 *vel,
                                  /*  5 */   real4 *acc0,
                                  /*  6 */   real4 *acc1,
                                  /*  7 */   float   *body_h,
                                  /*  8 */   float2  *body_dens,
                                  /*  9 */   real4 *pos,
                                  /* 10 */   real4 *pPos,
                                  /* 11 */   real4 *pVel,
                                  /* 12 */   uint  *unsorted,
                                  /* 13 */   real4 *acc0_new,
                                  	  	  	 float2 *time_new)
{
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid =  threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;
  if (idx >= n_bodies) return;

  //Check if particle is set to active during approx grav
  #ifdef DO_BLOCK_TIMESTEP
    if (active_list[idx] != 1) return;
  #endif


#if 0
  float4 v  = vel [idx];
  float4 a0 = acc0[idx];
  float4 a1 = acc1[idx];
  float  tb = time[idx].x;
  v = pVel[idx];
#else
  const uint unsortedIdx = unsorted[idx];


  float4 a0 = acc0[unsortedIdx];
  float4 a1 = acc1[idx];
  float  tb = time[unsortedIdx].x;
  float4 v  = pVel[unsortedIdx];

#endif

  //Store the predicted position as the one to use
  pos[idx] = pPos[idx];


  float dt_cb  = tc - tb;

  //Correct the velocity
  dt_cb *= 0.5f;
  v.x += (a1.x - a0.x)*dt_cb;
  v.y += (a1.y - a0.y)*dt_cb;
  v.z += (a1.z - a0.z)*dt_cb;


  //Store the corrected velocity, accelaration and the new time step info
  vel     [idx] = v;
  acc0_new[idx] = a1;
  time_new[idx] = time[unsortedIdx];
  unsorted[idx] = idx;  //Have to reset it in case we do not resort the particles

  //Adjust the search radius for the next iteration to get closer to the
  //requested number of neighbours
  body_h[idx] = adjustH(body_h[idx], body_dens[idx].y);
}



extern "C"  __global__ void compute_dt(const int n_bodies,
                                       float    tc,
                                       float    eta,
                                       int      dt_limit,
                                       float    eps2,
                                       float2   *time,
                                       real4    *vel,
                                       int      *ngb,
                                       real4    *bodies_pos,
                                       real4    *bodies_acc,
                                       uint     *active_list,
                                       float    timeStep){
  const int bid =  blockIdx.y *  gridDim.x +  blockIdx.x;
  const int tid =  threadIdx.y * blockDim.x + threadIdx.x;
  const int dim =  blockDim.x * blockDim.y;

  int idx = bid * dim + tid;
  if (idx >= n_bodies) return;

  //Check if particle is set to active during approx grav
  if (active_list[idx] != 1) return;

  int j = ngb[idx];

  float4 ri, rj;
  float4 vi, vj;
  float4 ai, aj;

  float ds2;
  ri = bodies_pos[idx];
  vi = vel[idx];
  ai = bodies_acc[idx];
  int j1, j2;

  if (j >= 0) {
    rj = bodies_pos[j];
    float3 dr = {ri.x - rj.x,
                 ri.y - rj.y,
                 ri.z - rj.z};
    ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
  } else  {
    j1 = max(0, idx - 1);
    rj = bodies_pos[j1];
    float3 dr = {ri.x - rj.x,
                 ri.y - rj.y,
                 ri.z - rj.z};
    if (idx != j1)  ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
    else            ds2 = 1.0e10f;

    j2 = min(n_bodies-1, idx + 1);
    rj = bodies_pos[j2];
    dr = make_float3(ri.x - rj.x,
                     ri.y - rj.y,
                     ri.z - rj.z);
    if (idx != j2) {
      if (dr.x*dr.x + dr.y*dr.y + dr.z*dr.z < ds2) {
        ds2 = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
        j = j2;
      } else {
        j = j1;
      };
    } else {
      j = j1;
    }
  }
  ds2 = ds2*__powf(10.0f, 0.666667f) + eps2;
//   ds2 += eps2;
  vj = vel[j];
  aj = bodies_acc[j];

  float3 vda = {ai.x - aj.x,
                ai.y - aj.y,
                ai.z - aj.z};
  float3 vdv = {vi.x - vj.x,
                vi.y - vj.y,
                vi.z - vj.z};
  float da = sqrtf(vda.x*vda.x + vda.y*vda.y + vda.z*vda.z);
  float dv = sqrtf(vdv.x*vdv.x + vdv.y*vdv.y + vdv.z*vdv.z);
  float ds = sqrtf(ds2);

  float dt = eta * dv/da*(sqrt(2*da*ds/(dv*dv) + 1) - 1);

  int power = -(int)__log2f(dt) + 1;
  power     = max(power, dt_limit);

  dt = 1.0f/(1 << power);
  while(fmodf(tc, dt) != 0.0f) dt *= 0.5f;      // could be slow!

//  dt = 0.015625;
  dt = 1.0f/(1 << 8);
  dt = 1.0f/(1 << 6);
  dt = 1.0f/(1 << 7);
  dt = timeStep;
  time[idx].x = tc;
  time[idx].y = tc + dt;
}



//Reduce function to get the energy of the system in double precision
static __device__ void compute_energy_doubleD(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            double2 *energy, volatile double *shDDataKin) {

  // perform first level of reduction,
  // reading from global memory, writing to shared memory
  const int blockSize   = blockDim.x;
  unsigned int tid      = threadIdx.x;
  unsigned int i        = blockIdx.x*(blockSize*2) + threadIdx.x;
  unsigned int gridSize = blockSize*2*gridDim.x;

  volatile double *shDDataPot = (double*)&shDDataKin [blockSize];
  double eKin, ePot;
  shDDataKin[tid] = eKin = 0;   //Stores Ekin
  shDDataPot[tid] = ePot = 0;   //Stores Epot

  real4 temp;
  // we reduce multiple elements per thread.  The number is determined by the
  // number of active thread blocks (via gridSize).  More blocks will result
  // in a larger gridSize and therefore fewer elements per thread
  while (i < n_bodies) {
    if (i             < n_bodies)
    {
      temp  = vel[i];
      eKin += pos[i].w*0.5*(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
      ePot += pos[i].w*0.5*acc[i].w;
    }

    if (i + blockSize < n_bodies)
    {
      temp  = vel[i + blockSize];
      eKin += pos[i + blockSize].w*0.5*(temp.x*temp.x + temp.y*temp.y + temp.z*temp.z);
      ePot += pos[i + blockSize].w*0.5*acc[i + blockSize].w;
    }

    i += gridSize;
  }
  shDDataKin[tid] = eKin;
  shDDataPot[tid] = ePot;

  __syncthreads();

  // do reduction in shared mem
  if (blockSize >= 512) { if (tid < 256) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 256];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 256];   } __syncthreads(); }
  if (blockSize >= 256) { if (tid < 128) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 128];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 128];   } __syncthreads(); }
  if (blockSize >= 128) { if (tid <  64) {
    shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 64];
    shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 64];   } __syncthreads(); }


#ifndef __DEVICE_EMULATION__
  if (tid < 32)
#endif
    {
      if (blockSize >=  64) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 32]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 32];  EMUSYNC; }
      if (blockSize >=  32) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid + 16]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid + 16];  EMUSYNC; }
      if (blockSize >=  16) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid +  8]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid +  8];  EMUSYNC; }
      if (blockSize >=   8) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid +  4]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid +  4];  EMUSYNC; }
      if (blockSize >=   4) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid +  2]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid +  2];  EMUSYNC; }
      if (blockSize >=   2) {shDDataKin[tid] = eKin = eKin + shDDataKin[tid +  1]; shDDataPot[tid] = ePot = ePot + shDDataPot[tid +  1];  EMUSYNC; }
  }

  // write result for this block to global mem
  if (tid == 0) energy[blockIdx.x] = make_double2(shDDataKin[0], shDDataPot[0]);
}


//Reduce function to get the energy of the system
KERNEL_DECLARE(compute_energy_double)(const int n_bodies,
                                            real4 *pos,
                                            real4 *vel,
                                            real4 *acc,
                                            double2 *energy) {
  extern __shared__ double shDDataKin[];
  compute_energy_doubleD(n_bodies, pos, vel, acc, energy, shDDataKin);
}



