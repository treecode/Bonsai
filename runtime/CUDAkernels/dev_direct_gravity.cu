#include "bonsai.h"

__device__ float3
bodyBodyInteraction(float3 ai,
                    float4 bi,
                    float4 bj,
                    float  eps2)
{
    float3 r;

    // r_ij  [3 FLOPS]
    r.x = bj.x - bi.x;
    r.y = bj.y - bi.y;
    r.z = bj.z - bi.z;

    // distSqr = dot(r_ij, r_ij) + EPS^2  [6 FLOPS]
    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z;
    distSqr += eps2;

    // invDistCube =1/distSqr^(3/2)  [4 FLOPS (2 mul, 1 sqrt, 1 inv)]
    float invDist = rsqrtf(distSqr);
    float invDistCube =  invDist * invDist * invDist;

    // s = m_j * invDistCube [1 FLOP]
    float s = bj.w * invDistCube;

    // a_i =  a_i + s * r_ij [6 FLOPS]
    ai.x += r.x * s;
    ai.y += r.y * s;
    ai.z += r.z * s;

    return ai;
}


// This is the "tile_calculation" function from the GPUG3 article.

__device__ float3
gravitation(float4 iPos,
            float3 accel,
            float4 *sharedPos,
            float  eps2)
{
    // The CUDA 1.1 compiler cannot determine that i is not going to
    // overflow in the loop below.  Therefore if int is used on 64-bit linux
    // or windows (or long instead of long long on win64), the compiler
    // generates suboptimal code.  Therefore we use long long on win64 and
    // long on everything else. (Workaround for Bug ID 347697)
#ifdef _Win64
    unsigned long long j = 0;
#else
    unsigned long j = 0;
#endif

    // Here we unroll the loop to reduce bookkeeping instruction overhead
    // 32x unrolling seems to provide best performance

    // Note that having an unsigned int loop counter and an unsigned
    // long index helps the compiler generate efficient code on 64-bit
    // OSes.  The compiler can't assume the 64-bit index won't overflow
    // so it incurs extra integer operations.  This is a standard issue
    // in porting 32-bit code to 64-bit OSes.

#pragma unroll 32

    for (unsigned int counter = 0; counter < blockDim.x; counter++)
    {
        accel = bodyBodyInteraction(accel, iPos, sharedPos[j++], eps2);
    }

    return accel;
}

// WRAP is used to force each block to start working on a different
// chunk (and wrap around back to the beginning of the array) so that
// not all multiprocessors try to read the same memory locations at
// once.
#define WRAP(x,m) (((x)<(m))?(x):((x)-(m)))  // Mod without divide, works on values from 0 up to 2m

//JB, different numbers of i-particles and j-particles incase tree.n_dust and tree.n are unequal
KERNEL_DECLARE(dev_direct_gravity)(float4 *accel, float4 *i_positions, float4 *j_positions, int numBodies_i, int numBodies_j, float eps2)
{
    extern __shared__ float4 sharedPos[];

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    if (index >= numBodies_i)
    {
        return;
    }

    float4 iPos = i_positions[index];

    float3 acc = {0.0f, 0.0f, 0.0f};

    int p        = blockDim.x;
    int n        = numBodies_j;
    int numTiles = (n + p - 1) / p;


    for (int tile = blockIdx.y; tile < numTiles + blockIdx.y; tile++)
    {
        int jindex = WRAP(blockIdx.x + tile, gridDim.x) * p + threadIdx.x;
        if (jindex < numBodies_j)
          sharedPos[threadIdx.x] = j_positions[jindex];
        else
          sharedPos[threadIdx.x] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        __syncthreads();

        // This is the "tile_calculation" function from the GPUG3 article.
        acc = gravitation(iPos, acc, sharedPos, eps2);

        __syncthreads();
    }

    accel[index] = make_float4(acc.x, acc.y, acc.z, 0.f);
}
