#ifndef SPH_INCLUDE_H
#define SPH_INCLUDE_H



#ifdef WIN32
    #define M_PI        3.14159265358979323846264338328
#endif



/***********************************/
/***** DENSITY   ******************/

/* FDPS Kernel */


#define  PARAM_SMTH 1.2


//__device__ __forceinline__ float pow3(float a) { return a*a*a; }
//__device__ __forceinline__ float W(const PS::F64vec dr, const PS::F64 h) const{
//    const float H = supportRadius() * h;
//    const float s = sqrt(dr * dr) / H;
//    float r_value;
//    r_value = (1.0 + s * (8.0 + s * (25.0 + s * (32.0)))) * math::pow8(math::plus(1.0 - s));
//    r_value *= (1365./64.) / (H * H * H * math::pi);
//    return r_value;
//}

namespace SPH
{
    //Wendland C6
    struct kernel_t{
        //W
        __device__ __forceinline__ float W(const float dr, const float h) const{
            const float H = supportRadius() * h;
            const float s = dr / H;
            float r_value;
            r_value = (1.0f + s * (8.0f + s * (25.0f + s * (32.0f)))) * pow8(plus(1.0f - s));
            r_value *= (1365.f/64.f) / (H * H * H * M_PI);
            return r_value;
        }

        //gradW
        __device__ __forceinline__ float abs_gradW(const float r, const float h) const{
              const float H = supportRadius() * h;
              const float s = r / H;
              float r_value;
              r_value = pow7(plus(1.0f - s)) * (plus(1.0f - s) * (8.0f + s * (50.0f + s * (96.0f))) - 8.0f * (1.0f + s * (8.0f + s * (25.0f + s * (32.0f)))));
              r_value *= (1365.f/64.f) / (H * H * H * M_PI);
              return r_value / (H  + 1.0e-6 * h);
             }


        static __device__  __forceinline__ float supportRadius(){
            return SPH_KERNEL_SIZE;
        }

        template <typename type>  type
        static __device__ __forceinline__  pow7(const type arg){
            const type arg2 = arg * arg;
            const type arg4 = arg2 * arg2;
            return arg4 * arg2 * arg;
        }

        template <typename type>  type
        static __device__ __forceinline__  pow8(const type arg){
            const type arg2 = arg * arg;
            const type arg4 = arg2 * arg2;
            return arg4 * arg4;
        }

        template <typename type>  type
        static __device__ __forceinline__  plus(const type arg){
            return (arg > 0) ? arg : 0;
        }
    };

    namespace density
    {
        typedef struct  __attribute__((aligned(8)))  data
        {
            float dens;
            float smth;

            __device__ __forceinline__ void finalize(const float mass)
            {
                //Normalize the smoothing range
                smth = PARAM_SMTH * cbrtf(mass / dens);
            }

            __device__ __forceinline__ void clear() {dens = 0; smth = 0;}

            __device__ __forceinline__ void operator=(SPH::density::data t) {dens=t.dens; smth=t.smth;};
            __device__ __forceinline__ float2 operator=(float2 t) {
                   dens=t.x;
                   smth=t.y;
                   return t;
               }

        } data;
    }

    namespace derivative
    {
        typedef struct  __attribute__((aligned(16)))  data
        {
            float x,y,z,w;
            __device__ __forceinline__ void finalize(const float density)
            {
               //Normalize by density range
               x /= density;
               y /= density;
               z /= density;
               w /= density;
            }
            __device__ __forceinline__ void clear() {x = 0; y = 0; z= 0; w = 0;}

            __device__ __forceinline__ void operator  =(SPH::derivative::data t) {x=t.x; y=t.y; z=t.z; w=t.w;};
            __device__ __forceinline__ float4 operator=(float4 t) {
                   x=t.x; y=t.y; x=t.z; y=t.w;
                   return t;
               }

        } data;
    }

    namespace hydroforce
    {
//        typedef struct  __attribute__((aligned(16)))  data
//        {
//            float x,y,z,w;
//            __device__ __forceinline__ void finalize(const float density)
//            {
//               //Normalize by density range
//               x /= density;
//               y /= density;
//               z /= density;
//               w /= density;
//            }
//            __device__ __forceinline__ void clear() {x = 0; y = 0; z= 0; w = 0;}
//
//            __device__ __forceinline__ void operator  =(SPH::derivative::data t) {x=t.x; y=t.y; z=t.z; w=t.w;};
//            __device__ __forceinline__ float4 operator=(float4 t) {
//                   x=t.x; y=t.y; x=t.z; y=t.w;
//                   return t;
//               }
//
//        } data;
    }

enum SPHVal {DENSITY, DERIVATIVE, HYDROFORCE};

#if 1
namespace density
{
    static __device__ __forceinline__ void addDensity(
        const float4    pos,
        const float     massj,
        const float3    posj,
        const float     eps2,
              float    &dens,
              int      &temp,
        const SPH::kernel_t &kernel)
    {
    #if 1  // to test performance of a tree-walk
      const float3 dr    = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = r2;// + eps2;
      const float r      = sqrtf(r2eps);

      temp = (fabs(massj*kernel.W(r, pos.w)) > 0);

      dens += massj*kernel.W(r, pos.w);
      //dens += 1;
      //Prevent adding ourself, TODO should make this 'if' more efficient, like multiply with something
      //Not needed for this SPH kernel?
      //if(r2 != 0) density +=tempD;
    #endif
    }

#if 1
    static __device__ __forceinline__ void addDensity2(
        const float4    pos,
        const float4    dr,
        const float     eps2,
              float    &dens,
              int      &temp,
        const SPH::kernel_t &kernel)
    {
    #if 1  // to test performance of a tree-walk
//      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = dr.x;// + eps2;
      const float r      = sqrtf(r2eps);

      temp = (fabs(dr.w*kernel.W(r, pos.w)) > 0);

      dens += dr.w*kernel.W(r, pos.w);

    #endif
    }

    static __device__ __forceinline__ void addDensity3(
        const float4    pos,
        const float     r2,
        const float     massj,
        const float     eps2,
              float    &dens,
        const SPH::kernel_t &kernel)
    {
    #if 1
      const float r      = sqrtf(r2);
      dens += massj*kernel.W(r, pos.w);
    #endif
    }
#endif

    template<int NI, bool FULL>
    struct directOperator {

        static const SPHVal type = SPH::DENSITY;




        __device__ __forceinline__ void operator()(
                float4  acc_i[NI],
          const float4  pos_i[NI],
          const float4  vel_i[NI],
          const int     ptclIdx,
          const float   eps2,
          SPH::density::data  density_i[NI],
          SPH::derivative::data gradient_i[NI],
          const float4  hydro_i[NI],  //Not used here
          const float4 *body_jpos,
          const float4 *body_jvel,
          const float2 *body_jdens,   //Not used here
          const float4 *body_hydro    //Not used here
          )
        {
          SPH::kernel_t kernel;
#if 1
          const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
//          for (int j = 0; j < WARP_SIZE; j++)
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl(M0.x, j ), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
            const float  jmass = jM0.w;
            const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
        #pragma unroll
            for (int k = 0; k < NI; k++)
            {
              int temp = 0;
              addDensity(pos_i[k], jmass, jpos, eps2, density_i[k].dens, temp, kernel);
//              density_i[k].smth++;


#if 0
          //    if(pos_i[0].x == 0.0 && pos_i[0].y == 0 && (pos_i[0].z > 0.03124 && pos_i[0].z < 0.03126))
       // if(((laneId + 1) % 8) == 0)
        if(pos_i[0].x == 0.5 && pos_i[0].y == 0.5 && pos_i[0].z == 49.5)
        {
            if(temp > 0)
            {
                printf("%d Interaction with: %f %f %f  with  %f\t%f\t%f \n", laneId, pos_i[0].x,pos_i[0].y,pos_i[0].z, jpos.x, jpos.y, jpos.z);
            }
        } 
#endif
              gradient_i[k].x++;       //Number of operations
              gradient_i[k].y += temp; //Number of useful operations
            }
          }
        }
#elif 0
        //Version that shuffles the particle index instead of the particle-data
        //With the idea being that each particle can work on it's own useful particle-index
        //once we have a list of useful particle indices...

        for (int j = 0; j < WARP_SIZE; j++)
        {
          int ptclIdx2 = __shfl(ptclIdx,j);
          const float4 jM0 = (FULL || ptclIdx2 >= 0) ? body_jpos[ptclIdx2] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float  jmass = jM0.w;
          const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
      #pragma unroll
          for (int k = 0; k < NI; k++)
          {
            int temp = 0;
            addDensity(pos_i[k], jmass, jpos, eps2, density_i[k].dens, temp, kernel);
            density_i[k].smth++;

            gradient_i[k].x++;       //Number of operations
            gradient_i[k].y += temp; //Number of useful operations
          }
        }
      }
#elif 0
        //Version that shuffles the particle index instead of the particle-data
        //With the idea being that each particle can work on it's own useful particle-index
        //once we have a list of useful particle indices...

        for (int j = 0; j < WARP_SIZE; j++)
        {
          int ptclIdx2 = __shfl(ptclIdx,j);
          const float4 jM0 = (FULL || ptclIdx2 >= 0) ? body_jpos[ptclIdx2] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float  jmass = jM0.w;
          const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
      #pragma unroll
          for (int k = 0; k < NI; k++)
          {
            int temp = 0;
            addDensity(pos_i[k], jmass, jpos, eps2, density_i[k].dens, temp, kernel);
            density_i[k].smth++;

            //Count occurrences in which all threads have to process this particle
            //temp = __all(temp);

            //Count the particles that are useful for at least half the threads
            //temp = __ballot(temp);
            //temp = (__popc(temp) >= 16);

            //Count the cases where I am the only one who finds this particle useful
            int temp2 = __ballot(temp);
            temp2 = (__popc(temp2));
            if(temp2 == 1 && temp == 1) temp = 1;
            else temp = 0;


            gradient_i[k].x++;       //Number of operations
            gradient_i[k].y += temp; //Number of useful operations
          }
        }
      }
#elif 0
        //Version that makes a personal interaction list, too bad it's 10x slower because memory bus being overloaded...
        float iH = pos_i[0].w*kernel.supportRadius();
        iH      *= iH;

        const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, -1.0f);

        for (int j = 0; j < WARP_SIZE; j++)
        {
            const float4 jM0   = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
            const float3 dr    = make_float3(jM0.x - pos_i[0].x, jM0.y - pos_i[0].y, jM0.z - pos_i[0].z);
            const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
            if(r2 <= iH && jM0.w >= 0) //Only use valid particles (no negative mass)
            {
                //pB[pC] = make_float4(dr.x, dr.y, dr.z, jM0.w);
                pB[pC] = make_float4(r2, dr.y, dr.z, jM0.w);
                pC++;
            }
        }

        //If one of the lists is more than half full we evaluate the currently stored particles
        if(__any(pC > 32) || !FULL)
        {
            density_i[0].smth += 1;

            gradient_i[0].x+= 32;
            const int k = pC > 32 ? 32 : pC;
            for(int z=0; z < k; z++)
            //for(int z=0; z < pC; z++)
            {
              int temp = 0;
              addDensity2(pos_i[0], pB[z], eps2, density_i[0].dens, temp, kernel);

              gradient_i[0].y++;
            }
            pC = 0;
        }
    }
#else
        //Version that first determines which of the ptclIndices are useful and then processes only
        //those particles
        float iH = pos_i[0].w*kernel.supportRadius();
        iH      *= iH;

        int markedIdx = 0;

        const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, -1.0f);

        pB[laneId] = M0;

        for (int j = 0; j < WARP_SIZE; j++)
        {
            const float4 jM0   = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
            const float3 dr    = make_float3(jM0.x - pos_i[0].x, jM0.y - pos_i[0].y, jM0.z - pos_i[0].z);
            const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;

            bool useParticle   = (r2 <= iH && jM0.w >= 0);
            //Warp vote to determine if this particle is useful to all threads, if so use it right away
//            if(__all(useParticle)) addDensity3(pos_i[0], r2, jM0.w, eps2, density_i[0].dens, kernel);
//            else
                markedIdx         |= useParticle << j;

            gradient_i[0].x++;       //Number of operations
//            density_i[0].smth += useParticle;
        }

        //Sum particles we need
//        gradient_i[0].y += __popc(markedIdx); //Number of useful operations


        //Process the particles we find useful
        while(__popc(markedIdx))
        {
            int idx             = __ffs(markedIdx)-1;   //The index of a useful particle
            density_i[0].smth += 1;

            //Zero the just used index
            markedIdx = markedIdx & (~(1 << idx));

            float   jmass = pB[idx].w;
            float3  jpos  = make_float3(pB[idx].x, pB[idx].y, pB[idx].z);
            int temp;

            addDensity(pos_i[0], jmass, jpos, eps2, density_i[0].dens, temp, kernel);
        }
    }
#endif

    };
}; //namespace density
#endif

#if 0
namespace density
{


    static __device__ __forceinline__ void addDensity(
        const float4    pos,
        const float     massj,
        const float3    posj,
        const float     eps2,
              float    &dens,
              int      &temp,
        const SPH::kernel_t &kernel)
    {
    #if 1  // to test performance of a tree-walk
      const float3 dr    = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = r2;// + eps2;
      const float r      = sqrtf(r2eps);

      temp = (fabs(massj*kernel.W(r, pos.w)) > 0);

      dens += massj*kernel.W(r, pos.w);
      //Prevent adding ourself, TODO should make this 'if' more efficient, like multiply with something
      //Not needed for this SPH kernel?
      //if(r2 != 0) density +=tempD;
    #endif
    }




    template<int NI, bool FULL>
    struct directOperator {

        static const SPHVal type = SPH::DENSITY;

        __device__ __forceinline__ void operator()(
                float4  acc_i[NI],
          const float4  pos_i[NI],
          const float4  vel_i[NI],
          const int     ptclIdx,
          const float   eps2,
          SPH::density::data  density_i[NI],
          SPH::derivative::data gradient_i[NI],
          const float4  hydro_i[NI],  //Not used here
          const float4 *body_jpos,
          const float4 *body_jvel,
          const float2 *body_jdens,   //Not used here
          const float4 *body_hydro    //Not used here
          )
        {
          SPH::kernel_t kernel;
          const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          for (int j = 0; j < WARP_SIZE; j++)
          {
            const float4 jM0   = make_float4(__shfl(M0.x, j), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
            const float  jmass = jM0.w;
            const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
        #pragma unroll
            for (int k = 0; k < NI; k++)
            {
              int temp = 0;
              addDensity(pos_i[k], jmass, jpos, eps2, density_i[k].dens, temp, kernel);
              density_i[k].smth++;

              gradient_i[k].x++;
              gradient_i[k].y += temp;
            }
          }
        }
    };
}; //namespace density
#endif

namespace derivative
{
    static __device__ __forceinline__ void addParticleEffect(
        const float4    posi,
        const float4    veli,
        const float     massj,
        const float3    posj,
        const float3    velj,
        const float     eps2,
              SPH::derivative::data  &gradient,
        const SPH::kernel_t &kernel)
    {
      const float3 dr    = make_float3(posj.x - posi.x, posj.y - posi.y, posj.z - posi.z);
      const float3 dv    = make_float3(velj.x - veli.x, velj.y - veli.y, velj.z - veli.z);
      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = r2;// + eps2;
      const float r      = sqrtf(r2eps);

      const float abs_gradW = kernel.abs_gradW(r, posi.w);

      const float4 gradW = (r > 0) ? make_float4(abs_gradW * dr.x / r, abs_gradW * dr.y / r, abs_gradW * dr.z / r, 0.0) : (float4){0.0, 0.0, 0.0, 0.0};

      gradient.x -= massj* (dv.y * gradW.z - dv.z * gradW.y);
      gradient.y -= massj* (dv.z * gradW.x - dv.x * gradW.z);
      gradient.z -= massj* (dv.x * gradW.y - dv.y * gradW.x);
      gradient.w -= massj* (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z);
    }





    template<int NI, bool FULL>
    struct directOperator {

         static const SPHVal type = SPH::DERIVATIVE;
#if 0
        __device__ __forceinline__ void operator()(
                  float4  acc_i[NI],
            const float4  pos_i[NI],
            const float4  vel_i[NI],
            const int     ptclIdx,
            const float   eps2,
            SPH::density::data  density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],  //Not used here
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,   //Not used here
            const float4 *body_hydro    //Not used here
            )
        {
          SPH::kernel_t kernel;
          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
//          for (int j = 0; j < WARP_SIZE; j++)
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl(MP.x, j), __shfl(MP.y, j), __shfl(MP.z, j), __shfl(MP.w,j));
            const float  jmass = jM0.w;
            const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
            const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));

        #pragma unroll
            for (int k = 0; k < NI; k++)
            {
              addParticleEffect(pos_i[k], vel_i[k], jmass, jpos, jvel, eps2, gradient_i[k], kernel);
            }
          }
        }
#else
        __device__ __forceinline__ void operator()(
                  float4  acc_i[NI],
            const float4  pos_i[NI],
            const float4  vel_i[NI],
            const int     ptclIdx,
            const float   eps2,
            SPH::density::data  density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],  //Not used here
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,   //Not used here
            const float4 *body_hydro    //Not used here
            )
        {
          SPH::kernel_t kernel;
          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
//          for (int j = 0; j < WARP_SIZE; j++)
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
                const float4 jM0      = make_float4(__shfl(MP.x, j), __shfl(MP.y, j), __shfl(MP.z, j), __shfl(MP.w,j));
                const float3 dr       = make_float3(jM0.x - pos_i[0].x, jM0.y - pos_i[0].y, jM0.z - pos_i[0].z);
                const float r         = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);

                const float abs_gradW = kernel.abs_gradW(r, pos_i[0].w);
                const float4 gradW = (r > 0) ? make_float4(abs_gradW * dr.x / r, abs_gradW * dr.y / r, abs_gradW * dr.z / r, 0.0) : (float4){0.0, 0.0, 0.0, 0.0};

                const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));
                const float3 dv    = make_float3(jvel.x - vel_i[0].x, jvel.y - vel_i[0].y, jvel.z - vel_i[0].z);
                gradient_i[0].x -= jM0.w* (dv.y * gradW.z - dv.z * gradW.y);
                gradient_i[0].y -= jM0.w* (dv.z * gradW.x - dv.x * gradW.z);
                gradient_i[0].z -= jM0.w* (dv.x * gradW.y - dv.y * gradW.x);
                gradient_i[0].w -= jM0.w* (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z);
          }
        }

#endif


    };
}; //namespace derivative


namespace hydroforce
{
//real4> bodies_hydro;   //The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
    static __device__ __forceinline__ void addParticleEffect(
        const float4    posi,
        const float4    veli,
        const float     massj,
        const float3    posj,
        const float3    velj,
        const float     eps2,
        const float4    hydroi,             //x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
        const float4    hydroj,             //x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
        const float     densi,
        const float     densj,
        const float     smthj,
              float4    &acc,
              float     &v_sig_max,
        const SPH::kernel_t &kernel)
    {
      const float3 dr    = make_float3(posi.x - posj.x, posi.y - posj.y, posi.z - posj.z);
      const float3 dv    = make_float3(veli.x - velj.x, veli.y - velj.y, veli.z - velj.z);
      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = r2;// + eps2;
      const float r      = sqrtf(r2eps);

      const float xv_inner = dr.x * dv.x + dr.y * dv.y + dr.z * dv.z;

      //AV
      const float w_ij      = (xv_inner < 0.0f) ? xv_inner / r : 0.0f;
      const float v_sig     = hydroi.y + hydroj.y - 3.0f * w_ij;
                  v_sig_max = (v_sig_max < v_sig) ? v_sig : v_sig_max;
      const float AV        = - 0.5f * v_sig * w_ij / (0.5f * (densi + densj)) * 0.5f * (hydroi.w + hydroj.w);
      //
      const float ith_abs_gradW = kernel.abs_gradW(r, posi.w);
      const float jth_abs_gradW = (smthj != 0) ? kernel.abs_gradW(r, smthj) : 0.0f; //Returns NaN if smthj is 0 which happens if we do not use the particle, so 'if' for now
      const float abs_gradW     = 0.5f * (ith_abs_gradW + jth_abs_gradW);
      const float4 gradW        = (r > 0) ? make_float4(abs_gradW * dr.x / r, abs_gradW * dr.y / r, abs_gradW * dr.z / r, 0.0f)
                                          : make_float4(0.0f,0.0f,0.0f,0.0f);

      float temp = massj * (hydroi.x / (densi * densi) + hydroj.x / (densj * densj) + AV);
      temp       = (smthj != 0) ? temp : 0; //Same as above, a 0 smoothing length leads to NaN (divide by 0)
      acc.x           -= temp  * gradW.x;
      acc.y           -= temp  * gradW.y;
      acc.z           -= temp  * gradW.z;
//      acc.y           += (fabs(abs_gradW) > 0);  //Count how often we do something useful in this function
//      acc.z           += 1; //Count how often we enter this function
      acc.w           += massj * (hydroi.x / (densi * densi) + 0.5 * AV) * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z); //eng_dot
    }


    template<int NI, bool FULL>
    struct directOperator {

         static const SPHVal type = SPH::HYDROFORCE;

#if 0
        __device__ __forceinline__ void operator()(
                  float4  acc_i[NI],
            const float4  pos_i[NI],
            const float4  vel_i[NI],
            const int     ptclIdx,
            const float   eps2,
            SPH::density::data    density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,
            const float4 *body_hydro
            , float4 *pB,int    &pC)
        {
          float v_sig_max = 0; //TODO implement this value/keep track of it over various directOp calls

          SPH::kernel_t kernel;
          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MH = (FULL || ptclIdx >= 0) ? body_hydro[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float2 MD = (FULL || ptclIdx >= 0) ? body_jdens[ptclIdx] : make_float2(0.0f, 0.0f);

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
          //for (int j = 0; j < WARP_SIZE; j++)
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl(MP.x, j), __shfl(MP.y, j), __shfl(MP.z, j), __shfl(MP.w,j));
            const float  jmass = jM0.w;
            const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);
            const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));

            const float4 jH    = make_float4(__shfl(MH.x, j), __shfl(MH.y, j), __shfl(MH.z, j), __shfl(MH.w,j));
            const float2 jD    = make_float2(__shfl(MD.x, j), __shfl(MD.y, j));

        #pragma unroll
            for (int k = 0; k < NI; k++)
            {
                addParticleEffect(pos_i[k], vel_i[k], jmass, jpos, jvel, eps2, hydro_i[k], jH, density_i[k].dens, jD.x, jD.y, acc_i[k], v_sig_max, kernel);
            }
          }//for WARP_SIZE

          //force[id].dt = PARAM::C_CFL * 2.0 * ith.smth / v_sig_max;
        }
#else

        __device__ __forceinline__ void operator()(
                  float4  acc_i[NI],
            const float4  pos_i[NI],
            const float4  vel_i[NI],
            const int     ptclIdx,
            const float   eps2,
            SPH::density::data    density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,
            const float4 *body_hydro)
        {
          float v_sig_max = 0; //TODO implement this value/keep track of it over various directOp calls

          SPH::kernel_t kernel;
          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MH = (FULL || ptclIdx >= 0) ? body_hydro[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float2 MD = (FULL || ptclIdx >= 0) ? body_jdens[ptclIdx] : make_float2(0.0f, 0.0f);

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
          //for (int j = 0; j < WARP_SIZE; j++)
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl(MP.x, j), __shfl(MP.y, j), __shfl(MP.z, j), __shfl(MP.w,j));
            const float3 dr    = make_float3(pos_i[0].x - jM0.x, pos_i[0].y - jM0.y, pos_i[0].z - jM0.z);

            const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));
            const float3 dv    = make_float3(vel_i[0].x - jvel.x, vel_i[0].y - jvel.y, vel_i[0].z - jvel.z);


            //
            const float r        = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
            const float xv_inner = dr.x * dv.x + dr.y * dv.y + dr.z * dv.z;
            const float w_ij     = (xv_inner < 0.0f) ? xv_inner / r : 0.0f;

            const float2 jD    = make_float2(__shfl(MD.x, j), __shfl(MD.y, j));
            //
            const float ith_abs_gradW = kernel.abs_gradW(r, pos_i[0].w);
            const float jth_abs_gradW = (jD.y != 0) ? kernel.abs_gradW(r, jD.y) : 0.0f; //Returns NaN if smthj is 0 which happens if we do not use the particle, so 'if' for now
            const float abs_gradW     = 0.5f * (ith_abs_gradW + jth_abs_gradW);
            const float4 gradW        = (r > 0) ? make_float4(abs_gradW * dr.x / r, abs_gradW * dr.y / r, abs_gradW * dr.z / r, 0.0f)
                                                : make_float4(0.0f,0.0f,0.0f,0.0f);
            //AV
            const float3 jH       = make_float3(__shfl(MH.x, j), __shfl(MH.y, j), __shfl(MH.w,j)); //Note w in z location

            const float v_sig     = hydro_i[0].y + jH.y - 3.0f * w_ij;
                        v_sig_max = (v_sig_max < v_sig) ? v_sig : v_sig_max;
            const float AV        = - 0.5f * v_sig * w_ij / (0.5f * (density_i[0].dens + jD.x)) * 0.5f * (hydro_i[0].w + jH.z);


            float temp = jM0.w * (hydro_i[0].x / (density_i[0].dens * density_i[0].dens) + jH.x / (jD.x * jD.x) + AV);
            temp       = (jD.y != 0) ? temp : 0; //Same as above, a 0 smoothing length leads to NaN (divide by 0)

      //      acc.y           += (fabs(abs_gradW) > 0);  //Count how often we do something useful in this function
      //      acc.z           += 1; //Count how often we enter this function
            acc_i[0].w           += jM0.w * (hydro_i[0].x / (density_i[0].dens * density_i[0].dens)  + 0.5 * AV) * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z); //eng_dot
            acc_i[0].x           -= temp  * gradW.x;
            acc_i[0].y           -= temp  * gradW.y;
            acc_i[0].z           -= temp  * gradW.z;
          }//for WARP_SIZE

          //force[id].dt = PARAM::C_CFL * 2.0 * ith.smth / v_sig_max;
        }
#endif

    };
}; //namespace hydroforce








} //namespace SPH

#endif
