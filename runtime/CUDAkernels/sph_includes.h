#ifndef SPH_INCLUDE_H
#define SPH_INCLUDE_H



#ifdef WIN32
    #define M_PI        3.14159265358979323846264338328
#endif


#define USE_BALSARA_SWITCH



/***********************************/
/***** DENSITY   ******************/

/* FDPS Kernel */


//#define STATS



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

    struct base_kernel
    {
        static __device__  __forceinline__ float supportRadius(){
            return SPH_KERNEL_SIZE;
        }

        template <typename type>  type
        static __device__ __forceinline__  pow3(const type arg){
            const type arg2 = arg * arg;
            const type arg3 = arg * arg2;
            return arg3;
        }

        template <typename type>  type
        static __device__ __forceinline__  pow4(const type arg){
            const type arg2 = arg * arg;
            const type arg4 = arg2 * arg2;
            return arg4;
        }


        template <typename type>  type
        static __device__ __forceinline__  pow5(const type arg){
            const type arg2 = arg * arg;
            const type arg4 = arg2 * arg2;
            return arg4 * arg;
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

    //Quintic kernel as defined in Phantom
#ifdef KERNEL_QUINTIC
    struct kernel_t : public base_kernel { //kernel_t_quintic

        static constexpr float cnormk = 1./(120.*M_PI);

        /* this one works with #define SPH_KERNEL_SIZE 3.0f #define  PARAM_SMTH 1.0 */
        //Quintic kernel as defined in Phantom
        __device__ __forceinline__ float W(const float dr, const float h) const{

            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi31 = hi*hi21;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = cnormk*hi31; //Quintic kernel

            if(q < 1.0f)
            {
                const float q4 = q2*q2;
                return cnormkh*(-10.0f*q4*q + 30.0f*q4 - 60.0f*q2 + 66.0f);
            }
            else if(q < 2.0f)
            {
                return cnormkh*(-pow5(q - 3.0f) + 6.0f*pow5((q - 2.0f)));
            }
            else if(q < 3.0f)
            {
                return cnormkh*(-pow5(q - 3.0f));
            }
            else
            {
                return 0;
            }
        }

        __device__ __forceinline__ float abs_gradW(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi41 = hi21*hi21;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = cnormk*hi41; //Quintic kernel

            if(q < 1.0f)
            {
                return cnormkh*(q*(-50.0f*q2*q + 120.0f*q2 - 120.0f));
            }
            else if(q < 2.0f)
            {
                return  cnormkh*(-5.0f*pow4(q - 3.0f) + 30.0f*pow4(q - 2.0f));
            }
            else if(q < 3.0f)
            {
                return cnormkh*(-5.0f*pow4(q - 3.0f));
            }
            else
            {
                return 0;
            }
        }

        __device__ __forceinline__ float abs_gradW2(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = 1; //Quintic kernel

            if(q < 1.0f)
            {
                return cnormkh*(q*(-50.0f*q2*q + 120.0f*q2 - 120.0f));
            }
            else if(q < 2.0f)
            {
                return  cnormkh*(-5.0f*pow4(q - 3.0f) + 30.0f*pow4(q - 2.0f));
            }
            else if(q < 3.0f)
            {
                return cnormkh*(-5.0f*pow4(q - 3.0f));
            }
            else
            {
                return 0;
            }
        }

        //Combined kernel, does not multiply with hi3 and/or hi4 nor with cnorm. This is done as final
        //step in the dev_sph code.
        __device__ __forceinline__ float abs_gradW2(const float dr, const float h, float &w) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;

            if(q < 1.0f)
            {
                const float q4 = q2*q2;
                w = (-10.0f*q4*q + 30.0f*q4 - 60.0f*q2 + 66.0f);
                return (q*(-50.0f*q2*q + 120.0f*q2 - 120.0f));
            }
            else if(q < 2.0f)
            {
                w = -pow5(q - 3.0f) + 6.0f*pow5((q - 2.0f));
                return  (-5.0f*pow4(q - 3.0f) + 30.0f*pow4(q - 2.0f));
            }
            else if(q < 3.0f)
            {
                w = -pow5(q - 3.0f);
                return (-5.0f*pow4(q - 3.0f));
            }
            else
            {
                w = 0;
                return 0;
            }
        }
    };
#endif

#ifdef KERNEL_M_4
    //M_4 kernel as defined in Phantom
    //struct kernel_t_m4 : public base_kernel {
    struct kernel_t : public base_kernel {

        static constexpr float cnormk = 1./M_PI;


        //M4 kernel as defined in Phantom
        __device__ __forceinline__ float W(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi31 = hi*hi21;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = cnormk*hi31; //Quintic kernel

            if(q < 1.0f)
            {
                return cnormkh*(0.75f*q2*q - 1.5f*q2 + 1.0f);
            }
            else if(q < 2.0f)
            {
                return cnormkh*(-0.25f*pow3(q-2.0f));
            }
            else
            {
                return 0;
            }
        }

        __device__ __forceinline__ float abs_gradW(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi41 = hi21*hi21;
            const float q  = dr*hi;

            const float cnormkh = cnormk*hi41; //Quintic kernel

            if(q < 1.0f)
            {
                return cnormkh*q*(2.25f*q - 3.0f);
            }
            else if(q < 2.0f)
            {
                return cnormkh*(-0.75f*((q-2.0f)*(q-2.0f)));
            }
            else
            {
                return 0;
            }
        }

        //Combined kernel, does not multiply with hi3 and/or hi4 nor with cnorm. This is done as final
        //step in the dev_sph code.
        __device__ __forceinline__ float abs_gradW2(const float dr, const float h, float &w) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;

            if(q < 1.0f)
            {
                w = (0.75f*q2*q - 1.5f*q2 + 1.0f);
                return q*(2.25f*q - 3.0f);
            }
            else if(q < 2.0f)
            {
                w = (-0.25f*pow3(q-2.0f));
                return  -0.75f*((q-2.0f)*(q-2.0f));
            }
            else
            {
                w = 0;
                return 0;
            }
        }
    };
#endif

#ifdef KERNEL_W_C6
    //Wendland C6 kernel as defined in Phantom
    struct kernel_t : public base_kernel {
        static constexpr float cnormk = 1365./(512.*M_PI);

        __device__ __forceinline__ float W(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float hi31 = hi*hi21;

            if(q >= 2.0) return 0;
            return hi31*cnormk*(pow8(0.5*q - 1.0f)*(4.0f*q2*q + 6.25f*q2 + 4.0f*q + 1.0f));

        }
        __device__ __forceinline__ float abs_gradW(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float hi41 = hi21*hi21;

            if(q >= 2.0) return 0;
            return hi41*cnormk*q*pow7(0.5*q - 1.0f)*(22.0f*q2 + 19.25f*q + 5.5f);
        }

        //Combined kernel, does not multiply with hi3 and/or hi4 nor with cnorm. This is done as final
        //step in the dev_sph code.
        __device__ __forceinline__ float abs_gradW2(const float dr, const float h, float &w) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;

            if(q >= 2.0)
            {
                w = 0.0f;
                return 0;
            }
            w = pow8(0.5*q - 1.0f)*(4.0f*q2*q + 6.25f*q2 + 4.0f*q + 1.0f);
            return q*pow7(0.5*q - 1.0f)*(22.0f*q2 + 19.25f*q + 5.5f);
         }

    };
#endif

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

#if 1
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
#endif

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

    static __device__ __forceinline__ void addDensity_gradh(
        const float4    pos,
        const float     massj,
        const float3    posj,
        const float     eps2,
              float    &dens,
              float    &gradh,
              float    &temp,
        const SPH::kernel_t &kernel)
    {
    #if 1  // to test performance of a tree-walk
      //const float3 dr    = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);
        float3 dr    = make_float3(posj.x - pos.x, posj.y - pos.y, posj.z - pos.z);


      const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
      const float r2eps  = r2;// + eps2;
      const float r      = sqrtf(r2eps);
      const float qi     = r/pos.w;

//      if(r2eps == 0) return;

      float temp1, temp2;
      temp2 = kernel.abs_gradW2(r, pos.w, temp1);

      temp = temp1;
      dens  += massj*temp1;  //Correct one

      //Use this for testing
//      if(massj == 0) temp1 = 0;
//      dens  += temp1;


      gradh += massj*(-qi*temp2 - 3*temp1);

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
          const sphParameters     SPHParams,
          SPH::density::data  density_i[NI],
          SPH::derivative::data gradient_i[NI],
          const float4  hydro_i[NI],  //Not used here
          const float4 *body_jpos,
          const float4 *body_jvel,
          const float2 *body_jdens,   //Not used here
          const float4 *body_hydro,    //Not used here
          const unsigned long long IDi,
          const unsigned long long *IDs
          )
        {
          SPH::kernel_t kernel;
#if 0
          const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

          const int  IDjx = (FULL || ptclIdx >= 0) ? IDs[ptclIdx] : 0;

//          if(IDi == 64767){
//              printf("On dev [ %d %d ] interact2-in: %.16lg \n", blockIdx.x, threadIdx.x, density_i[0].dens);
//          }

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
              float temp2 = gradient_i[k].x;
//              float temp3 = density_i[k].dens;
              float temp = 0;
//              addDensity(pos_i[k], jmass, jpos, eps2, density_i[k].dens, temp, kernel);
              addDensity_gradh(pos_i[k], jmass, jpos, eps2, density_i[k].dens, gradient_i[k].x, temp, kernel);



              const int IDj = __shfl(IDjx, j);
//

//              if(IDi == 64767){
//                 //if(0 != (gradient_i[k].x-temp2) && jmass > 0)
//////                  if(IDj == 100000000)
//                  if(temp != 0)
//                  {
//
//
//                     const float3 dr    = make_float3(jpos.x - pos_i[k].x, jpos.y - pos_i[k].y, jpos.z - pos_i[k].z);
//                     const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
////
////                     float hi1  = 1.0f/ pos_i[0].w;
////                     float hi21 = hi1*hi1;
////                     float phantom = r2*hi21;
//
//                     int tempID = (IDj >= 100000000 ? IDj-100000000 : IDj);
//                     printf("ON DEV %d interact: %d dist: %.16lg \t sum: %f increase: %.16lg\n",
//                             (int)IDi, tempID+1, sqrtf(r2), density_i[k].dens, temp);
////                     printf("ON DEV %d interact: %d dist: %.16lg \t %.16f %.16f %.16f\n",
////                                               (int)IDi, tempID+1, sqrtf(r2), dr.x,dr.y,dr.z);
//
////                             phantom, hi21, dr.x,dr.y,dr.z,jpos.x,jpos.y,jpos.z);
//
//
//
//
////                      printf("ON DEV %d interact: %d dist: %f res: %f diff:\t %lg %lg || Sum: %f %f %f \t %f %f %f \n",
////                              (int)IDi, IDj,
////                              r2,
////                              83157344.000000*(gradient_i[k].x-temp2),
////                              (density_i[k].dens-temp3)/jmass,
////                              temp,
////                              gradient_i[k].x/jmass,
//////                              gradient_i[k].x-temp2,
////                              83157344.000000*gradient_i[k].x,
////                              pos_i[k].x, pos_i[k].y, pos_i[k].z,
////                              jpos.x, jpos.y, jpos.z);
//                  }
//              }

//              gradient_i[k].x++;       //Number of operations
//              gradient_i[k].y += temp; //Number of useful operations

            }
          }
//          if(IDi == 64767){
//              printf("On dev [ %d %d ] interact2-out: %.16lg \n", blockIdx.x, threadIdx.x, density_i[0].dens);
//          }
        }


#elif 1

        const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        const int  IDjx = (FULL || ptclIdx >= 0) ? IDs[ptclIdx] : 0;


        const int NGROUPTemp = NCRIT;
        const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
        for (int j = offset; j < offset+NGROUPTemp; j++)
        {
          const float4 jM0   = make_float4(__shfl(M0.x, j ), __shfl(M0.y, j), __shfl(M0.z, j), __shfl(M0.w,j));
          const float  jmass = jM0.w;
          const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);


      #pragma unroll
          for (int k = 0; k < NI; k++)
          {
            float3 dr    = make_float3(jpos.x - pos_i[k].x, jpos.y - pos_i[k].y, jpos.z - pos_i[k].z);
            const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
            const float r2eps  = r2;// + eps2;
            const float r      = sqrtf(r2eps);
            const float qi     = r/pos_i[k].w;

            float temp1, temp2;
            temp2 = kernel.abs_gradW2(r, pos_i[k].w, temp1); //temp1 is density kernel, temp2 = derivative kernel

            density_i[k].dens  += jmass*temp1;                   //Density
            acc_i[k].x         += jmass*(-qi*temp2 - 3*temp1);   //Derivative

#ifdef USE_BALSARA_SWITCH
            //Balsara switch, TODO(jbedorf): In theory this is only needed when we perform the final density iteration
            const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));
            const float3 dv    = make_float3(jvel.x - vel_i[0].x, jvel.y - vel_i[0].y, jvel.z - vel_i[0].z);

            const float4 gradW = (r > 0) ? make_float4(temp2 * dr.x / r, temp2 * dr.y / r, temp2 * dr.z / r, 0.0) : (float4){0.0, 0.0, 0.0, 0.0};
            gradient_i[0].x -= jmass * (dv.y * gradW.z - dv.z * gradW.y);
//            gradient_i[0].y -= jmass * (dv.z * gradW.x - dv.x * gradW.z);
//            gradient_i[0].z -= jmass * (dv.x * gradW.y - dv.y * gradW.x);
            gradient_i[0].w -= jmass * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z);

            //For interaction stats
#ifdef STATS
            gradient_i[0].z++;       //Number of operations
            gradient_i[0].y += fabs(jmass * temp1) > 0; //Number of useful operations
#else
            gradient_i[0].y -= jmass * (dv.z * gradW.x - dv.x * gradW.z);
            gradient_i[0].z -= jmass * (dv.x * gradW.y - dv.y * gradW.x);
#endif
#endif //USE_BALSARA_SWITCH

            //End Balsara
          } //for k
        } //for offset
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



namespace hydroforce
{
//real4> bodies_hydro;   //The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
#if 0
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

      //Natsuki
      acc.w           += massj * (hydroi.x / (densi * densi) + 0.5 * AV) * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z); //eng_dot
      //Gasoline acc.w           += massj * (hydroi.x / (densi * densj) + 0.5 * AV) * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z); //eng_dot
    }
#endif

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
            const sphParameters     SPHParams,
            SPH::density::data    density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],              //x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,
            const float4 *body_hydro,
            const unsigned long long IDi,
            const unsigned long long *IDs)
        {
          //Get v_sig_max from gradient_i x ?
          float v_sig_max = 0; //TODO implement this value/keep track of it over various directOp calls

          float omegai = vel_i[0].w;


          SPH::kernel_t kernel;
          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MH = (FULL || ptclIdx >= 0) ? body_hydro[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float2 MD = (FULL || ptclIdx >= 0) ? body_jdens[ptclIdx] : make_float2(0.0f, 0.0f);
          const int  IDjx = (FULL || ptclIdx >= 0) ? IDs[ptclIdx] : -1;

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl(MP.x, j), __shfl(MP.y, j), __shfl(MP.z, j), __shfl(MP.w,j));
            const float3 dr    = make_float3(pos_i[0].x - jM0.x, pos_i[0].y - jM0.y, pos_i[0].z - jM0.z);

            const float3 jvel  = make_float3(__shfl(MV.x, j), __shfl(MV.y, j), __shfl(MV.z, j));
            const float3 dv    = make_float3(vel_i[0].x - jvel.x, vel_i[0].y - jvel.y, vel_i[0].z - jvel.z);

            const float omegaj = __shfl(MV.w, j);

            //
            const float r        = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
            const float xv_inner = dr.x * dv.x + dr.y * dv.y + dr.z * dv.z;
            const float w_ij     = (xv_inner < 0.0f) ? xv_inner / r : 0.0f;
            const float projv    = (r != 0) ? xv_inner / r : 0.0f;

            const float2 jD      = make_float2(__shfl(MD.x, j), __shfl(MD.y, j));
            //
            float ith_abs_gradW = kernel.abs_gradW(r, pos_i[0].w);
            float jth_abs_gradW = (jD.y != 0) ? kernel.abs_gradW(r, jD.y) : 0.0f; //Returns NaN if smthj is 0 which happens if we do not use the particle, so 'if' for now
            ith_abs_gradW *= omegai;
            jth_abs_gradW *= omegaj;



            const int IDj = __shfl(IDjx, j);
            if(IDi+1 == -26001 && IDj+1 == 16400)
            {
                printf("ON DEV %d with %d  grkern-i: %f  grkern-j: %f \n", (int)IDi+1, IDj+1, ith_abs_gradW, jth_abs_gradW);
//                kernel.abs_gradW_print(r, jD.y);
                printf("ON DEV2b  %d with %d  grkern-j: %f\t gradh: %f \n", (int)IDi+1, IDj+1, kernel.abs_gradW(r, jD.y), omegaj);
            }


            const float abs_gradW     = 0.5f * (ith_abs_gradW + jth_abs_gradW);
        //    const float4 gradW        = (r > 0) ? make_float4(abs_gradW * dr.x / r, abs_gradW * dr.y / r, abs_gradW * dr.z / r, 0.0f)
        //                                        : make_float4(0.0f,0.0f,0.0f,0.0f);

            //AV
            const float4 jH       = make_float4(__shfl(MH.x, j), __shfl(MH.y, j), __shfl(MH.z,j), __shfl(MH.w,j)); //OLD: Note w in z location

#if 1
            const float aAV       = SPHParams.av_alpha; //AV_ALPHA; //1.0f; //Alpha parameter for artificial viscosity
            const float bAV       = SPHParams.av_beta; //AV_BETA; //2.0f; //Beta parameter for artificial viscosity
            const float aC        = SPHParams.ac_param; //AC_PARAM; //1.0f; //Artificial Conductivity parameter

            const float v_sigi     = aAV*hydro_i[0].y - bAV * projv; //a*csi - b*drdv
            const float v_sigj     = aAV*jH.y         - bAV * projv; //a*csj - b*drdv



            //Determine the maximum for the dt variable
            const float v_sig = max(v_sigi,v_sigj);
            v_sig_max = (v_sig_max < v_sig) ? v_sig : v_sig_max;

//                  float AV        = - 0.5f * v_sig * w_ij / (0.5f * (density_i[0].dens + jD.x));
//                  AV             *= 0.5f * (hydro_i[0].w + jH.w);   //Eq 63
            //AV  =  AV  * (0.5*(BalA+BalB)) , see RossWog text between Eq 63 and 59
#else
            const float aAV   = 1.0f;
            const float bAV   = 3.0f;
            const float v_sig     = aAV*hydro_i[0].y + jH.y - bAV * w_ij;

            const float hij   = 0.5 * (density_i[0].smth + jD.y);
            //const float uij   = hij * xv_inner / (r*r + 0.0001 * hij * hij); //As in gadget2, Eq 12
            const float uij   = hij * xv_inner / (r*r + 0.01 *(hij * hij)); //As in gadget2, Eq 12
                  float AV    = -aAV*(0.5*(hydro_i[0].y + jH.y))*uij + bAV*(uij*uij); //Gadget2 eq 11
                        AV    = AV / (0.5f * (density_i[0].dens + jD.x));
                        AV   *= 0.5f * (hydro_i[0].w + jH.z);   //Eq 63

#endif

           float PA = 0, PB2 = 0, AVi = 0, AVj = 0;
           float PAi  =  (1.0 / (density_i[0].dens* density_i[0].dens))*hydro_i[0].x;  //Note to self, can compute this in outerloops
           float PAj  =  (1.0 / (jD.x * jD.x))* jH.x ;

           if(w_ij < 0)
           {
               //Artificial viscosity switches, note that we do this per particle
               AVi = 0.5f*(1.0 / density_i[0].dens)*v_sigi*projv;
               AVj = 0.5f*(1.0 / jD.x)*v_sigj*projv;

#ifdef USE_BALSARA_SWITCH
               //Multiply with the Balsara switch
               AVi *= hydro_i[0].w;
               AVj *= jH.w;
#endif
           }

           //Force according to equation 120 , with the addition of the Artificial Viscosity
           PA   = jM0.w*(PAi-AVi)*ith_abs_gradW;
           PB2  = jM0.w*(PAj-AVj)*jth_abs_gradW;

           if(IDi+1 == -26001 && IDj+1 == 16400)
           {
               printf("ON DEV3:  %d with %d PA: %f PB2: %f  w_ij: %f dr.x / r: %f \n",
                       (int)IDi+1, IDj+1, PA,PB2, w_ij, dr.x / r);
               printf("ON DEV4: %g %g (%g - %g)  %f \n", jM0.w,(PAj-AVj), PAj,AVj,jth_abs_gradW);
               printf("ON DEV5: %f %f %f\n", (1.0 / (jD.x * jD.x))* jH.x, jD.x, jH.x);
           }


           float PB = PA+PB2;
           //Scale by distance
           const float4 gradW2 = (r > 0) ? make_float4(PB * dr.x / r, PB * dr.y / r, PB * dr.z / r, 0.0f)
                                                           : make_float4(0.0f,0.0f,0.0f,0.0f);


            //TODO the PAi part can be moved to outside the loop
            //Energy following equation, Rosswog 2009, eq 119
            //                     (      grkerni      )
//            float du = jM0.w*PAi*projv*omegai*ith_abs_gradW;
           float du = jM0.w*PAi*projv*ith_abs_gradW;

            //Add AV to the energy, Rosswog eq 62 / Phantom eq 40
//            du += -AVi*jM0.w*projv*omegai*ith_abs_gradW;
           du += -AVi*jM0.w*projv*ith_abs_gradW;

            //Add conductivity to the energy, Phantom eq 40.

            float avg_rho = 2.0f/(density_i[0].dens + jD.x);
            float vsigu   = sqrtf(fabs(hydro_i[0].x - jH.x)*avg_rho); //Phantom Eq 41, note for gravity use:  vsigu = abs(projv)
            float autermi = aC*0.5f*jM0.w*(1.0f/density_i[0].dens); //TODO this should use the i-particle mass, but we don't have that here, w=smth
            float autermj = aC*0.5f*jM0.w *(1.0f/jD.x);
            float denij   = hydro_i[0].z - jH.z;

//            float AVC = vsigu*denij*(autermi*omegai*ith_abs_gradW + autermj*omegaj*jth_abs_gradW); //Phantom Eq 40
            float AVC = vsigu*denij*(autermi*ith_abs_gradW + autermj*jth_abs_gradW); //Phantom Eq 40

            du += AVC;



             //Older work

         //   const int IDj = __shfl(IDjx, j);


//                if(IDi == 148739 && gradW2.x != 0)
                //if(IDi == 64767 && gradW2.x != 0)
//            if(IDi == 100863 && IDj != 0)
//            if(IDi == 16577500  && IDj != 999999)
//                if(0)
                {
//                    int tempID = (IDj >= 100000000 ? IDj-100000000 : IDj);
//                    int boundary  = (IDj >= 100000000 ? 1 : 0);
//                    const float hi = 1.0f/pos_i[0].w;
//                    const float hi21 = hi*hi;
//                    const float hi41 = hi21*hi21;
//                    float cnormk = 1./(120.*M_PI);
//                    const float cnormkh2 = cnormk*hi41; //Quintic kernel
//
//                    if(fabs(gradW2.x) > 0.0f)
//                    printf("ON DEV: %d\tj: %d\tForce: %.16f %.16f %.16f\n",
//                            (int)IDi, IDj,
//                            acc_i[0].x, gradW2.x, acc_i[0].x - gradW2.x);
//


//                    if(IDj == 147470)
//                    {
//                        float cnormk = 1./(120.*M_PI);

//                        float hi2= (1.0f/pos_i[0].w)*(1.0f/pos_i[0].w);
//                        float hj2= (1.0f/jD.y)*(1.0f/jD.y);

//                        float tempi = kernel.abs_gradWJB(r, pos_i[0].w);
//                        float tempj = kernel.abs_gradWJB(r, jD.y);
//
//
//                        float tempi2 = tempi*hi2*hi2*cnormk*omegai;
//                        float tempj2 = tempj*hj2*hj2*cnormk*omegaj;
//
////                        float tempj2 = tempi*temp*temp*cnormk*omegai;
//
//                        double dti2 = tempi*hi2*hi2*cnormk*omegai;
//                        double dtj2 = tempj*hj2*hj2*cnormk*omegaj;
//
//                        printf("ON DEVX: %d\tj: %d\tForce: %.16f | %f %f | PA: %f %f | %f %f %f %.16f\n",
//                                 (int)IDi, IDj,gradW2.x,
//                                 ith_abs_gradW, jth_abs_gradW,
//                                 PA, PB2,
//                                 PAi, AVi, w_ij, jM0.w );

//                        printf("ON DEVX: %d\tj: %d\tForce: %.16f | %f %f | %f %f | %f %f | %f %f | %f %f | %f %f | %f %f | %f %f\n",
//                                (int)IDi, IDj,gradW2.x,
//                                ith_abs_gradW, jth_abs_gradW,
//                                pos_i[0].w, jD.y,
//                                density_i[0].dens, jD.x,
//                                omegai, omegaj,
//                                r, jD.y,
//                                tempi, tempj,
//                                tempi2, dti2,
//                                tempj2, dtj2);

//                    }

//                    if(AVC != 0)
//                    printf("ON DEV, %d\t%d\t%d\t\told:  %f %f\t%.16f\n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//
//                            denij, vsigu,
//                            AVC);



//                    printf("ON DEV, %d\t%d\t%d\t\told:  %f %f %f %f %f %f \n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//
//                            acc_i[0].w,
//                            du,
//                            w_ij,
//                            projv,
//                            omegai*ith_abs_gradW,
//                            acc_i[0].w + du);
//                            vel_i[0].x, vel_i[0].y, vel_i[0].z);
//                            vel_i[0].y, jvel.y, dv.y);
//                            jvel.x, jvel.y, jvel.z);
//                            dv.x, dv.y, dv.z);




//                    printf("ON DEV, %d\t%d\t%d\t\told:  %f %f %f \n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//                            PA,
//                            PB2);

//                    printf("ON DEV, %d\t%d\t%d\t\told: rhoi: %f  v_sigi: %f w_ij: %f  res: %f || %f %f %f %f %f %f %f\n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//                            1.0 / density_i[0].dens,
//                            v_sigi, w_ij, -0.5*(1.0 / density_i[0].dens)*v_sigi*w_ij,
//                            dr.x, dr.y, dr.z, dv.x, dv.y, dv.z, r);
//                    printf("ON DEV, %d\t%d\t%d\t\told: rhoi: %f  v_sigi: %f w_ij: %f  res: %f || %f %f %f %f %f %f %f\n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//                            1.0 / jD.x,
//                            v_sigj, w_ij, -0.5*(1.0 / jD.x)*v_sigj*w_ij,
//                            dr.x, dr.y, dr.z, dv.x, dv.y, dv.z, r);



//                    printf("ON DEV, %d\t%d\t%d\t\told: %f\tnew: %f\t diff: %f | sigij: %f %f | P: %f rho2i %f AV: %f %f | kern: %f\t%f | rho: %f %f | wij: %f cs: %f\n",
//                            (int)IDi+1,
//                            boundary, // IDj+1,
//                            tempID+1,
//                            acc_i[0].x,
//                            acc_i[0].x - gradW2.x,
//                            gradW2.x, //diff
//                            PB,
//                            v_sigj,
//
//                            hydro_i[0].x,
//                            (1.0 / (density_i[0].dens* density_i[0].dens)),
//
//                            AVi,
//                            AVj,
//
//                            -jM0.w*PA,
//                            -jM0.w*PB2,
//                            omegai*ith_abs_gradW,
//                            omegaj*jth_abs_gradW,
//                            density_i[0].dens,
//                            jD.x,
//                            w_ij,
//                            hydro_i[0].y);


                const int IDj = __shfl(IDjx, j);
                if(IDi == -26000 && gradW2.x != 0 && IDj >= 0)
                {
                    printf("ON DEV %d with %d  gives: %f \n", (int)IDi+1, IDj+1, gradW2.x);
                }

                } //ID



          if(jD.x != 0) {
            acc_i[0].x    -= gradW2.x;
            acc_i[0].y    -= gradW2.y;
            acc_i[0].z    -= gradW2.z;
            acc_i[0].w    += du;
#ifdef STATS
            gradient_i[0].y += (fabs(abs_gradW)) > 0; //Number of useful operations
          }
          gradient_i[0].z++; //Number of times we enter this function
#else
          }
#endif



          }//for WARP_SIZE

          gradient_i[0].x = max(v_sig_max, gradient_i[0].x);    //This is for the dt parameter
          //force[id].dt = PARAM::C_CFL * 2.0 * ith.smth / v_sig_max;
        }
#endif

    };
}; //namespace hydroforce








} //namespace SPH

#endif
