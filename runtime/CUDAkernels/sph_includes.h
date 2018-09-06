#ifndef SPH_INCLUDE_H
#define SPH_INCLUDE_H



#ifdef WIN32
    #define M_PI        3.14159265358979323846264338328
#endif


//#define USE_BALSARA_SWITCH



//#define STATS


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

        /* this one works with SPH_KERNEL_SIZE 3.0f and PARAM_SMTH 1.0 */
        //Quintic kernel as defined in Phantom

        __device__ __forceinline__ float abs_gradW(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi41 = hi21*hi21;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = cnormk*hi41;

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
        __device__ __forceinline__ float abs_gradW2(const float q, float &w) const{
            const float q2 = q * q;

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
    struct kernel_t : public base_kernel {

        static constexpr float cnormk = 1./M_PI;

#if 0
        __device__ __forceinline__ float W(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi31 = hi*hi21;
            const float q  = dr*hi;
            const float q2 = (dr*dr)*hi21;
            const float cnormkh = cnormk*hi31;

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
#endif

        __device__ __forceinline__ float abs_gradW(const float dr, const float h) const{
            const float hi = 1.0f/h;
            const float hi21 = hi*hi;
            const float hi41 = hi21*hi21;
            const float q  = dr*hi;

            const float cnormkh = cnormk*hi41;

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
                return 0.0f;
            }
        }

        //Combined kernel, does not multiply with hi3 and/or hi4 nor with cnorm. This is done as final
        //step in the dev_sph code.
        __device__ __forceinline__ float abs_gradW2(const float q, float &w) const{
            if(q < 1.0f)
            {
                w = (0.75f*q*q*q - 1.5f*q*q + 1.0f);
                return q*(2.25f*q - 3.0f);
            }
            else if(q < 2.0f)
            {
                w = (-0.25f*pow3(q-2.0f));
                return  -0.75f*((q-2.0f)*(q-2.0f));
            }
            else
            {
                w = 0.0f;
                return 0.0f;
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
         __device__ __forceinline__ float abs_gradW2(const float q, float &w) const{
             if(q >= 2.0)
             {
                 w = 0.0f;
                 return 0;
             }
             const float q2 = q*q;
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

enum SPHVal {DENSITY, HYDROFORCE};

#if 1
namespace density
{
    template<int NI, bool FULL>
    struct directOperator {

        static const SPHVal type = SPH::DENSITY;


        __device__ __forceinline__ void operator()(
                float4  acc_i[NI],
          const float4  pos_i[NI],
          const float4  vel_i[NI],
          const int     ptclIdx,
          const sphParameters     SPHParams,
          SPH::density::data  density_i[NI],
          SPH::derivative::data gradient_i[NI],
          const float4  hydro_i[NI],  //Not used here
          const float4 *body_jpos,
          const float4 *body_jvel,
          const float2 *body_jdens,   //Not used here
          const float4 *body_hydro)   //Not used here
        {
            SPH::kernel_t kernel;

            const float4 M0 = (FULL || ptclIdx >= 0) ? body_jpos[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);


            const int NGROUPTemp = NCRIT;
            const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
            for (int j = offset; j < offset+NGROUPTemp; j++)
            {
              const float4 jM0   = make_float4(__shfl_sync(FULL_MASK, M0.x, j), __shfl_sync(FULL_MASK, M0.y, j),
                                               __shfl_sync(FULL_MASK, M0.z, j), __shfl_sync(FULL_MASK, M0.w,j));
              const float  jmass = jM0.w;
              const float3 jpos  = make_float3(jM0.x, jM0.y, jM0.z);

              const float hi=1.0f/pos_i[0].w;

          #pragma unroll
              for (int k = 0; k < NI; k++)
              {
                const float3 dr    = make_float3(jpos.x - pos_i[k].x, jpos.y - pos_i[k].y, jpos.z - pos_i[k].z);
                const float r2     = dr.x*dr.x + dr.y*dr.y + dr.z*dr.z;
                const float r      = sqrtf(r2);
                const float qi     = r/pos_i[k].w;

                float temp1, temp2;
                temp2 = kernel.abs_gradW2(r*hi, temp1); //temp1 is density kernel, temp2 = derivative kernel
                //temp2 = kernel.abs_gradW2(r, pos_i[k].w, temp1); //temp1 is density kernel, temp2 = derivative kernel

                density_i[k].dens  += jmass*temp1;                   //Density
                acc_i[k].x         += jmass*(-qi*temp2 - 3*temp1);   //Derivative

#ifdef USE_BALSARA_SWITCH
                //Balsara switch, TODO(jbedorf): In theory this is only needed when we perform the final density iteration
                const float3 jvel  = make_float3(__shfl_sync(FULL_MASK, MV.x, j), __shfl_sync(FULL_MASK, MV.y, j), __shfl_sync(FULL_MASK, MV.z, j));
                const float3 dv    = make_float3(jvel.x - vel_i[0].x, jvel.y - vel_i[0].y, jvel.z - vel_i[0].z);

                temp2 /= r;
                const float3 gradW = (r > 0.0f) ? make_float3(temp2 * dr.x, temp2 * dr.y, temp2 * dr.z) : (float3){0.0f, 0.0f, 0.0f};

                gradient_i[0].x -= jmass * (dv.y * gradW.z - dv.z * gradW.y);
                gradient_i[0].w -= jmass * (dv.x * gradW.x + dv.y * gradW.y + dv.z * gradW.z);
                //Note y and z below within the ifdef statements
#ifdef STATS
                //For interaction stats
                gradient_i[0].z++;       //Number of operations
                gradient_i[0].y += fabs(jmass * temp1) > 0; //Number of useful operations
#else
                gradient_i[0].y -= jmass * (dv.z * gradW.x - dv.x * gradW.z);
                gradient_i[0].z -= jmass * (dv.x * gradW.y - dv.y * gradW.x);
#endif //STATS
#endif //USE_BALSARA_SWITCH

              } //for k
            } //for offset
        } //end operator()
    }; // struct directOperator
}; //namespace density
#endif



namespace hydroforce
{
//real4> bodies_hydro;   //The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch

    template<int NI, bool FULL>
    struct directOperator {

         static const SPHVal type = SPH::HYDROFORCE;


        __device__ __forceinline__ void operator()(
                  float4  acc_i[NI],
            const float4  pos_i[NI],
            const float4  vel_i[NI],
            const int     ptclIdx,
            const sphParameters     SPHParams,
            SPH::density::data    density_i[NI],
            SPH::derivative::data gradient_i[NI],
            const float4  hydro_i[NI],              //x = pressure, y = soundspeed, z = Energy , w = Balsala Switch
            const float4 *body_jpos,
            const float4 *body_jvel,
            const float2 *body_jdens,
            const float4 *body_hydro)
        {
          SPH::kernel_t kernel;
          float v_sig_max = 0.0f; //Over various cells it is tracked via gradient_i x

          const float omegai = vel_i[0].w;

          const float densii =  1.0f / density_i[0].dens;
          const float PAi    =  densii*densii*hydro_i[0].x; //Can be computed outside this function


          const float4 MP = (FULL || ptclIdx >= 0) ? body_jpos [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float4 MV = (FULL || ptclIdx >= 0) ? body_jvel [ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 MH = (FULL || ptclIdx >= 0) ? body_hydro[ptclIdx] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
          const float2 MD = (FULL || ptclIdx >= 0) ? body_jdens[ptclIdx] : make_float2(0.0f, 0.0f);

          MH.y         *= SPHParams.av_alpha;
          const float tmp_hydro_y = hydro_i[0].y*SPHParams.av_alpha;

          const int NGROUPTemp = NCRIT;
          const int offset     = NGROUPTemp*(laneId / NGROUPTemp);
          for (int j = offset; j < offset+NGROUPTemp; j++)
          {
            const float4 jM0   = make_float4(__shfl_sync(FULL_MASK, MP.x, j), __shfl_sync(FULL_MASK, MP.y, j),
                                             __shfl_sync(FULL_MASK, MP.z, j), __shfl_sync(FULL_MASK, MP.w, j));
            const float3 dr    = make_float3(pos_i[0].x - jM0.x, pos_i[0].y - jM0.y, pos_i[0].z - jM0.z);

            const float3 jvel  = make_float3(__shfl_sync(FULL_MASK, MV.x, j), __shfl_sync(FULL_MASK, MV.y, j), __shfl_sync(FULL_MASK, MV.z, j));
            const float3 dv    = make_float3(vel_i[0].x - jvel.x, vel_i[0].y - jvel.y, vel_i[0].z - jvel.z);

            const float r        = sqrtf(dr.x*dr.x + dr.y*dr.y + dr.z*dr.z);
            const float xv_inner = dr.x * dv.x + dr.y * dv.y + dr.z * dv.z;
            const float w_ij     = (xv_inner < 0.0f) ? xv_inner / r : 0.0f;
            const float projv    = (r != 0) ? xv_inner / r : 0.0f;


            //Artificial viscosity
            const float4 jH       = make_float4(__shfl_sync(FULL_MASK, MH.x, j), __shfl_sync(FULL_MASK, MH.y, j),
                                                __shfl_sync(FULL_MASK, MH.z, j), __shfl_sync(FULL_MASK, MH.w, j));

            const float v_sigi     = tmp_hydro_y  - SPHParams.av_beta * projv;
            const float v_sigj     = jH.y         - SPHParams.av_beta * projv;


            //Determine the maximum for the dt variable
            v_sig_max         = max(v_sig_max, max(v_sigi,v_sigj));


            float PA = 0, PB2 = 0, AVi = 0, AVj = 0;

            const float2 jD    = make_float2(__shfl_sync(FULL_MASK, MD.x, j), __shfl_sync(FULL_MASK, MD.y, j));
            const float densji = (jM0.w != 0) ? 1.0f / jD.x : 0;


            if(w_ij < 0)
            {
               //Artificial viscosity switches, note that we do this per particle
               AVi = 0.5f*densii*v_sigi*projv;
               AVj = 0.5f*densji*v_sigj*projv;

#ifdef USE_BALSARA_SWITCH
               //Multiply with the Balsara switch
               AVi *= hydro_i[0].w;
               AVj *= jH.w;
#endif
            }

            float ith_abs_gradW = kernel.abs_gradW(r, pos_i[0].w);
            ith_abs_gradW *= omegai;
            float jth_abs_gradW = (jD.y != 0) ? kernel.abs_gradW(r, jD.y) : 0.0f; //Returns NaN if smthj is 0 which happens if we do not use the particle, so 'if' for now
            jth_abs_gradW *= __shfl_sync(FULL_MASK, MV.w, j); //multiply with omegaj


            //Force according to equation 120 , with the addition of the Artificial Viscosity
            float PAj  =  (densji*densji)* jH.x ;

            PA   = jM0.w*(PAi-AVi)*ith_abs_gradW;
            PB2  = jM0.w*(PAj-AVj)*jth_abs_gradW;


            float PB = (PA+PB2) / r; //Combine and scale by distance
            const float3 gradW2 = (r > 0) ? make_float3(PB * dr.x, PB * dr.y, PB * dr.z) : make_float3(0.0f,0.0f,0.0f);


            //Energy following equation, Rosswog 2009, eq 119
            float du = jM0.w*PAi*projv*ith_abs_gradW;

            //Add AV to the energy, Rosswog eq 62 / Phantom eq 40
            du += -AVi*jM0.w*projv*ith_abs_gradW;

            //Add conductivity to the energy, Phantom eq 40.
            float avg_rho = 2.0f/(density_i[0].dens + jD.x);
            float vsigu   = sqrtf(fabs(hydro_i[0].x - jH.x)*avg_rho); //Phantom Eq 41, note for gravity use:  vsigu = abs(projv)

            float autermi = SPHParams.ac_param*0.5f*jM0.w*densii; //TODO this should use the i-particle mass, but we don't have that here, w=smth
            float autermj = SPHParams.ac_param*0.5f*jM0.w*densji;
            float denij   = hydro_i[0].z - jH.z;
            du += vsigu*denij*(autermi*ith_abs_gradW + autermj*jth_abs_gradW); //Phantom Eq 40

          if(jD.x != 0) {
            acc_i[0].x    -= gradW2.x;
            acc_i[0].y    -= gradW2.y;
            acc_i[0].z    -= gradW2.z;
            acc_i[0].w    += du;
#ifdef STATS
            gradient_i[0].y += (fabs(ith_abs_gradW) + fabs(jth_abs_gradW)) > 0; //Number of useful operations
          }
          gradient_i[0].z++; //Number of times we enter this function
#else
          }
#endif

          }//for WARP_SIZE

          gradient_i[0].x = max(v_sig_max, gradient_i[0].x); //This is for the dt parameter
        } //operator()
    }; //Struct
}; //namespace hydroforce


} //namespace SPH

#endif
