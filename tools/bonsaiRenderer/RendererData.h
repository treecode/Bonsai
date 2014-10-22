#pragma once

#include <mpi.h>
#include <cassert>
#include <cmath>
#include <array>
#include <parallel/algorithm>
#include <iostream>
#include "IDType.h"
#include "CameraPath.h"

#if 0
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x,y,z,w; };
struct int2   { int x, y; };
inline static int2 make_int2(int _x, int _y) 
{ 
  int2 v; v.x = _x; v.y = _y;; return v;
}
inline static float2 make_float2(float _x, float _y) 
{ 
  float2 v; v.x = _x; v.y = _y;; return v;
}
inline static float3 make_float3(float x, float y, float z)
{
  float3 v; v.x = x; v.y = y; v.z=z; return v;
}
inline static float4 make_float4(float x, float y, float z, float w)
{
  float4 v; v.x = x; v.y = y; v.z=z; v.w=w; return v;
}
#endif

#include <algorithm>

class RendererData
{
  public:
    typedef unsigned long long long_t;
    enum Attribute_t {
      MASS,
      VEL,
      RHO,
      H,
      NPROP};
  protected:
    const int rank, nrank;
    const MPI_Comm &comm;
    struct particle_t
    {
      float posx, posy, posz;
      IDType ID;
      float attribute[NPROP];
    };
    std::vector<particle_t> data;
    bool new_data;

    float _xmin, _ymin, _zmin, _rmin;
    float _xmax, _ymax, _zmax, _rmax;

    float _xminl, _yminl, _zminl, _rminl;
    float _xmaxl, _ymaxl, _zmaxl, _rmaxl;

    float _attributeMin[NPROP];
    float _attributeMax[NPROP];
    float _attributeMinL[NPROP];
    float _attributeMaxL[NPROP];
  
    void minmaxAttributeGlb(const Attribute_t p);
    int  getMaster() const { return 0; }
    bool isMaster() const { return getMaster() == rank; }
    
    const CameraPath *cameraPtr;

    bool firstData;
    double time;
    size_t nBodySim;

  public:
    RendererData(const int rank, const int nrank, const MPI_Comm &comm) : 
      rank(rank), nrank(nrank), comm(comm), cameraPtr(nullptr), firstData(true)
  {
    assert(rank < nrank);
    new_data = false;
  }

    double getTime() const { return time; }
    void setTime(const double time) { this->time = time; }
    void setNbodySim(const size_t n) { nBodySim = n; }
    size_t getNbodySim() const { return nBodySim; }

    void setCameraPath(const CameraPath *ptr) { cameraPtr = ptr; }
    bool isCameraPath() const { return cameraPtr != nullptr; }
    const CameraPath& getCamera() const {return *cameraPtr;}

    void setNewData() {new_data = true;}
    bool unsetNewData() { 
      new_data = false; 
      const bool ret = firstData;
      firstData = false;
      return ret;
    }
    bool isNewData() const {return new_data;}

    int n() const { return data.size(); }
    int size() const { return data.size(); }
    void resize(const int n) { data.resize(n); }
    ~RendererData() {}

    float  posx(const int i) const { return data[i].posx; }
    float& posx(const int i)       { return data[i].posx; }
    float  posy(const int i) const { return data[i].posy; }
    float& posy(const int i)       { return data[i].posy; }
    float  posz(const int i) const { return data[i].posz; }
    float& posz(const int i)       { return data[i].posz; }

    float  attribute(const Attribute_t p, const int i) const {return data[i].attribute[p]; }
    float& attribute(const Attribute_t p, const int i)       {return data[i].attribute[p]; }


    IDType  ID(const long_t i) const { return data[i].ID; }
    IDType& ID(const long_t i)       { return data[i].ID; }

    float xmin() const { return _xmin;} 
    float ymin() const { return _ymin;} 
    float zmin() const { return _zmin;} 
    float rmin() const { return _rmin;} 

    float xmax() const { return _xmax;} 
    float ymax() const { return _ymax;} 
    float zmax() const { return _zmax;} 
    float rmax() const { return _rmax;} 

    float attributeMin(const Attribute_t p) const { return _attributeMin[p]; }
    float attributeMax(const Attribute_t p) const { return _attributeMax[p]; }

    void setAttributeMin(const Attribute_t p, const float val) { _attributeMin[p] = val; }
    void setAttributeMax(const Attribute_t p, const float val) { _attributeMax[p] = val; }

    float xminLoc() const { return _xminl;} 
    float yminLoc() const { return _yminl;} 
    float zminLoc() const { return _zminl;} 
    float rminLoc() const { return _rminl;} 

    float xmaxLoc() const { return _xmaxl;} 
    float ymaxLoc() const { return _ymaxl;} 
    float zmaxLoc() const { return _zmaxl;} 
    float rmaxLoc() const { return _rmaxl;} 

    float attributeMinLoc(const Attribute_t p) const { return _attributeMinL[p]; }
    float attributeMaxLoc(const Attribute_t p) const { return _attributeMaxL[p]; }
    
    void randomShuffle();
    void computeMinMax();

    template<typename Func> void rescale(const Attribute_t p, const Func &scale);
    void rescaleLinear(const Attribute_t p, const float newMin, const float newMax);
    void scaleLog(const Attribute_t p, const float zeroPoint = 1.0f);
    void scaleExp(const Attribute_t p, const float zeroPoint = 1.0f);
    void clampMinMax(const Attribute_t p, const float min, const float max);

    // virtual methods

    virtual bool  isDistributed() const { return false; }
    virtual void  setNMAXSAMPLE(const int n) {};
    virtual void  distribute() {}
    virtual float getBoundBoxLow (const int i) const 
    {
      switch (i)
      {
        case  0: return xmin(); 
        case  1: return ymin();
        case  2: return zmin();
        default: return rmin();
      }
    }
    virtual float getBoundBoxHigh(const int i) const 
    {
      switch (i)
      {
        case  0: return xmax(); 
        case  1: return ymax();
        case  2: return zmax();
        default: return rmax();
      }
    }
    virtual std::vector<int> getVisibilityOrder(const std::array<float,3> camPos) const
    {
      return std::vector<int>();
    }

};

class RendererDataDistribute : public RendererData
{
  private:
    enum { NMAXPROC   = 1024};
    int NMAXSAMPLE;
    int sample_freq;

    float xlow[3], xhigh[3];
    int npx, npy, npz;
    bool distributed;

    using vector3 = std::array<double,3>;
    struct float4
    {
      typedef float  v4sf __attribute__ ((vector_size(16)));
      typedef double v2df __attribute__ ((vector_size(16)));
      static v4sf v4sf_abs(v4sf x){
        typedef int v4si __attribute__ ((vector_size(16)));
        v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        return __builtin_ia32_andps(x, (v4sf)mask);
      }
      union{
        v4sf v;
        struct{
          float x, y, z, w;
        };
      };
      float4() : v((v4sf){0.f, 0.f, 0.f, 0.f}) {}
      float4(float x, float y, float z, float w) : v((v4sf){x, y, z, w}) {}
      float4(float x) : v((v4sf){x, x, x, x}) {}
      float4(v4sf _v) : v(_v) {}
      float4 abs(){
        typedef int v4si __attribute__ ((vector_size(16)));
        v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
        return float4(__builtin_ia32_andps(v, (v4sf)mask));
      }
      void dump(){
        std::cerr << x << " "
          << y << " "
          << z << " "
          << w << std::endl;
      }
#if 1
      v4sf operator=(const float4 &rhs){
        v = rhs.v;
        return v;
      }
      float4(const float4 &rhs){
        v = rhs.v;
      }
#endif
#if 0
      const v4sf_stream operator=(const v4sf_stream &s){
        __builtin_ia32_movntps((float *)&v, s.v);
        return s;
      }
      float4(const v4sf_stream &s){
        __builtin_ia32_movntps((float *)&v, s.v);
      }
#endif
    };


    struct Boundary
    {
      double xlow, xhigh;
      double ylow, yhigh;
      double zlow, zhigh;
      Boundary(const vector3 &low, const vector3 &high){
        xlow = low[0]; xhigh = high[0];
        ylow = low[1]; yhigh = high[1];
        zlow = low[2]; zhigh = high[2];
      }
      bool isinbox(const vector3 &pos) const {
        return !(
            (pos[0] < xlow ) || 
            (pos[1] < ylow ) || 
            (pos[2] < zlow ) || 
            (pos[0] > xhigh) || 
            (pos[1] > yhigh) || 
            (pos[2] > zhigh) );
      }
    };

  public:

    RendererDataDistribute(const int rank, const int nrank, const MPI_Comm &comm) : 
      RendererData(rank,nrank,comm), NMAXSAMPLE(200000), distributed(false)
  {
    assert(nrank <= NMAXPROC);
  }

    virtual void setNMAXSAMPLE(const int n) {NMAXSAMPLE = n;}
    virtual bool isDistributed() const { return distributed; }

  private:

    void create_division();
    int determine_sample_freq();
    void initialize_division();
    void collect_sample_particles(std::vector<vector3> &sample_array, const int sample_freq);

    void determine_division( // nitadori's version
        std::vector<float4>  &pos,
        const float rmax,
        vector3  xlow[],  // left-bottom coordinate of divisions
        vector3 xhigh[]);  // size of divisions

    inline int which_box(
        const vector3 &pos,
        const vector3 xlow[],
        const vector3 xhigh[]);

    inline void which_boxes(
        const vector3 &pos,
        const float h,
        const vector3 xlow[],
        const vector3 xhigh[],
        std::vector<int> &boxes);

    void alltoallv(std::vector<particle_t> psend[], std::vector<particle_t> precv[]);


    void exchange_particles_alltoall_vector(
        const vector3  xlow[],
        const vector3 xhigh[]);


    /////////////////////
    //
  public:

    virtual void distribute();
    virtual float getBoundBoxLow (const int i) const {return  xlow[i];}
    virtual float getBoundBoxHigh(const int i) const {return xhigh[i];}
    virtual std::vector<int> getVisibilityOrder(const std::array<float,3> camPos) const;
};
