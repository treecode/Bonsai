#pragma once

#include <mpi.h>
#include <cassert>
#include <cmath>
#include <array>
#include <parallel/algorithm>
#include <iostream>

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
      long_t ID;
      int type;
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
  
    float xlow[3], xhigh[3];
    int npx, npy, npz;
   
    void minmaxAttributeGlb(const Attribute_t p)   
    {
      MPI_Allreduce(&_attributeMinL[p], &_attributeMin[p], 1, MPI_FLOAT, MPI_MIN, comm);
      MPI_Allreduce(&_attributeMaxL[p], &_attributeMax[p], 1, MPI_FLOAT, MPI_MAX, comm);
    }

    int  getMaster() const { return 0; }
    bool isMaster() const { return getMaster() == rank; }


  public:
    RendererData(const int rank, const int nrank, const MPI_Comm &comm) : 
      rank(rank), nrank(nrank), comm(comm), distributed(false)
  {
    assert(rank < nrank);
    new_data = false;
  }

    void setNewData() {new_data = true;}
    void unsetNewData() { new_data = false; }
    bool isNewData() const {return new_data;}

    std::array<int,3> getRankFactor() const
    {
      return {{npx,npy,npz}};
    }

    void resize(const int n)
    {
      data.resize(n);
    }

    ~RendererData()
    {
    }

    float getBoundBoxLow (const int i) const {return  xlow[i];}
    float getBoundBoxHigh(const int i) const {return xhigh[i];}

    int n() const { return data.size(); }

    float  posx(const int i) const { return data[i].posx; }
    float& posx(const int i)       { return data[i].posx; }
    float  posy(const int i) const { return data[i].posy; }
    float& posy(const int i)       { return data[i].posy; }
    float  posz(const int i) const { return data[i].posz; }
    float& posz(const int i)       { return data[i].posz; }

    float  attribute(const Attribute_t p, const int i) const {return data[i].attribute[p]; }
    float& attribute(const Attribute_t p, const int i)       {return data[i].attribute[p]; }

    int  type(const int i) const { return data[i].type; }
    int& type(const int i)       { return data[i].type; }

    long_t  ID(const long_t i) const { return data[i].ID; }
    long_t& ID(const long_t i)       { return data[i].ID; }
   
    void randomShuffle()   
    {
      std::random_shuffle(data.begin(), data.end());
    }

    void computeMinMax()
    {
      _xminl=_yminl=_zminl=_rminl = +HUGE;
      _xmaxl=_ymaxl=_zmaxl=_rmaxl = -HUGE;
      for (int p = 0; p < NPROP; p++)
      {
        _attributeMinL[p] = +HUGE;
        _attributeMaxL[p] = -HUGE;
      }

      const int _n = data.size();
      for (int i = 0; i < _n; i++)
      {
        _xminl = std::min(_xminl, posx(i));
        _yminl = std::min(_yminl, posy(i));
        _zminl = std::min(_zminl, posz(i));
        _xmaxl = std::max(_xmaxl, posx(i));
        _ymaxl = std::max(_ymaxl, posy(i));
        _zmaxl = std::max(_zmaxl, posz(i));
        for (int p = 0; p < NPROP; p++)
        {
          _attributeMinL[p] = std::min(_attributeMinL[p], attribute(static_cast<Attribute_t>(p),i));
          _attributeMaxL[p] = std::max(_attributeMaxL[p], attribute(static_cast<Attribute_t>(p),i));
        }
      }
      _rminl = std::min(_rminl, _xminl);
      _rminl = std::min(_rminl, _yminl);
      _rminl = std::min(_rminl, _zminl);
      _rmaxl = std::max(_rmaxl, _xmaxl);
      _rmaxl = std::max(_rmaxl, _ymaxl);
      _rmaxl = std::max(_rmaxl, _zmaxl);

      for (int i = 0; i < _n; i++)
      {
        assert(posx(i) >= _xminl && posx(i) <= _xmaxl);
        assert(posy(i) >= _yminl && posy(i) <= _ymaxl);
        assert(posz(i) >= _zminl && posz(i) <= _zmaxl);
        assert(posx(i) >= _rminl && posx(i) <= _rmaxl);
        assert(posy(i) >= _rminl && posy(i) <= _rmaxl);
        assert(posz(i) >= _rminl && posz(i) <= _rmaxl);
      }


      float minloc[] = {_xminl, _yminl, _zminl, _rminl};
      float minglb[] = {_xminl, _yminl, _zminl, _rminl};

      float maxloc[] = {_xmaxl, _ymaxl, _zmaxl, _rmaxl};
      float maxglb[] = {_xmaxl, _ymaxl, _zmaxl, _rmaxl};

      MPI_Allreduce(minloc, minglb, 4, MPI_FLOAT, MPI_MIN, comm);
      MPI_Allreduce(maxloc, maxglb, 4, MPI_FLOAT, MPI_MAX, comm);

      _xmin = minglb[0];
      _ymin = minglb[1];
      _zmin = minglb[2];
      _rmin = minglb[3];
      _xmax = maxglb[0];
      _ymax = maxglb[1];
      _zmax = maxglb[2];
      _rmax = maxglb[3];
      
      for (int p = 0; p < NPROP; p++)
        minmaxAttributeGlb(static_cast<Attribute_t>(p));
    }

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

    template<typename Func>
      void rescale(const Attribute_t p, const Func &scale)
      {
        float min = +HUGE, max = -HUGE;
        const int _n = data.size();
        for (int i = 0; i < _n; i++)
        {
          attribute(p,i) = scale(attribute(p,i));
          min = std::min(min, attribute(p,i));
          max = std::max(max, attribute(p,i));
        }

        _attributeMinL[p] = min;
        _attributeMaxL[p] = max;

#if 0
        minmaxAttributeGlb(p);
#else
        _attributeMin[p] = scale(attributeMin(p));
        _attributeMax[p] = scale(attributeMax(p));
#endif
      }


    void rescaleLinear(const Attribute_t p, const float newMin, const float newMax)
    {
      const float oldMin = attributeMin(p);
      const float oldMax = attributeMax(p);
     
      const float oldRange = oldMax - oldMin ;
      assert(oldRange != 0.0);
      
      const float slope = (newMax - newMin)/oldRange;
      rescale(p,[&](const float x) { return slope * (x - oldMin) + newMin;});

    }

    void scaleLog(const Attribute_t p, const float zeroPoint = 1.0f)
    {
      rescale(p, [&](const float x) {return std::log(x + zeroPoint);});
    }
    void scaleExp(const Attribute_t p, const float zeroPoint = 1.0f)
    {
      rescale(p,[&](const float x) {return std::exp(x) - zeroPoint;});
    }

#if 0
    void clamp(const Attribute_t p, const float left, const float right)
    {
      assert(left  >= 0.0f && left  < 0.5f);
      assert(right >= 0.0f && right < 0.5f);

      const float oldMin = attributeMin(p);
      const float oldMax = attributeMax(p);
      const float oldRange = oldMax - oldMin ;
      assert(oldRange > 0.0f);

      const float valMin = oldMin + left *oldRange;
      const float valMax = oldMax - right*oldRange;
      assert(valMin < valMax);

      rescale(p,[&](const float x) {return std::max(valMin, std::min(valMax,x));});
    }
#endif
    void clampMinMax(const Attribute_t p, const float min, const float max)
    {
      rescale(p,[&](const float x) { return std::max(min, std::min(max, x)); });

#if 1
      _attributeMin[p] = min;
      _attributeMax[p] = max;
#endif
    }


    // virtual methods
   

    bool distributed;
    bool isDistributed() const { return distributed; }
    virtual void setNMAXSAMPLE(const int n) {};
    virtual void distribute() {}

};

class RendererDataDistribute : public RendererData
{
  private:
    enum { NMAXPROC   = 1024};
    int NMAXSAMPLE;
    int sample_freq;

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
      RendererData(rank,nrank,comm), NMAXSAMPLE(200000)
  {
    assert(nrank <= NMAXPROC);
  }

    void setNMAXSAMPLE(const int n) {NMAXSAMPLE = n;}
    bool isDistributed() const { return true; }

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

    void distribute();
};
