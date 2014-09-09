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
  
    bool  distributed; 
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
    bool isDistributed() const { return distributed; }

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
      void rescale(const Attribute_t p, const Func &scale, const bool preserveGlobal = true)
      {
        float min = +HUGE, max = -HUGE;
        const int _n = data.size();
        for (int i = 0; i < _n; i++)
        {
          attribute(p,i) = scale(attribute(p,i));
          min = std::min(min, attribute(p,i));
          max = std::max(max, attribute(p,i));
        }

        if (preserveGlobal)
        {
          min = scale(attributeMin(p));
          max = scale(attributeMax(p));
        }
        _attributeMinL[p] = min;
        _attributeMaxL[p] = max;

        minmaxAttributeGlb(p);
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

      rescale(p,[&](const float x) {return std::max(valMin, std::min(valMax,x));}, false);
    }
    void clampMinMax(const Attribute_t p, const float min, const float max)
    {
      rescale(p,[&](const float x) { return std::max(min, std::min(max, x)); });

      _attributeMinL[p] = min;
      _attributeMaxL[p] = max;
      
      minmaxAttributeGlb(p);
    }
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


    template <int mask> struct CmpFloat4{
      bool operator()(const float4 &lhs, const float4 &rhs){
        return 
          mask & __builtin_ia32_movmskps(
              (float4::v4sf)__builtin_ia32_cmpltps(lhs.v, rhs.v));
      }
    };

  public:

    RendererDataDistribute(const int rank, const int nrank, const MPI_Comm &comm) : 
      RendererData(rank,nrank,comm), NMAXSAMPLE(200000)
  {
    assert(nrank <= NMAXPROC);
  }

    void setNMAXSAMPLE(const int n) {NMAXSAMPLE = n;}

  private:

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

    void create_division()
    { 
      int &nx = npx;
      int &ny = npy;
      int &nz = npz;
      ////////
      const int n = nrank;
      int n0, n1; 
      n0 = (int)pow(n+0.1,0.33333333333333333333);
      while(n%n0)n0--;
      if (isMaster())
        fprintf(stderr, "n= %d  n0= %d \n", n, n0);
      nx = n0;
      n1 = n/nx;
      n0 = (int)sqrt(n1+0.1);
      while(n1%n0)n0++;
      if(isMaster()){
        fprintf(stderr, "n1= %d  n0= %d \n", n1, n0);
      }
      ny = n0; nz = n1/n0;
      // int ntmp;
      if (nz > ny){
        // ntmp = nz; nz = ny; ny = ntmp;
        std::swap(ny, nz);
      }
      if (ny > nx){
        // ntmp = nx; nx = ny; ny = ntmp;
        std::swap(nx, ny);
      }
      if (nz > ny){
        // ntmp = nz; nz = ny; ny = ntmp;
        std::swap(ny, nz);
      }
      if (nx*ny*nz != n){
        std::cerr << "create_division: Intenal Error " << n << " " << nx
          << " " << ny << " " << nz <<std::endl;
        MPI_Abort(comm, 1);
      }
      if(isMaster()) {
        fprintf(stderr, "[nx, ny, nz] = %d %d %d\n", nx, ny, nz);
      }
    }

    int determine_sample_freq()
    {
      const int _n = data.size();
      const int nbody = _n;
#if 0
      int nreal = nbody;
      MPI_int_sum(nreal);
      int maxsample = (int)(NMAXSAMPLE*0.8); // 0.8 is safety factor
      int sample_freq = (nreal+maxsample-1)/maxsample;
#else
      double nglb = nbody, nloc = nbody;
      MPI_Allreduce(&nloc, &nglb, 1, MPI_DOUBLE, MPI_SUM, comm);
      // double maxsample = (NMAXSAMPLE*0.8); // 0.8 is safety factor
      double maxsample = (NMAXSAMPLE*0.8); // 0.8 is safety factor
      int sample_freq = int((nglb+maxsample)/maxsample);
#endif
      MPI_Bcast(&sample_freq,1,MPI_INT,getMaster(),comm);
      return sample_freq;
    }

    void initialize_division()
    {
      static bool initcall = true;
      if(initcall)
      {
        sample_freq = determine_sample_freq();
        create_division();
        initcall = false;
      }
    }

    void collect_sample_particles(std::vector<vector3> &sample_array, const int sample_freq)
    {
      const int _n = data.size();
      const int nbody = _n;
      sample_array.clear();
      for(int i=0,  ii=0; ii<nbody; i++, ii+=sample_freq)
        sample_array.push_back(vector3{{posx(i), posy(i), posz(i)}});

      /* gather sample coords */
      int nsample = sample_array.size();
      if (!isMaster())
      {
        MPI_Send(&nsample,         1,         MPI_INT,    getMaster(), rank*2,   comm);
        MPI_Send(&sample_array[0], 3*nsample, MPI_DOUBLE, getMaster(), rank*2+1, comm);
      }
      else
      {
        MPI_Status status;
        for (int p = 0; p < nrank; p++)
          if (p != getMaster())
          {
            int nrecv;
            MPI_Recv(&nrecv, 1, MPI_INT, p, p*2, comm, &status);
            sample_array.resize(nsample+nrecv);
            MPI_Recv(&sample_array[nsample], 3*nrecv, MPI_DOUBLE, p, p*2+1, comm, &status);
            nsample += nrecv;
          }

      }
    }

    void determine_division( // nitadori's version
        std::vector<float4>  &pos,
        const float rmax,
        vector3  xlow[],  // left-bottom coordinate of divisions
        vector3 xhigh[])  // size of divisions
    {
      const int nx = npx;
      const int ny = npy;
      const int nz = npz;
      const int np = pos.size();

      struct Address{
        const int nx, ny, nz;
        std::vector<int> offset;

        Address(int _nx, int _ny, int _nz, int np) :
          nx(_nx), ny(_ny), nz(_nz), offset(1 + nx*ny*nz)
        {
          const int n = nx*ny*nz;
          for(int i=0; i<=n; i++){
            offset[i] = (i*np)/n;
          }
        }
        int idx(int ix, int iy, int iz){
          return ix + nx*(iy + ny*(iz));
        }
        int xdi(int ix, int iy, int iz){
          return iz + nz*(iy + ny*(ix));
        }
        int off(int ix, int iy, int iz){
          return offset[xdi(ix, iy, iz)];
        }
      };

      const int n = nx*ny*nz;
      assert(n <= NMAXPROC);

      Address addr(nx, ny, nz, np);


      double buf[NMAXPROC+1];
      // divide on x
      {
        double *xoff = buf; // xoff[nx+1]
        __gnu_parallel::sort(&pos[addr.off(0, 0, 0)], &pos[addr.off(nx, 0, 0)], CmpFloat4<1>()); // sort by x
        for(int ix=0; ix<nx; ix++)
        {
          const int ioff = addr.off(ix, 0, 0);
          xoff[ix] = 0.5 * (pos[ioff].x + pos[1+ioff].x);
          // PRC(xoff[ix]);
        }
        // cerr << endl;
        xoff[0]  = -rmax;
        xoff[nx] = +rmax;
        for(int ix=0; ix<nx; ix++)
          for(int iy=0; iy<ny; iy++)
            for(int iz=0; iz<nz; iz++)
            {
              const int ii = addr.xdi(ix, iy, iz);
              // PRC(ix); PRC(iy); PRC(iz); PRL(ii);
              xlow [ii][0] = xoff[ix];
              xhigh[ii][0] = xoff[ix+1];
            }
      }

      // divide on y
      {
        double *yoff = buf; // yoff[ny+1];
        for(int ix=0; ix<nx; ix++)
        {
          __gnu_parallel::sort(&pos[addr.off(ix, 0, 0)], &pos[addr.off(ix, ny, 0)], CmpFloat4<2>()); // sort by y
          for(int iy=0; iy<ny; iy++)
          {
            const int ioff = addr.off(ix, iy, 0);
            yoff[iy] = 0.5 * (pos[ioff].y + pos[1+ioff].y);
            // PRC(yoff[iy]);
          }
          // cerr << endl;
          yoff[0]  = -rmax;
          yoff[ny] = +rmax;
          for(int iy=0; iy<ny; iy++)
            for(int iz=0; iz<nz; iz++)
            {
              const int ii = addr.xdi(ix, iy, iz);
              xlow [ii][1] = yoff[iy];
              xhigh[ii][1] = yoff[iy+1];
            }
        }
      }
      // divide on z
      {
        double *zoff = buf; // zoff[nz+1];
        for(int ix=0; ix<nx; ix++)
          for(int iy=0; iy<ny; iy++)
          {
            __gnu_parallel::sort(&pos[addr.off(ix, iy, 0)], &pos[addr.off(ix, iy, nz)], CmpFloat4<4>()); // sort by z
            for(int iz=0; iz<nz; iz++)
            {
              const int ioff = addr.off(ix, iy, iz);
              zoff[iz] = 0.5 * (pos[ioff].z + pos[1+ioff].z);
            }
            // cerr << endl;
            zoff[0]  = -rmax;
            zoff[nz] = +rmax;
            for(int iz=0; iz<nz; iz++){
              const int ii = addr.xdi(ix, iy, iz);
              xlow [ii][2] = zoff[iz];
              xhigh[ii][2] = zoff[iz+1];
            }
          }
      }
    }

    inline int which_box(
        const vector3 &pos,
        const vector3 xlow[],
        const vector3 xhigh[])
    {
      int p = 0;
      if(pos[0] < xlow[p][0]) return -1;
      for(int ix=0; ix<npx; ix++, p+=npy*npz){
        if(pos[0] < xhigh[p][0]) break;
      }
      if(pos[0] > xhigh[p][0]) return -1;

      if(pos[1] < xlow[p][1]) return -1;
      for(int iy=0; iy<npy; iy++, p+=npz){
        if(pos[1] < xhigh[p][1]) break;
      }
      if(pos[1] > xhigh[p][1]) return -1;

      if(pos[2] < xlow[p][2]) return -1;
      for(int iy=0; iy<npy; iy++, p++){
        if(pos[2] < xhigh[p][2]) break;
      }
      if(pos[2] > xhigh[p][2]) return -1;

      return p;
    }
    
    inline void which_boxes(
        const vector3 &pos,
        const float h,
        const vector3 xlow[],
        const vector3 xhigh[],
        std::vector<int> &boxes)
    {
      for (int p = 0; p < nrank; p++)
      {
        if (
            pos[0]+h >= xlow[p][0]  && pos[0]-h <= xhigh[p][0] &&
            pos[1]+h >= xlow[p][1]  && pos[1]-h <= xhigh[p][1] &&
            pos[2]+h >= xlow[p][2]  && pos[2]-h <= xhigh[p][2])
          boxes.push_back(p);
      }
    }

    void alltoallv(std::vector<particle_t> psend[], std::vector<particle_t> precv[])
    {
      static MPI_Datatype MPI_PARTICLE = 0;
      if (!MPI_PARTICLE)
      {
        int ss = sizeof(particle_t) / sizeof(float);
        assert(0 == sizeof(particle_t) % sizeof(float));
        MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_PARTICLE);
        MPI_Type_commit(&MPI_PARTICLE);
      }

      static std::vector<int> nsend(nrank), senddispl(nrank+1,0);
      int nsendtot = 0;
      for (int p = 0; p < nrank; p++)
      {
        nsend[p] = psend[p].size();
        senddispl[p+1] = senddispl[p] + nsend[p];
        nsendtot += nsend[p];
      }

      static std::vector<int> nrecv(nrank), recvdispl(nrank+1,0);
      MPI_Alltoall(&nsend[0], 1, MPI_INT, &nrecv[0], 1, MPI_INT, comm);

      int nrecvtot = 0;
      for (int p = 0; p < nrank; p++)
      {
        recvdispl[p+1] = recvdispl[p] + nrecv[p];
        nrecvtot += nrecv[p];
      }


      static std::vector<particle_t> sendbuf, recvbuf;
      sendbuf.resize(nsendtot); 
      recvbuf.resize(nrecvtot);

      int iloc = 0;
      for (int p = 0; p < nrank; p++)
        for (int i = 0; i < nsend[p]; i++)
          sendbuf[iloc++] = psend[p][i];

      assert(senddispl[nrank] == nsendtot);
      assert(recvdispl[nrank] == nrecvtot);

      MPI_Alltoallv(
          &sendbuf[0], &nsend[0], &senddispl[0], MPI_PARTICLE,
          &recvbuf[0], &nrecv[0], &recvdispl[0], MPI_PARTICLE, 
          comm);

      for (int p = 0; p < nrank; p++)
      {
        precv[p].resize(nrecv[p]);
        for (int i = 0; i < nrecv[p]; i++)
          precv[p][i] = recvbuf[recvdispl[p] + i];
      }

    }


    void exchange_particles_alltoall_vector(
        const vector3  xlow[],
        const vector3 xhigh[])
    {
      int myid = rank;
      int nprocs = nrank;

      static std::vector<particle_t> psend[NMAXPROC];
      static std::vector<particle_t> precv[NMAXPROC];
      const int nbody = data.size();

      static bool initcall = true;
      if(initcall)
      {
        initcall = false;
        for(int p=0; p<nprocs; p++)
        {
          psend[p].reserve(64);
          precv[p].reserve(64);
        }
      }

      int iloc = 0;
      Boundary boundary(xlow[myid], xhigh[myid]);
      for(int i=0; i<nbody; i++)
        if(boundary.isinbox(vector3{{data[i].posx, data[i].posy, data[i].posz}}))
          std::swap(data[i],data[iloc++]);

      for(int p=0; p<nprocs; p++)
      {
        psend[p].clear();
        precv[p].clear();
      }

#if 0
      for(int i=0; i<nbody; i++)
      {
        int ibox = which_box(vector3{{data[i].posx,data[i].posy,data[i].posz}}, xlow, xhigh);
        if (nbody < iloc)
          assert(ibox == rank);
        if(ibox < 0)
        {
          std::cerr << myid <<" exchange_particle error: particle in no box..." << std::endl;
          vector3 fpos{{data[i].posx,data[i].posy,data[i].posz}};
          // cerr << pb[i].get_pos() << endl;
          std::cout // << boost::format("[%f %f %f], [%lx %lx %lx]")
            << fpos[0] << " " << fpos[1] << " " << fpos[2] << std::endl;
          //        pb[i].dump();
       
          for (int p = 0; p < nrank; p++)
          {
            fprintf(stderr," rank= %d: xlow= %g %g %g  xhigh= %g %g %g \n", p,
                xlow[p][0],  xlow[p][1],  xlow[p][2],
                xhigh[p][0], xhigh[p][1], xhigh[p][2]);
          }
          MPI_Abort(comm,1);
        }
        else
        {
          psend[ibox].push_back(data[i]);
        }
      }
#else
      std::vector<int> boxes;
      boxes.reserve(nrank);
      for(int i=0; i<nbody; i++)
      {
        boxes.clear();
        which_boxes(vector3{{data[i].posx,data[i].posy,data[i].posz}}, 1.1*data[i].attribute[Attribute_t::H], xlow, xhigh, boxes);
//        which_boxes(vector3{{data[i].posx,data[i].posy,data[i].posz}}, 0, xlow, xhigh, boxes);
        assert(!boxes.empty());
        for (auto ibox : boxes)
          psend[ibox].push_back(data[i]);
      }
#endif

      double dtime = 1.e9;
      {
        const double t0 = MPI_Wtime();
        alltoallv(psend, precv);
        const double t1 = MPI_Wtime();
        dtime = t1 - t0;
        if (isMaster())
          fprintf(stderr, "alltoallv= %g sec \n", t1-t0);
      }
     
#if 0 
      {
        assert(precv[rank].size() == precv[rank].size());
        for(int p=0; p<nrank; p++)
          for (int i = 0; i < (int)precv[p].size(); i++)
            assert(boundary.isinbox(vector3{{precv[p][i].posx, precv[p][i].posy, precv[p][i].posz}}));
      }
#endif


      assert(precv[rank].size() == psend[rank].size());

      int nsendtot = 0, nrecvtot = 0;
      for(int p=0; p<nprocs; p++)
      {
        nsendtot += psend[p].size();
        nrecvtot += precv[p].size();
      }
      int nsendloc = nsendtot, nrecvloc = nrecvtot;
      MPI_Allreduce(&nsendloc,&nsendtot,1, MPI_INT, MPI_SUM,comm);
      MPI_Allreduce(&nrecvloc,&nrecvtot,1, MPI_INT, MPI_SUM,comm);
      double bw = 2.0 * double(sizeof(particle_t) * nsendtot) / dtime * 1.e-9;
      if (isMaster())
      {
        assert(nsendtot == nrecvtot);
        std::cout << "Exchanged particles = " << nsendtot << ", " << dtime << "sec" << std::endl;
        std::cout << "Global Bandwidth " << bw << " GB/s" << std::endl;
      }
      data.clear();
      data.resize(nrecvloc);
      int ip = 0;
      for(int p=0; p<nprocs; p++)
      {
        int size = precv[p].size();
        for(int i=0; i<size; i++)
          data[ip++] = precv[p][i];
      }
      assert(ip == nrecvloc);


      randomShuffle();
      computeMinMax();
    }


    /////////////////////
    //
  public:

    void distribute()
    {
      initialize_division();
      std::vector<vector3> sample_array;
      collect_sample_particles(sample_array, sample_freq);

      /* determine division */
      vector3  xlow[NMAXPROC];
      vector3 xhigh[NMAXPROC];
      const float rmax = std::max(std::abs(_rmin), std::abs(_rmax)) * 1.0001;

      const int nsample = sample_array.size();
      if (rank == 0)
        fprintf(stderr, " -- nsample= %d\n", nsample);
      std::vector<float4> pos(nsample);
#pragma omp parallel for schedule(static)
      for (int i = 0; i < nsample; i++)
        pos[i] = float4(sample_array[i][0], sample_array[i][1], sample_array[i][2],0.0f);

      if (rank == 0)
        determine_division(pos, rmax, xlow, xhigh);


      const int nwords=nrank*3;
      MPI_Bcast(& xlow[0],nwords,MPI_DOUBLE,getMaster(),comm);
      MPI_Bcast(&xhigh[0],nwords,MPI_DOUBLE,getMaster(),comm);

#if 0
      if (isMaster())
      {
        for (int p = 0; p < nrank; p++)
        {
          fprintf(stderr," rank= %d: xlow= %g %g %g  xhigh= %g %g %g \n", p,
               xlow[p][0],  xlow[p][1],  xlow[p][2],
              xhigh[p][0], xhigh[p][1], xhigh[p][2]);
        }

      }
#endif
      exchange_particles_alltoall_vector(xlow, xhigh);

      for (int k = 0; k < 3; k++)
      {
        this-> xlow[k] =  xlow[rank][k];
        this->xhigh[k] = xhigh[rank][k];
      }

#if 0
      {
        const int nbody = data.size();
        Boundary boundary(xlow[rank], xhigh[rank]);
        for(int i=0; i<nbody; i++)
          assert(boundary.isinbox(vector3{{data[i].posx,data[i].posy,data[i].posz}}));
      }
#endif
      distributed = true;
    }
};
