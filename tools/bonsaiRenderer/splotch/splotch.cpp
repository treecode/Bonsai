#include "Splotch.h"
#include "GLSLProgram.h"
#include <GL/glew.h>
#if defined(__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

const char splotchVS[] =
{
  " attribute float spriteSize;                                      \n "
  " uniform float pointScale;                                        \n "
  " void main()                                                      \n "
  " {                                                                \n "
  "   vec4 wpos = vec4(gl_Vertex.xyz, 1.0);                          \n "
  "   gl_Position = gl_ModelViewPorjectionMatrix * wpos;             \n "
  "   vec4 eyeSpacePos = gl_ModelViewMatrix * wpos;                  \n "
  "   float dist = length(eyeSpacePos.xyz);                          \n "
  "   float pointSize = spriteSize*pointScale;                       \n "
  "   gl_PointSize = max(1.0, pointSize / dist);                     \n "
  "   gl_FrontColor = vec4(gl_Color.xyz, 1.0);                       \n "
  " }                                                                \n "
};

const char splotchPS[] =
{
  " uniform sampler2D spriteTex; \n" 
  " void main()                                                      \n "
  " {                                                                \n "
  "    float alpha = texture2D(spriteTex, gl_TexCoord[0].xy).x;      \n "
  "    gl_FragColor = vec4(gl_Color.x*alpha, gl_Color.y*alpha, gl_Color.z*alpha, 1.0); \n"
  " }                                                                \n "
};


static inline float Wkernel(const float q2)
{
  const float q = std::sqrt(q2);
  const float sigma = 8.0f/M_PI;

  const float qm = 1.0f - q;
  if      (q < 0.5f) return sigma * (1.0f + (-6.0f)*q*q*qm);
  else if (q < 1.0f) return sigma * 2.0f*qm*qm*qm;

  return 0.0f;
}

template<typename T>
static inline float4 lMatVec(const T m[4][4], const float4 pos)
{
  return make_float4(
      m[0][0]*pos.x + m[1][0]*pos.y + m[2][0]*pos.z + m[3][0]*pos.w,
      m[0][1]*pos.x + m[1][1]*pos.y + m[2][1]*pos.z + m[3][1]*pos.w,
      m[0][2]*pos.x + m[1][2]*pos.y + m[2][2]*pos.z + m[3][2]*pos.w,
      m[0][3]*pos.x + m[1][3]*pos.y + m[2][3]*pos.z + m[3][3]*pos.w);
}

float4 Splotch::modelView(const float4 pos) const
{
  return lMatVec(modelViewMatrix,pos);
}
float4 Splotch::projection(const float4 pos) const
{
  return lMatVec(projectionMatrix,pos);
}

void Splotch::transform(const bool perspective)
{
  const int np = vtxArray.size();
  vtxArrayView.realloc(np);
  depthArray.resize(np);

  const auto &colorMapTex = *colorMapTexPtr;


  int nVisible = 0;
#pragma omp parallel for schedule(runtime) reduction(+:nVisible)
  for (int i = 0; i < np; i++)
  {
    const auto &vtx = vtxArray[i];
    const float4 pos0 = make_float4(vtx.pos.x,vtx.pos.y,vtx.pos.z,1.0f);
    const float4 posO = modelView(pos0);
    const float4 posP = projection(posO);

    const float  wclip = -1.0f/posO.z;
    float4 posV = make_float4(posP.x*wclip, posP.y*wclip, posP.z*wclip, -1.0f);
    float3 col = make_float3(-1.0f);

    const float depth = posV.z;
    if (depth >= depthMin && depth <= depthMax)
    {

#if 0
      posV.x = (posV.x + 1.0f) * 0.5f * width;
      posV.y = (1.0f - posV.y) * 0.5f * height;
#else
      posV.x = (posV.x + 1.0f)*0.5f*width;
      posV.y = (posV.y + 1.0f)*0.5f*height;
#endif
      using std::abs;
      posV.w = vtx.pos.h * 0.5f * width *abs(wclip);
      assert(posV.w > 0.0f);

      using std::sqrt;
      using std::min;
      posV.w = sqrt(posV.w*posV.w + minHpix*minHpix);
      posV.w = min(posV.w, maxHpix);
      assert(posV.w > 0.0f);

      if (   posV.x - posV.w <= width
          && posV.x + posV.w >= 0
          && posV.y - posV.w <= height
          && posV.y + posV.w >= 0)
      {
        const float s = vtx.attr.vel;
        const float t = vtx.attr.rho;
        assert(s>=0.0f && s<=1.0f);
        assert(t>=0.0f && t<=1.0f);
        const auto &tex = colorMapTex(s,t);
        col = make_float3(tex[0],tex[1],tex[2]);
      }
      else
        posV.w = -1.0;
    }

    depthArray  [i] = depth;
    vtxArrayView[i] = 
    {
      pos2d_t(posV.x, posV.y, posV.w),
      make_float4(col, 1.0f),
      vtx.attr
    };

    nVisible += vtxArrayView[i].isVisible();
  }
  fprintf(stderr, "nParticles= %d nVisible= %d\n", np, nVisible);
}

void Splotch::depthSort()
{
  const int np = depthArray.size();

  using pair = std::pair<float,int>;
  std::vector<pair> depthMap;
  depthMap.reserve(np);

  for (int i = 0; i < np; i++)
    if (vtxArrayView[i].isVisible())
      depthMap.push_back(std::make_pair(depthArray[i],i));

#if 0
  __gnu_parallel::sort(depthMap.begin(), depthMap.end(),
      [](const pair &a, const pair &b) { return a.first < b.first;} );
#endif

  const int npVis = depthMap.size();
  VertexArrayView vtxView(npVis);

#pragma omp parallel for 
  for (int i = 0; i < npVis; i++)
  {
    const auto &map = depthMap[i];
    assert(map.second >= 0);
    assert(map.second < np);
#if 0
    fprintf(stderr, "i= %d  npVis= %d  map.second= %d  np= %d\n",
        i, npVis, map.second, np);
#endif
    vtxView[i] = vtxArrayView[map.second];
  }

  swap(vtxArrayView,vtxView);
}

// assumes atomic execution
Splotch::Quad Splotch::rasterize(const VertexView &vtx, const Splotch::Quad &range, std::vector<color_t> &fb)
{
  using std::max;
  using std::min;
  using std::floor;
  using std::ceil;

  Quad q;
  q.x0  = max(range.x0, floor(vtx.pos.x - vtx.pos.h));
  q.x1  = min(range.x1, ceil (vtx.pos.x + vtx.pos.h));
  q.y0  = max(range.y0, floor(vtx.pos.y - vtx.pos.h));
  q.y1  = min(range.y1, ceil (vtx.pos.y + vtx.pos.h));

  const float invh  = 1.0f/vtx.pos.h;
  const float invh2 = invh*invh;
  const float width  = range.x1 - range.x0;
  const float height = range.y1 - range.y0;
  for (float iy = q.y0; iy < q.y1; iy++)
    for(float ix = q.x0; ix < q.x1; ix++)
    {
      const float dx = ix - vtx.pos.x;
      const float dy = iy - vtx.pos.y;
      const float q2 = (dx*dx + dy*dy) * invh2;
      const float fac = Wkernel(q2);

      float4 color = vtx.color;
      color.w = fac; /* alpha */

      const int idx = (ix - range.x0) + width*(iy - range.y0);
      assert(idx >= 0);
      assert(idx < width*height);
      fb[idx] = Blending::getColor<Blending::ONE,Blending::SRC_ALPHA>(fb[idx],color);
    }

  return q;
}

void Splotch::render()
{
  const int np = vtxArrayView.size();

  image.resize(width*height);
  std::fill(image.begin(), image.end(), make_float4(0.0f));

#define NTHREADMAX 256
  std::vector<color_t> fbVec[NTHREADMAX];

#pragma omp parallel
  {
    const int nt = omp_get_num_threads();
    assert(nt <= NTHREADMAX);
    const int tid = omp_get_thread_num();
    auto &fb = fbVec[tid];
    fb.resize(width*height);
    std::fill(fb.begin(), fb.end(), make_float4(0.0f));

    Quad range;
    range.x0 = 0;
    range.x1 = width;
    range.y0 = 0;
    range.y1 = height;

    if (tid == 0)
      fprintf(stderr, "rasterize begin .. \n");

    const int ipert  = (np+nt-1)/nt;
    const int ibeg = tid * ipert;
    const int iend = std::min(ibeg+ipert, np);

    for (int i = ibeg; i < iend; i++)
    {
      rasterize(vtxArrayView[i], range, fb);
    }

    if (tid == 0)
      fprintf(stderr, "rasterize end .. \n");

#pragma omp barrier

#pragma omp for schedule(runtime) 
    for (int idx = 0; idx < width*height; idx++)
      for (int k = 0; k < nt; k++)
        image[idx] = Blending::getColor<Blending::ONE,Blending::ONE>(image[idx], fbVec[k][idx]);
  }
}


void Splotch::finalize()
{
#pragma omp for schedule(runtime) collapse(2)
  for (int j = 0; j < height; j++)
    for (int i = 0; i < width; i++)
    {
      const int idx = j*width + i;
      float4 src = image[idx];
      float4 dst;
#if 0
      fprintf(stderr, " (%3d,%3d): %g %g %g \n",
          i,j, src.x,src.y,src.z);
#endif

#if 1
      const float scale = 0.05f;
      const float gamma = 0.5f;
      src.x *= scale;
      src.y *= scale;
      src.z *= scale;
      src.x = std::pow(src.x, gamma);
      src.y = std::pow(src.y, gamma);
      src.z = std::pow(src.z, gamma);
#endif

      dst.x = 1.0f - exp(-src.x);
      dst.y = 1.0f - exp(-src.y);
      dst.z = 1.0f - exp(-src.z);
      dst.w = 1.0f;

#if 0
      const float scale = 1.0f;
      const float gamma = 0.2f;
      dst.x *= scale;
      dst.y *= scale;
      dst.z *= scale;
      dst.x = std::pow(dst.x, gamma);
      dst.y = std::pow(dst.y, gamma);
      dst.z = std::pow(dst.z, gamma);
#endif
      

      image[idx] = dst;
    }
}


void Splotch::genImage(const bool perspective)
{
  fprintf(stderr , " --- transform \n");
  transform(perspective);
  fprintf(stderr , " --- depthSort \n");
  depthSort();
  fprintf(stderr , " --- render \n");
  render();
  fprintf(stderr , " --- finalize \n");
  finalize();
}
