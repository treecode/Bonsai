#pragma once
#include <omp.h>
#include <parallel/algorithm>
#include <cmath>
#include <cassert>
#include <algorithm>
#include "vector_math.h"
#include "Texture.h"
#include "Vertex.h"
#include "Blending.h"
#include "MathArray.h"


class Splotch
{
  public:
    using pos2d_t = Pos2D<float>;
    using pos3d_t = Pos3D<float>;
    using attr_t  = Attribute<float>;  
    using color_t = float4;

  private:

    using VertexArray     = VertexArrayT<pos3d_t,attr_t,color_t>;
    using VertexArrayView = VertexArrayT<pos2d_t,attr_t,color_t>;
    using Vertex          = VertexArray::Vertex;
    using VertexRef       = VertexArray::VertexRef;
    using VertexView      = VertexArrayView::Vertex;
    using ShortVec3       = MathArray<float,3>;
    
    bool useGL;

    VertexArray     vtxArray;
    VertexArrayView vtxArrayView;
    std::vector<float> depthArray;
    float2 invProjRange;


    struct Quad
    {
      float x0,x1;
      float y0,y1;
    };
    
    int width, height;
    std::vector<float4> image;

    double  modelViewMatrix[4][4];
    double projectionMatrix[4][4];

    float4 baseColor;
    float spriteSizeScale;

    float depthMin;
    float depthMax;
    float minHpix, maxHpix;

    Texture2D<ShortVec3> *colorMapTexPtr;

  public:
    Splotch(const bool _useGL = true ) :
      useGL(_useGL),
      spriteSizeScale(1.0f),
      depthMin(0.2f),
      depthMax(1.0f),
      minHpix(0.1f),
      maxHpix(100.0f),
      colorMapTexPtr(NULL)
  {}
    ~Splotch() 
    {
      if (colorMapTexPtr) delete colorMapTexPtr;
    }
   
    /* getters/setters */ 
    void  setColorMap(const float3 *img, const int w, const int h, const float scale = 1.0f)
    { 
      std::vector<ShortVec3> tex(w*h);
      for (int i = 0; i < w*h; i++)
      {
        tex[i][0] = img[i].x * scale;
        tex[i][1] = img[i].y * scale;
        tex[i][2] = img[i].z * scale;
      }
      colorMapTexPtr = new Texture2D<ShortVec3>(&tex[0], w, h); 
    }
    const std::vector<float4>& getImage() const {return image;}
    float4 getPixel(const int i, const int j)
    {
      assert(i >= 0 && i < width);
      assert(j >= 0 && j < height);
      return image[j*width + i];
    }

    void setWidth(const int w)  {width = w;}
    void setHeight(const int h) {height = h;}
    int  getWidth()  const {return width;}
    int  getHeight() const {return height;}
    
    void  setDepthMin(const float x) {depthMin = x;}
    void  setDepthMax(const float x) {depthMax = x;}
    float getDepthMin() const {return depthMin;}
    float getDepthMax() const {return depthMax;}

    void setModelViewMatrix(const double m[4][4])
    {
      for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
          modelViewMatrix[j][i] = m[j][i];
    }
    void setProjectionMatrix(const double m[4][4])
    {
      for (int j = 0; j < 4; j++)
        for (int i = 0; i < 4; i++)
          projectionMatrix[j][i] = m[j][i];
    }

    void resize(const int n)
    {
      vtxArray.realloc(n);
    }
    VertexRef vertex_at(const int i) {return vtxArray[i]; }

  private:
    float4 modelView(const float4 pos) const;
    float4 projection(const float4 pos) const;

    void transform(const bool perspective);
    void depthSort();

    // assumes atomic execution
    Quad rasterize(const VertexView &vtx, const Quad &range, std::vector<color_t> &fb);
    void render();
    void finalize();

  public:
    void genImage(const bool perspective = true);

};

