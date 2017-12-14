#pragma once

class Blending
{
  public:
    enum Type {ONE, ZERO, SRC_ALPHA, ONE_MINUS_SRC_ALPHA};

  private:
    template<Type TYPE>
    static float4 blendColor(const float4 col, const float4 src, const float4 dst)
    {
      float4 res = col;
      switch (TYPE)
      {
        case ONE_MINUS_SRC_ALPHA:
          res.x *= 1.0f - src.w;
          res.y *= 1.0f - src.w;
          res.z *= 1.0f - src.w;
          res.w *= 1.0f - src.w;
          break;
        case SRC_ALPHA:
          res.x *= src.w;
          res.y *= src.w;
          res.z *= src.w;
          res.w *= src.w;
          break;
        case ZERO:
          res.x = res.y = res.z = res.w = 0;
          break;
        case ONE:
          break;
        default:
          assert(0);
      }
      return res;
    }

  public:

    template<Type TYPE_DST, Type TYPE_SRC>
    static float4 getColor(const float4 d, const float4 s)
    {
      float4 res;

      const float4 src = blendColor<TYPE_SRC>(s,s,d);
      const float4 dst = blendColor<TYPE_DST>(d,s,d);

      res.x = src.x + dst.x;
      res.y = src.y + dst.y;
      res.z = src.z + dst.z;
      res.w = src.w + dst.w;

#if 0
      using std::min;
      res.w = min(1.0f, res.w);
#endif

      return res;
    }
};
