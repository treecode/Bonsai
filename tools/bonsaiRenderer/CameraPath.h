#pragma once

#include <fstream>
#include <string>
#include <cassert>
#include <vector>
#include <cmath>

class CameraPath
{
  public:
    using real = double;
    struct camera_t
    {
      union
      {
        struct{
          real rotx,roty,rotz;
          real tranx,trany,tranz;
        };
        real data[6];
      };
    };

  private:
    std::vector<camera_t> cameraVec_base, cameraVec;

  public:
    CameraPath(const std::string &fileName)
    {
      std::ifstream fin(fileName);
      int nval;
      real time;
      fin >> nval >> time;
      cameraVec_base.resize(nval);
      for (int i = 0; i < nval; i++)
      {
        int idum;
        fin >> idum;
        assert(idum == i+1);

        auto &cam = cameraVec_base[i];
        fin >> cam.tranx >> cam.trany >> cam.tranz;
        fin >> cam.rotx >> cam.roty >> cam.rotz;

        cam.rotx *= -180.0/M_PI;
        cam.roty *= -180.0/M_PI;
        cam.rotz *= -180.0/M_PI;
        cam.tranx *= -1.0;
        cam.trany *= -1.0;
        cam.tranz *= -1.0;

        std::string order;
        fin >> order;
        assert(order == "XYZ");
      }
      cameraVec = cameraVec_base;
    }
    
    camera_t interpolate(const real r) const
    {
      assert(r >= 0.0);
      assert(r <= 1.0);
      const int nframes = cameraVec_base.size();
      const real t  = r * (nframes-1);
      const real t0 = floor(t);
      const real t1 = t0 + 1.0;
      
      auto c0 = cameraVec_base[static_cast<int>(t0)];
      auto c1 = cameraVec_base[std::min(static_cast<int>(t1),nframes-1)];

      const real f = (t-t0)/(t1-t0);
      auto cvt = [&](const real f0, const real f1)
      {
        return f0 + (f1-f0)*f;
      };

      /* correct angles, to ensure continuited across [-PI;+PI] boudary */
#if 0
      for (int k = 0; k < 3; k++)
         if (std::abs(c0.data[k] - c1.data[k]) > M_PI)
         {
    //       fprintf(stdout, " k0: %d  %5.2f  %5.2f \n", k, c0.data[k]*M_PI/180.0, c1.data[k]*M_PI/180.0);
           if (c0.data[k] > 0.0)
             c1.data[k] += 360.0;
           else if (c0.data[k] < 0.0)
             c1.data[k] -= 360.0;
     //      fprintf(stdout, " k1: %d  %5.2f  %5.2f \n", k, c0.data[k]*M_PI/180.0, c1.data[k]*M_PI/180.0);
         }
#endif

      camera_t cam;
      for (int k = 0; k < 6; k++)
        cam.data[k] = cvt(c0.data[k], c1.data[k]);

      return cam; 
    }

    void reframe(const int nframe)
    {
      assert(nframe > 0);
      cameraVec.resize(nframe);
      for (int i = 0; i < nframe; i++)
      {
        const real f = static_cast<real>(i)/(nframe-1);
        cameraVec[i] = interpolate(f);
      }
    }

    int nFrames() const { return cameraVec.size(); }
    const camera_t& getFrame(const int step) const { return cameraVec[step%nFrames()]; }
};
