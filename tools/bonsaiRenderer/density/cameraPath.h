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
      real rotx,roty,rotz;
      real tranx,trany,tranz;
    };

  private:
    std::vector<camera_t> cameraVec;

  public:
    CameraPath(const std::string &fileName)
    {
      std::ifstream fin(fileName);
      int nval;
      real time;
      fin >> nval >> time;
      cameraVec.resize(nval);
      for (int i = 0; i < nval; i++)
      {
        int idum;
        fin >> idum;
        assert(idum == i+1);

        auto &cam = cameraVec[i];
        fin >> cam.rotx >> cam.roty >> cam.rotz;
        fin >> cam.tranx >> cam.trany >> cam.tranz;

        std::string order;
        fin >> order;
        assert(order == "XYZ");
      }
    }

    int nsteps() const { return cameraVec.size(); }
    const camera_t& getStep(const int step) const { return cameraVec[step%nsteps()]; }
    camera_t getRange(const real r) const
    {
      assert(r >= 0.0);
      assert(r <= 1.0);
      const real t  = r * (nsteps()-1);
      const real t0 = floor(t);
      const real t1 = t0 + 1.0;
      
      const auto& c0 = cameraVec[static_cast<int>(t0)];
      const auto& c1 = cameraVec[std::min(static_cast<int>(t1),nsteps()-1)];

      const float f = (t-t0)/(t1-t0);
      auto cvt = [&](const real f0, const real f1)
      {
        return f0 + (f1-f0)*f;
      };


      camera_t cam;
      cam. rotx = cvt(c0. rotx, c1. rotx);
      cam. roty = cvt(c0. roty, c1. roty);
      cam. rotz = cvt(c0. rotz, c1. rotz);
      cam.tranx = cvt(c0.tranx, c1.tranx);
      cam.trany = cvt(c0.trany, c1.trany);
      cam.tranz = cvt(c0.tranz, c1.tranz);
     
      return cam; 
    }
};
