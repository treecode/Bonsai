#pragma once

#include <string>
#include "IDType.h"
#ifdef BONSAI_CATALYST_STDLIB
 #include <boost/lexical_cast.hpp>
 #define bonsaistd boost
 #define jb_to_string boost::lexical_cast<std::string>
#else
 #define jb_to_string std::to_string
#endif

struct BonsaiSharedHeader
{
  float tCurrent;
  size_t nBodies;
  char fileName[256];
  bool handshake;
  bool done_writing;
};
  
struct BonsaiSharedData
{
  IDType ID;
  float x,y,z,mass;
  float vx,vy,vz,vw;
  float rho,h;
};

struct BonsaiSharedQuickHeader : public BonsaiSharedHeader
{
  static std::string sharedFile(const int rank, const int pid)
  {
    const std::string fn = "/BonsaiQuickHeader-"+jb_to_string(pid)+"-"+jb_to_string(rank);
    return fn;
  }
};

struct BonsaiSharedQuickData : public BonsaiSharedData
{
  static std::string sharedFile(const int rank, const int pid)
  {
    const std::string fn = "/BonsaiQuickData-"+jb_to_string(pid)+"-"+jb_to_string(rank);
    return fn;
  }
};

struct BonsaiSharedSnapHeader : public BonsaiSharedHeader
{
  static std::string sharedFile(const int rank, const int pid)
  {
    const std::string fn = "/BonsaiSnapHeader-"+jb_to_string(pid)+"-"+jb_to_string(rank);
    return fn;
  }
};

struct BonsaiSharedSnapData : public BonsaiSharedData
{
  static std::string sharedFile(const int rank, const int pid)
  {
    const std::string fn = "/BonsaiSnapData-"+jb_to_string(pid)+"-"+jb_to_string(rank);
    return fn;
  }
};
