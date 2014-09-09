#pragma once

#include <string>
#include "IDType.h"

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
  static const char* sharedFile(const int rank)
  {
    const std::string fn = "/BonsaiQuickHeader-"+std::to_string(rank);
    return fn.c_str();
  }
};

struct BonsaiSharedQuickData : public BonsaiSharedData
{
  static const char* sharedFile(const int rank)
  {
    const std::string fn = "/BonsaiQuickData-"+std::to_string(rank);
    return fn.c_str();
  }
};

struct BonsaiSharedSnapHeader : public BonsaiSharedHeader
{
  static const char* sharedFile(const int rank)
  {
    const std::string fn = "/BonsaiSnapHeader-"+std::to_string(rank);
    return fn.c_str();
  }
};

struct BonsaiSharedSnapData : public BonsaiSharedData
{
  static const char* sharedFile(const int rank)
  {
    const std::string fn = "/BonsaiSnapData-"+std::to_string(rank);
    return fn.c_str();
  }
};
