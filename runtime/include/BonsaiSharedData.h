#pragma once

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
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiQuickHeader";
    return fn;
  }
};

struct BonsaiSharedQuickData : public BonsaiSharedData
{
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiQuickData";
    return fn;
  }
};

struct BonsaiSharedSnapHeader : public BonsaiSharedHeader
{
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapHeader";
    return fn;
  }
};

struct BonsaiSharedSnapData : public BonsaiSharedData
{
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapData";
    return fn;
  }
};
