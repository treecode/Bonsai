#pragma once

#include "IDType.h"

struct BonsaiSharedQuickHeader
{
  float tCurrent;
  size_t nBodies;
  char fileName[256];
  bool handshake;
  bool done_writing;
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiQuickHeader";
    return fn;
  }
};

struct BonsaiSharedQuickData
{
  IDType ID;
  float x,y,z,mass;
  float vx,vy,vz,vw;
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiQuickData";
    return fn;
  }
};

struct BonsaiSharedSnapHeader
{
  float tCurrent;
  size_t nBodies;
  char fileName[256];
  bool handshake;
  bool done_writing;
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapHeader";
    return fn;
  }
};

struct BonsaiSharedSnapData
{
  IDType ID;
  float x,y,z,mass;
  float vx,vy,vz,vw;
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapData";
    return fn;
  }
};
