#pragma once

struct BonsaiSharedQuickHeader
{
  float tCurrent;
  size_t nBodies;
  char fileName[256];
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiQuickHeader";
    return fn;
  }
};

struct BonsaiSharedQuickData
{
  long long ID;
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
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapHeader";
    return fn;
  }
};

struct BonsaiSharedSnapData
{
  long long ID;
  float x,y,z,mass;
  float vx,vy,vz,vw;
  static const char* sharedFile()
  {
    static const char fn[] = "/BonsaiSnapData";
    return fn;
  }
};
