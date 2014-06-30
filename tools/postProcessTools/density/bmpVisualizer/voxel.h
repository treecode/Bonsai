#ifndef _VOXEL_INCLUDED
#define _VOXEL_INCLUDED

#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>
#include<cassert>


using namespace std;


class Voxel{

 public:

  int xvoxel;
  int yvoxel;
  int zvoxel;
  int nvoxel;

  double *val;  // value of voxel
  double *val2;  // value of voxel


  Voxel();
  Voxel( const int nfile, const char *fname, const int idxOffset);
  virtual ~Voxel();
  virtual void initVoxel();
  virtual void resetVoxel();
  virtual int getVoxelIndex( const int x, const int y, const int z) const;

  virtual double getMaxVal() const;
  virtual double getMinVal() const;
  virtual double getMaxVal2() const;
  virtual double getMinVal2() const;
  virtual void convertLinear( const double left, const double right);
  virtual void convertLinear2( const double left, const double right);
  virtual void convertLog( const double left, const double right);


  double & operator ()(int x, int y, int z);


};



inline int Voxel::getVoxelIndex( const int x, const int y, const int z) const{
  return x*yvoxel*zvoxel + y*zvoxel + z;
}



inline double & Voxel::operator ()(int x, int y, int z){
  return val[z+zvoxel*(y+yvoxel*x)];
  //return val[x+xvoxel*y];
}



#endif
