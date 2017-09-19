#ifndef _DISPLAY_INCLUDED
#define _DISPLAY_INCLUDED

#include<iostream>
#include<fstream>
#include<cstdlib>
#include<cstdio>
#include<cmath>

#include "voxel.h"



using namespace std;



class Display{

 private:

  int width;  // image width
  int height; // image height
  int depth;  // image depth

 public:

  Voxel *voxel;

  Display( const int i_width, const int i_height, const int i_depth, Voxel *const i_voxel);
  Display( Voxel *const i_voxel);
  virtual ~Display();

  virtual void oImageBmp( const char *ofile, const unsigned char *colormap,
			  const int colormap_size) const;


};



typedef struct BmpHeader{

  unsigned int  bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned int  bfOffBits;

  unsigned int  biSize;
  unsigned int  biWidth;
  unsigned int  biHeight;
  unsigned short biPlanes;
  unsigned short biBitCount;
  unsigned int  biCompression;
  unsigned int  biSizeImage;
  unsigned int  biXPixPerMeter;
  unsigned int  biYPixPerMeter;
  unsigned int  biClrUsed;
  unsigned int  biClrImporant;

}BmpHeader, *pBmpHeader;

void outputBmp(const int width, const int height,
               unsigned char *color_array, const char *ofile);
void readBmp( const char *infile,
	      unsigned char *color_array,
	      int &width, int &height);


#endif
