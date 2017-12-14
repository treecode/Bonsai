#include "display.h"



Display::Display( const int i_width, const int i_height, 
		  const int i_depth, Voxel *const i_voxel){

  width  = i_width;
  height = i_height;
  depth  = i_depth;
  voxel  = i_voxel;

}



Display::Display( Voxel *const i_voxel){

  voxel  = i_voxel;
  width  = voxel->xvoxel;
  height = voxel->yvoxel;
  depth  = voxel->zvoxel;

}



Display::~Display(){

}




void Display::oImageBmp( const char *ofile, const unsigned char *colormap,
			 const int colormap_size) const{

  unsigned char *color_array = (unsigned char *) malloc( sizeof(unsigned char) * width * height * 3);

  int offset = 0;
  for( int i=0; i<width; i++){
    for( int j=0; j<height; j++){
      int index = voxel->getVoxelIndex( j, i, 0);
      int rho = (int)voxel->val[index];
      int v2  = (int)voxel->val2[index];
      if( rho < 0)  rho = 0;
      if( v2 < 0)  v2 = 0;
      if( rho > colormap_size-1)  rho = colormap_size-1;
      if( v2  > colormap_size-1)  v2 = colormap_size-1;
      int ic = rho*colormap_size + v2;
      color_array[3*offset+0] = colormap[3*ic+0];
      color_array[3*offset+1] = colormap[3*ic+1];
      color_array[3*offset+2] = colormap[3*ic+2];
      offset ++;
    }
  }

  outputBmp( width, height, color_array, ofile);
  free( color_array);

}







void outputBmp(const int width, const int height,
               unsigned char *color_array, const char *ofile){

  BmpHeader bmp;

  char bfType[2];
  bfType[0]          = 'B';
  bfType[1]          = 'M';
  bmp.bfSize         = width*height*3 + 54;
  bmp.bfReserved1    = 0;
  bmp.bfReserved2    = 0;
  bmp.bfOffBits      = 54;
  bmp.biSize         = 40;
  bmp.biWidth        = width;
  bmp.biHeight       = height;
  bmp.biPlanes       = 1;
  bmp.biBitCount     = 24;
  bmp.biCompression  = 0;
  bmp.biSizeImage    = 0;
  bmp.biXPixPerMeter = 0;
  bmp.biYPixPerMeter = 0;
  bmp.biClrUsed      = 0;
  bmp.biClrImporant  = 0;

  FILE *outstream = fopen( ofile, "w");
  fwrite( &bfType, sizeof(char), 2, outstream);
  fwrite( &bmp, sizeof(BmpHeader), 1, outstream);
  fwrite( color_array, sizeof(unsigned char), width*height*3, outstream);
  fclose( outstream);

}



void readBmp( const char *infile,
	      unsigned char *color_array, 
	      int &width, int &height){

  BmpHeader bmp;
  char bfType[2];

  FILE *instream = fopen( infile, "r");
  fread( &bfType, sizeof(char), 2, instream);
  fread( &bmp, sizeof(BmpHeader), 1, instream);
  width = (int)bmp.biWidth;
  height = (int)bmp.biHeight;
  fread( color_array, sizeof(unsigned char), width*height*3, instream);
  fclose( instream);

}
