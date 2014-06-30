/*
 * By Tomoaki Ishiyama 
 * 
 * June 3, edit by JB compatible with Bonsai output
 * 
 */


#include<iostream>
#include<unistd.h>

#include "voxel.h"
#include "display.h"


using namespace std;

const int CHARMAX = 256;


int main( int argc, char **argv){

  int nfile = 1;
  char inputfile[CHARMAX];
  char outputfile[CHARMAX];
  char outputfile2[CHARMAX];
  char colormap_file[CHARMAX];

  fprintf(stderr,"Usage: %s  inputfile outputfile color_map_file \n", argv[0]);
 

  if(argc != 4 )
  {
    exit(0);
  }
  
  sprintf(inputfile,"%s", argv[1]);
  sprintf(outputfile,"%s", argv[2]);
  sprintf(colormap_file,"%s", argv[3]);  

  cerr << "Processing: " << inputfile << std::endl;

  int csize = 256;
  int cmap_size = csize * csize * 3;
  unsigned char *colormap_array = new unsigned char[cmap_size];
  int cwidth, cheight;
  readBmp( colormap_file, colormap_array, cwidth, cheight);
  assert( cwidth == csize);
  assert( cheight == csize);

  //Create top view
  {
    Voxel *voxel = new Voxel( nfile, inputfile, 0 );

    voxel->convertLinear( 0, 60000.0);
    voxel->convertLinear2( 0, 3000.0);
    voxel->convertLog( 0, 0);
    voxel->convertLinear( 0, 255.0);
    voxel->convertLinear2( 0, 255.0);

    Display *display = new Display( voxel);
    sprintf(outputfile2, "%s-top.bmp", outputfile);
    display->oImageBmp( outputfile2, colormap_array, csize);
    delete display;
    delete voxel;
  }
  
  //Create front view
  {
    Voxel *voxel = new Voxel( nfile, inputfile, 2 );

    voxel->convertLinear( 0, 60000.0);
    voxel->convertLinear2( 0, 3000.0);
    voxel->convertLog( 0, 0);
    voxel->convertLinear( 0, 255.0);
    voxel->convertLinear2( 0, 255.0);

    Display *display = new Display( voxel);
    sprintf(outputfile2, "%s-front.bmp", outputfile);
    display->oImageBmp( outputfile2, colormap_array, csize);
    delete display;
    delete voxel;
  }




}


