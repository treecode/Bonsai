// g++ convDensBinToAscii.cpp -O3 convDensBinToAscii
#include <stdio.h>
#include <iostream>
#include <stdlib.h>

using namespace std;

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    fprintf(stderr,"Usage: %s infile outfile\n",argv[0]);
    exit(0);
  }

  FILE *fin  = fopen( argv[1], "rb");
  FILE *fout = fopen( argv[2], "w");
  
  if(fin == NULL)
  {
    printf("Failed to open file: %s \n", argv[1]);
    exit(0);
  }
  if(fout == NULL)
  {
    printf("Failed to open file: %s \n", argv[2]);
    exit(0);
  }
  
  int header[4]; double range[6];
  fread( header, sizeof(int), 4, fin);
  fread( range,  sizeof(double), 6, fin);
  if( header[0] != -1){
    cerr << "this format can not be read" << endl;
  }
  int nx = header[1];  int ny = header[2];  int nz = header[3];

  int nynz = nx * ny;
  cerr << nx << "\t" << ny << "\t" << nz << endl;
  cerr << range[0] << "\t" << range[1] << "\t"
       << range[2] << "\t" << range[3] << "\t"
       << range[4] << "\t" << range[5] << endl;


  typedef struct pack_f5{
    float f[5];
  }pack_f5;
  pack_f5 *voxel_part = new pack_f5[nynz];
  float tempF = 0.0;
  
  fprintf(fout, "# Time %f \n", range[0]);
  fprintf(fout, "# X Y DTop DVTop DFront DVFront Rphi\n");

  fread( voxel_part, sizeof(pack_f5), nynz, fin); //Read the density xy,vxy,xz,vxz
    
  for( int k=0; k<nx; k++){
    for( int l=0; l<ny; l++){
      
      int idx = k*ny + l;
      
      float rphi =  voxel_part[idx].f[4];
      
      //If statement for easier gnuplot usage
      if(rphi <= 0)
      {
        fprintf(fout,"%d %d %f %f %f %f -\n", k, l, 
                      voxel_part[idx].f[0], voxel_part[idx].f[1],
                      voxel_part[idx].f[2], voxel_part[idx].f[3]);
      }
      else
      {
        fprintf(fout,"%d %d %f %f %f %f %f\n", k, l, 
                      voxel_part[idx].f[0], voxel_part[idx].f[1],
                      voxel_part[idx].f[2], voxel_part[idx].f[3], 
                      voxel_part[idx].f[4]);
      }             
    }
  }

  delete [] voxel_part;
  fclose(fin);
  fclose(fout);
}
