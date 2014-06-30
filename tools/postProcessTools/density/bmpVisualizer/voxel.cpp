#include "voxel.h"
#include "stdio.h"

Voxel::Voxel(){

  initVoxel();

}




Voxel::Voxel( const int nfile, const char *fname, const int idxIncrease){

  int offset = 0;
  for( int i=0; i<nfile; i++){
    char ctmp[256];
    //sprintf( ctmp, "%s-%d", fname, i);
    sprintf( ctmp, "%s", fname);
    cerr << ctmp << endl;

    FILE *fin = fopen( ctmp, "rb");
    
    if(fin == NULL)
    {
      printf("Failed to open file: %s \n", ctmp);
      exit(0);
    }
    
    int header[4]; double range[6];
    fread( header, sizeof(int), 4, fin);
    fread( range, sizeof(double), 6, fin);
    if( header[0] != -1){
      cerr << "this format can not be read" << endl;
    }
    int nx = header[1];  int ny = header[2];  int nz = header[3];

    // add temporarily
    nz = nx;
    nx = 1;
    // end of add

    int nynz = ny * nz;
    cerr << nx << "\t" << ny << "\t" << nz << endl;
    cerr << range[0] << "\t" << range[1] << "\t"
	 << range[2] << "\t" << range[3] << "\t"
	 << range[4] << "\t" << range[5] << endl;

    if( i==0){
      xvoxel = nz;
      yvoxel = ny;
      zvoxel = 1;
      initVoxel();
    }

    typedef struct pack_f5{
      float f[5];
    }pack_f5;
    pack_f5 *voxel_part = new pack_f5[nynz];
    float tempF = 0.0;
    for( int j=0; j<nx; j++){
      fread( voxel_part, sizeof(pack_f5), nynz, fin); //Read the density xy,vxy,xz,vxz
      
      for( int k=0; k<ny; k++){
        for( int l=0; l<nz; l++){

	  int ii = getVoxelIndex( l, k, 0);
	  assert( ii < ny*nz);
          int vi = l + nz*k;
	  assert( vi < nynz*nx);
		
          //	  fprintf(stderr,"%d\t%d\t\t %f %f %f %f %f \n", k, l, 
          //		  voxel_part[vi].f[0], voxel_part[vi].f[1], voxel_part[vi].f[2],
          //		  voxel_part[vi].f[3], voxel_part[vi].f[4]);

          val[ii]  += voxel_part[vi].f[0 + idxIncrease];
          val2[ii] += voxel_part[vi].f[1 + idxIncrease];                    
        }
      }
    }
    delete [] voxel_part;

    offset += nx;
    fclose(fin);
  }

}



void Voxel::initVoxel(){

  nvoxel = xvoxel * yvoxel * zvoxel;

  val = new double[nvoxel];
  val2 = new double[nvoxel];
  resetVoxel();

}



Voxel::~Voxel(){

  delete [] val;
  delete [] val2;

}



void Voxel::resetVoxel(){

  for( int i=0; i<nvoxel; i++){
    val[i] = 0.0;
    val2[i] = 0.0;
  }
 
}




double Voxel::getMaxVal() const{

  double maxval = 0.0;
  for( int i=0; i<nvoxel; i++){
    if( maxval < val[i]){
      maxval = val[i];
    }
  }

  return maxval;

}



double Voxel::getMinVal() const{

  double minval = HUGE;
  for( int i=0; i<nvoxel; i++){
    if( minval > val[i]){
      minval = val[i];
    }
  }

  return minval;

}



double Voxel::getMaxVal2() const{

  double maxval = 0.0;
  for( int i=0; i<nvoxel; i++){
    if( maxval < val2[i]){
      maxval = val2[i];
    }
  }

  return maxval;

}



double Voxel::getMinVal2() const{

  double minval = HUGE;
  for( int i=0; i<nvoxel; i++){
    if( minval > val2[i]){
      minval = val2[i];
    }
  }

  return minval;

}



void Voxel::convertLinear( const double minval, const double maxval){

  double minnow = getMinVal();
  double maxnow = getMaxVal();

  double slope = ( maxval - minval) / ( maxnow - minnow);
  fprintf( stderr, "Normalization min:%e max:%e slope:%e\n", minnow, maxnow, slope);

  for( int i=0; i<nvoxel; i++){
    val[i] = slope * ( val[i] - minnow) + minval;
  }

}


void Voxel::convertLinear2( const double minval, const double maxval){

  double minnow = getMinVal2();
  double maxnow = getMaxVal2();

  double slope = ( maxval - minval) / ( maxnow - minnow);
  fprintf( stderr, "Normalization min:%e max:%e slope:%e\n", minnow, maxnow, slope);

  for( int i=0; i<nvoxel; i++){
    val2[i] = slope * ( val2[i] - minnow) + minval;
  }

}



void Voxel::convertLog( const double left, const double right){

  for( int i=0; i<nvoxel; i++){
    val[i] = log( val[i] + 1.0);
    val2[i] = log( val2[i] + 1.0);
  }

}



