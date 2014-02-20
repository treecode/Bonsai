#pragma once
#include <iostream>
#include <fstream>
#include <cstring>

#include "tipsydefs.h"
#include <sion.h>
#define TSZ 5

size_t sion_write_snapshot(
    const real4 *bodyPositions, 
    const real4 *bodyVelocities, 
    const int *bodyIds, 
    const int n, 
    const std::string &fileName, 
    const float time,
    const int rank, 
    const int nrank,
    int nfiles, 
    const MPI_Comm &comm)
{
  const int NTotal = n;
  int NFirst = 0,  NSecond = 0,  NThird = 0;
  int sid;
  MPI_Comm lComm;
  FILE* fptr;
  char* newfname;
  char myfname[255];
  sion_int64 chunksize;
  int blksize=-1;
  int myrank=rank;
  double t[TSZ],dt[TSZ],tmin[TSZ],tmax[TSZ],tmean[TSZ];

for(int i=0; i < n; i++)
  {
    //Specific for JB his files
    if(bodyIds[i] >= 0         && bodyIds[i] < 100000000) NThird++;
    if(bodyIds[i] >= 100000000 && bodyIds[i] < 200000000) NSecond++;
    if(bodyIds[i] >= 200000000 && bodyIds[i] < 300000000) NFirst++;
  }

  //Sync them to process 0
  int NCombTotal, NCombFirst, NCombSecond, NCombThird;
  NCombTotal  = (NTotal);
  NCombFirst  = (NFirst);
  NCombSecond = (NSecond);
  NCombThird  = (NThird);

  //Since the dust and disk particles are star particles
  //lets add them
  NCombSecond += NCombThird;

  lComm=MPI_COMM_NULL;
  std::strcpy(myfname,fileName.c_str());
  chunksize=sizeof(dump)+NCombFirst*sizeof(dark_particle)+NCombSecond*sizeof(star_particle);
  t[0] = rtc();
  sid = sion_paropen_mpi(myfname, "wb,posix,buffered", &nfiles, comm, &lComm, &chunksize, &blksize, &myrank, &fptr, &newfname);
  t[1] = rtc();

  dump  h;

  //Create tipsy header
  h.time = time;
  h.nbodies = NCombTotal;
  h.ndim = 3;
  h.ndark = NCombFirst;
  h.nstar = NCombSecond;    //Incase of disks we have to finish this
  h.nsph = 0;
  sion_fwrite((char*)&h, sizeof(h), 1, sid);
//  fwrite((char*)&h, sizeof(h), 1, fptr);

  size_t nbytes = 0;

  //Frist write the dark matter particles
  for(int i=0; i < NCombTotal ; i++)
  {
    if(bodyIds[i] >= 200000000 && bodyIds[i] < 300000000)
    {
      //Set particle properties
      dark_particle d;
      d.eps = bodyVelocities[i].w;
      d.mass = bodyPositions[i].w;
      d.pos[0] = bodyPositions[i].x;
      d.pos[1] = bodyPositions[i].y;
      d.pos[2] = bodyPositions[i].z;
      d.vel[0] = bodyVelocities[i].x;
      d.vel[1] = bodyVelocities[i].y;
      d.vel[2] = bodyVelocities[i].z;
      d.phi = bodyIds[i];      //Custom change to tipsy format

      sion_fwrite((char*)&d, sizeof(d), 1, sid);
//      fwrite((char*)&d, sizeof(d), 1, fptr);
      nbytes += sizeof(d);
    } //end if
  } //end i loop

  //Next write the star particles
  for(int i=0; i < NCombTotal ; i++)
  {
    if(bodyIds[i] >= 0 && bodyIds[i] < 200000000)
    {
      //Set particle properties
      star_particle s;
      s.eps = bodyVelocities[i].w;
      s.mass = bodyPositions[i].w;
      s.pos[0] = bodyPositions[i].x;
      s.pos[1] = bodyPositions[i].y;
      s.pos[2] = bodyPositions[i].z;
      s.vel[0] = bodyVelocities[i].x;
      s.vel[1] = bodyVelocities[i].y;
      s.vel[2] = bodyVelocities[i].z;
      s.phi = bodyIds[i];      //Custom change to tipsy format

      s.metals = 0;
      s.tform = 0;
      sion_fwrite((char*)&s, sizeof(s), 1, sid);
//      fwrite((char*)&s, sizeof(s), 1, fptr);
      nbytes += sizeof(s);
      //         outputFile << s;
    } //end if
  } //end i loop

  t[2] = rtc();
  sion_ensure_free_space(sid,chunksize);
  t[3] = rtc();
  sion_parclose_mpi(sid);
  t[4] = rtc();

  dt[0]=t[1]-t[0];
  dt[1]=t[2]-t[1];
  dt[2]=t[3]-t[2];
  dt[3]=t[4]-t[3];
 
  MPI_Reduce(dt,tmin,TSZ,MPI_DOUBLE,MPI_MIN,0,comm);
  MPI_Reduce(dt,tmax,TSZ,MPI_DOUBLE,MPI_MAX,0,comm);
  MPI_Reduce(dt,tmean,TSZ,MPI_DOUBLE,MPI_SUM,0,comm);
  if (rank == 0) printf("Avg. time (s) for open, write, resize, close: %g, %g, %g, %g\n",
    tmean[0]/nrank,tmean[1]/nrank,tmean[2]/nrank,tmean[3]/nrank);
  if (rank == 0) printf("Min. time (s) for open, write, resize, close: %g, %g, %g, %g\n",
    tmin[0],tmin[1],tmin[2],tmin[3]);
  if (rank == 0) printf("Max. time (s) for open, write, resize, close: %g, %g, %g, %g\n",
    tmax[0],tmax[1],tmax[2],tmax[3]);

//  LOGF(stderr,"Wrote %d bodies to tipsy file \n", NCombTotal);

  return nbytes;

}
