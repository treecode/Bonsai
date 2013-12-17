#pragma once
#include <iostream>
#include <fstream>
#include "tipsydefs.h"

struct real4
{
  float x,y,z,w;
};

size_t write_snapshot(
    const real4 *bodyPositions, 
    const real4 *bodyVelocities, 
    const int *bodyIds, 
    const int n, 
    const std::string &fileName, 
    const float time,
    const int rank, 
    const int nrank, 
    const MPI_Comm &comm)
{
  const int NTotal = n;
  int NFirst = 0,  NSecond = 0,  NThird = 0;

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


  std::ofstream outputFile;
  outputFile.open(fileName.c_str(), std::ios::out | std::ios::binary);

  dump  h;

  if(!outputFile.is_open())
  {
    std::cerr << "Can't open output file: "<< fileName << std::endl;
    exit(0);
  }

  //Create tipsy header
  h.time = time;
  h.nbodies = NCombTotal;
  h.ndim = 3;
  h.ndark = NCombFirst;
  h.nstar = NCombSecond;    //Incase of disks we have to finish this
  h.nsph = 0;
  outputFile.write((char*)&h, sizeof(h));

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

      outputFile.write((char*)&d, sizeof(d));
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
      outputFile.write((char*)&s, sizeof(s));
      nbytes += sizeof(s);
      //         outputFile << s;
    } //end if
  } //end i loop

  outputFile.close();

//  LOGF(stderr,"Wrote %d bodies to tipsy file \n", NCombTotal);

  return nbytes;

}
