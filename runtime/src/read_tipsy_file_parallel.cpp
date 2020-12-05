/*
 * read_tipsy_file_parallel.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "read_tipsy_file_parallel.h"

using namespace std;

void read_tipsy_file_parallel(std::vector<real4> &bodyPositions, std::vector<real4> &bodyVelocities,
                              std::vector<ullong> &bodiesIDs,  float eps2, string fileName,
                              int rank, int procs, int &NTotal2, int &NFirst,
                              int &NSecond, int &NThird, octree *tree,
                              std::vector<real4> &dustPositions, std::vector<real4> &dustVelocities,
                              std::vector<ullong> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor,
                              const bool restart)
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /*

     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the
     location where previously the potential was stored.
  */


  char fullFileName[256];
  if(restart)
    sprintf(fullFileName, "%s%d", fileName.c_str(), rank);
  else
    sprintf(fullFileName, "%s", fileName.c_str());

  LOG("Trying to read file: %s \n", fullFileName);



  ifstream inputFile(fullFileName, ios::in | ios::binary);
  if(!inputFile.is_open())
  {
    LOG("Can't open input file \n");
    exit(0);
  }

  dump  h;
  inputFile.read((char*)&h, sizeof(h));

  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;


  //Read tipsy header
  NTotal        = h.nbodies;
  NFirst        = h.ndark;
  NSecond       = h.nstar;
  NThird        = h.nsph;

  if (tree) tree->set_t_current((float) h.time);

  //Rough divide
  uint perProc = (NTotal / procs) /reduce_bodies_factor;
  if(restart) perProc = NTotal /reduce_bodies_factor; //don't subdivide when using restart
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;

  dark_particle d;
  star_particle s;

  int globalParticleCount = 0;
  int bodyCount = 0;
  int dustCount = 0;

  for(int i=0; i < NTotal; i++)
  {
    if(i < NFirst)
    {
      inputFile.read((char*)&d, sizeof(d));
      velocity.w        = d.eps;
      positions.w       = d.mass;
      positions.x       = d.pos[0];
      positions.y       = d.pos[1];
      positions.z       = d.pos[2];
      velocity.x        = d.vel[0];
      velocity.y        = d.vel[1];
      velocity.z        = d.vel[2];
      idummy            = d.phi;
    }
    else
    {
      inputFile.read((char*)&s, sizeof(s));
      velocity.w        = s.eps;
      positions.w       = s.mass;
      positions.x       = s.pos[0];
      positions.y       = s.pos[1];
      positions.z       = s.pos[2];
      velocity.x        = s.vel[0];
      velocity.y        = s.vel[1];
      velocity.z        = s.vel[2];
      idummy            = s.phi;
    }


    if(positions.z < -10e10)
    {
       fprintf(stderr," Removing particle %d because of Z is: %f \n", globalParticleCount, positions.z);
       continue;
    }


	globalParticleCount++;

    #ifdef USE_DUST
      if(idummy >= 50000000 && idummy < 100000000)
      {
        dustCount++;
        if( dustCount % reduce_dust_factor == 0 )
          positions.w *= reduce_dust_factor;

        if( dustCount % reduce_dust_factor != 0 )
          continue;
        dustPositions.push_back(positions);
        dustVelocities.push_back(velocity);
        dustIDs.push_back(idummy);
      }
      else
      {
        bodyCount++;
        if( bodyCount % reduce_bodies_factor == 0 )
		      positions.w *= reduce_bodies_factor;

	      if( bodyCount % reduce_bodies_factor != 0 )
		      continue;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);
      }


    #else
      if( globalParticleCount % reduce_bodies_factor == 0 )
        positions.w *= reduce_bodies_factor;

      if( globalParticleCount % reduce_bodies_factor != 0 )
        continue;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(idummy);
    #endif

    particleCount++;


//    if(!restart)
//    {
//      if(bodyPositions.size() > perProc && procCntr != procs)
//      {
//    	if (tree) tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
//        procCntr++;
//
//        bodyPositions.clear();
//        bodyVelocities.clear();
//        bodiesIDs.clear();
//      }
//    }
  }//end while

  inputFile.close();

  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);
//   NTotal2 = particleCount-1;
  NTotal2 = particleCount;
  LOGF(stderr,"NTotal: %d\tper proc: %d\tFor ourself: %d \tNDust: %d \n",
               NTotal, perProc, (int)bodiesIDs.size(), (int)dustPositions.size());
}
