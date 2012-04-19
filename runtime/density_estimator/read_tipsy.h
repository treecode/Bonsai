#ifndef __READ_TIPSY_H__
#define __READ_TIPSY_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include "tipsydefs.h"


class ReadTipsy
{
  public:
    struct Real4 {Real x, y, z, w;};

    /***********************/

    std::string fileName;
    int    procs;
    int    rank;

    /***********************/

    std::vector< int > IDs;
    std::vector<Real4> positions;
    std::vector<Real4> velocities;

    int    NTotal;
    int    NFirst;
    int    NSecond;
    int    NThird;

    /***********************/

    ReadTipsy(const std::string &_fileName, const int _procs = 1, const int _rank = 0) : 
      fileName(_fileName), procs(_procs), rank(_rank)
  {
    //Process 0 does the file reading and sends the data
    //to the other processes

    //Now we have different types of files, try to determine which one is used
    /*****
      If individual softening is on there is only one option:
      Header is formatted as follows: 
      N     #       #       #
      so read the first number and compute how particles should be distributed

      If individual softening is NOT enabled, i can be anything, but for ease I assume standard dumbp files:
      no Header
      ID mass x y z vx vy vz
      now the next step is risky, we assume mass adds up to 1, so number of particles will be : 1 / mass
      use this as initial particle distribution

*/


    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());

    std::cerr << "Trying to read file: " << fullFileName << std::endl;

    std::ifstream inputFile(fullFileName, std::ios::in | std::ios::binary);
    if(!inputFile.is_open())
    {
      std::cerr << "Can't open input file \n";
      exit(0);
    }

    read_inputFile(inputFile);

    inputFile.close();
  }

    ReadTipsy(const int _procs = 1, const int _rank = 0) :
      procs(_procs), rank(_rank)
  {
    fprintf(stderr, " Reading TIPSY file from stdin ... \n");
    read_inputFile(std::cin);
  }

  protected:

    void read_inputFile(std::istream &inputFile)
    {
      dump  h;
      inputFile.read((char*)&h, sizeof(h));  

      int iPad, NTotal;
      Real4 pos, vel;

      //Read tipsy header  
      NTotal        = h.nbodies;
      NFirst        = h.ndark;
      NSecond       = h.nstar;
      NThird        = h.nsph;

      //Rough divide
      unsigned int perProc = NTotal / procs;
      positions .reserve(perProc+10);
      velocities.reserve(perProc+10);
      IDs       .reserve(perProc+10);
      perProc -= 1;

      //Start reading
      int particleCount = 0;
      int procCntr = 1;

      dark_particle d;
      star_particle s;

      for(int i=0; i < NTotal; i++)
      {
        if(i < NFirst)
        {
          inputFile.read((char*)&d, sizeof(d));
          vel.w = d.eps;
          pos.w = d.mass;
          pos.x = d.pos[0];
          pos.y = d.pos[1];
          pos.z = d.pos[2];
          vel.x = d.vel[0];
          vel.y = d.vel[1];
          vel.z = d.vel[2];
          iPad  = d.phi;
        }
        else
        {
          inputFile.read((char*)&s, sizeof(s));
          vel.w = s.eps;
          pos.w = s.mass;
          pos.x = s.pos[0];
          pos.y = s.pos[1];
          pos.z = s.pos[2];
          vel.x = s.vel[0];
          vel.y = s.vel[1];
          vel.z = s.vel[2];
          iPad  = s.phi;
        }

        positions .push_back(pos);
        velocities.push_back(vel);
        IDs       .push_back(iPad);  

        particleCount++;


        if(positions.size() > perProc && procCntr != procs)
        { 
          assert(false);
#if 0
          tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
#endif
          procCntr++;

          positions .clear();
          velocities.clear();
          IDs       .clear();
        }
      }//end while


      //Clear the last one since its double
      //   bodyPositions.resize(bodyPositions.size()-1);  
      //   NTotal2 = particleCount-1;
      this->NTotal = particleCount;
      std::cerr << "NTotal: " << NTotal << "\tper proc: " << perProc << "\tFor ourself:" << IDs.size() << std::endl;
    }

};

#endif /* __READ_TIPSY_H__ */
