/*
 * tipsyIO.cpp
 *
 *  Created on: Sep 6, 2016
 *      Author: jbedorf
 */

#include "tipsyIO.h"



void tipsyIO::ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities, real4 *bodyAccelerations,  ullong *bodiesIDs,
                     int toSend, const MPI_Comm &mpiCommWorld)
{
#ifdef USE_MPI
  //First send the number of particles, then the actual sample data
  MPI_Send(&toSend, 1, MPI_INT, destination, destination*2 , mpiCommWorld);

  //Send the positions, velocities and ids
  MPI_Send( bodyPositions,  toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+1, mpiCommWorld);
  MPI_Send( bodyVelocities, toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+2, mpiCommWorld);
  MPI_Send( bodyAccelerations, toSend*sizeof(real)*4, MPI_BYTE, destination, destination*2+2, mpiCommWorld);
  MPI_Send( bodiesIDs,      toSend*sizeof(ullong), MPI_BYTE, destination, destination*2+3, mpiCommWorld);
#endif
}

void tipsyIO::ICRecv(int recvFrom, int procId, std::vector<real4> &bodyPositions, std::vector<real4> &bodyVelocities,
                     std::vector<real4> &bodyAccelerations,
                     std::vector<ullong> &bodiesIDs, const MPI_Comm &mpiCommWorld)
{
#ifdef USE_MPI
  MPI_Status status;
  int nreceive;

  //First receive the number of particles, then the actual sample data
  MPI_Recv(&nreceive, 1, MPI_INT, recvFrom, procId*2, mpiCommWorld,&status);

  bodyPositions.resize(nreceive);
  bodyVelocities.resize(nreceive);
  bodyAccelerations.resize(nreceive);
  bodiesIDs.resize(nreceive);

  //Receive the positions, velocities and ids
  MPI_Recv( (real*  )&bodyPositions[0],  nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+1, mpiCommWorld,&status);
  MPI_Recv( (real*  )&bodyVelocities[0], nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+2, mpiCommWorld,&status);
  MPI_Recv( (real*  )&bodyAccelerations[0], nreceive*sizeof(real)*4, MPI_BYTE, recvFrom, procId*2+2, mpiCommWorld,&status);
  MPI_Recv( (ullong*)&bodiesIDs[0],      nreceive*sizeof(ullong), MPI_BYTE, recvFrom, procId*2+3, mpiCommWorld,&status);
#endif
}

void tipsyIO::writeFile(real4 *bodyPositions, real4 *bodyVelocities, real4 *bodyAccelerations, ullong* bodyIds,
                        int n, std::string fileName, float time,
                        const int rank, const int nProcs, const MPI_Comm &mpiCommWorld,
                        bool perProcess)
{
    if(!perProcess && rank != 0)
    {
      //Send data to process 0 and that's it for us
      ICSend(0,  bodyPositions, bodyVelocities, bodyAccelerations, bodyIds, n, mpiCommWorld);
      return;
    }

    std::ofstream outputFile;
    outputFile.open(fileName.c_str(), std::ios::out | std::ios::binary);
    if(!outputFile.is_open())
    {
        std::cerr << "Can't open output file: "<< fileName << std::endl;
      ::exit(0);
    }

    //Buffer to store complete snapshot
    std::vector<real4>   allPositions;
    std::vector<real4>   allVelocities;
    std::vector<real4>   allAccelerations;
    std::vector<ullong>  allIds;

    allPositions. insert(allPositions.begin(),  &bodyPositions[0],  &bodyPositions[n]);
    allVelocities.insert(allVelocities.begin(), &bodyVelocities[0], &bodyVelocities[n]);
    allAccelerations.insert(allAccelerations.begin(), &bodyAccelerations[0], &bodyAccelerations[n]);
    allIds.       insert(allIds.begin(),        &bodyIds[0],        &bodyIds[n]);

    if(!perProcess)
    {
        //Now receive the data from the other processes
        std::vector<real4>   extPositions;
        std::vector<real4>   extVelocities;
        std::vector<real4>   extAccelerations;
        std::vector<ullong>  extIds;

        for(int recvFrom=1; recvFrom < nProcs; recvFrom++)
        {
          ICRecv(recvFrom, rank, extPositions, extVelocities, extAccelerations, extIds, mpiCommWorld);
          allPositions.insert(allPositions.end(), extPositions.begin(), extPositions.end());
          allVelocities.insert(allVelocities.end(), extVelocities.begin(), extVelocities.end());
          allAccelerations.insert(allAccelerations.end(), extAccelerations.begin(), extAccelerations.end());
          allIds.insert(allIds.end(), extIds.begin(), extIds.end());
        }
    }

    //Count the particle types
    int NDM = 0, NStar = 0;
    for(int i=0; i < allIds.size(); i++)
    {
      if(allIds[i] >= DARKMATTERID) NDM++;
      else                          NStar++;
    }

    //Create and write Tipsy header
    dumpV2  h;
    h.time    = time;
    h.nbodies = allIds.size();
    h.ndim    = 3;
    h.ndark   = NDM;
    h.nstar   = NStar;
    h.nsph    = 0;
    h.version = 6; // GL 6 as 2 | 4, 2 because original, 4 because it includes acceleration
    outputFile.write((char*)&h, sizeof(h));

    //First write the dark matter particles
    for(int i=0; i < allIds.size(); i++)
    {
      if(allIds[i] >= DARKMATTERID)
      {
        //Set particle properties
        dark_particleV2 d;
        d.mass   = allPositions[i].w;
        d.pos[0] = allPositions[i].x;
        d.pos[1] = allPositions[i].y;
        d.pos[2] = allPositions[i].z;
        d.vel[0] = allVelocities[i].x;
        d.vel[1] = allVelocities[i].y;
        d.vel[2] = allVelocities[i].z;
        d.acc[0] = allAccelerations[i].x;
        d.acc[1] = allAccelerations[i].y;
        d.acc[2] = allAccelerations[i].z;
        d.epot   = allAccelerations[i].w;
        d.setID(allIds[i]);
        outputFile.write((char*)&d, sizeof(d));
      } //end if
    } //end i loop


    //Next write the star particles
    for(int i=0; i < allIds.size(); i++)
    {
      if(allIds[i] < DARKMATTERID)
      {
        //Set particle properties
        star_particleV2 s;
        s.mass = allPositions[i].w;
        s.pos[0] = allPositions[i].x;
        s.pos[1] = allPositions[i].y;
        s.pos[2] = allPositions[i].z;
        s.vel[0] = allVelocities[i].x;
        s.vel[1] = allVelocities[i].y;
        s.vel[2] = allVelocities[i].z;
        s.acc[0] = allAccelerations[i].x;
        s.acc[1] = allAccelerations[i].y;
        s.acc[2] = allAccelerations[i].z;
        s.epot   = allAccelerations[i].w;
        s.setID(allIds[i]);
        s.metals = 0;
        s.tform = 0;
        outputFile.write((char*)&s, sizeof(s));
     } //end if
    } //end i loop
    outputFile.close();
}

// GL: todo check whether we can try to read tipsy files with or without accelerations
// proposal: add new fileFormatVersion+=4 to indicate accelerations are included.
//           if fileformatVersion ==4 ==> original format=0 with accelerations
//           if fileformatVersion ==6 ==> original format=2 with accelerations
void tipsyIO::readFile(const MPI_Comm &mpiCommWorld,
                       std::vector<real4> &bodyPositions,
                       std::vector<real4> &bodyVelocities,
                       std::vector<real4> &bodyAccelerations,  // GL may be NULL, then assume it is not include din the file not read
                       std::vector<ullong> &bodiesIDs,
                       std::string fileName,
                       int rank,
                       int procs,
                       float &snapshotTime,
                       int reduce_bodies_factor,
                       const bool restart)

{
  /*
    If the input file is a single file then process 0 does the file reading and sends
    the data to the other processes.
    If we restart then it means each process as written it's own subset of data, and hence
    has to read in it's own data.

    Read in our custom version of the Tipsy file format, the most important change is that we store
    particle id on the location where previously the potential was stored.
    
    GL: next change, if (version & 4) accelerations + potential energies are stored as well
    
  */

  if(!restart && rank > 0)
  {
      ICRecv(0, rank, bodyPositions, bodyVelocities,  bodyAccelerations, bodiesIDs, mpiCommWorld);
  }
  else
  {
      char fullFileName[256];
      if(restart)   sprintf(fullFileName, "%s-%d", fileName.c_str(), rank);
      else          sprintf(fullFileName, "%s",   fileName.c_str());

      LOG("Trying to read file: %s \n", fullFileName);

      std::ifstream inputFile(fullFileName, std::ios::in | std::ios::binary);
      if(!inputFile.is_open())
      {
        LOG("Can't open input file \n");
        ::exit(0);
      }

      //Read Tipsy header
      dumpV2  h;
      inputFile.read((char*)&h, sizeof(h));
      int NTotal         = h.nbodies;
      int NDMparticles   = h.ndark;
      int NStarparticles = h.nstar;
      snapshotTime       = (float) h.time;
      assert(h.nsph == 0); //Bonsai does not support these particles
      assert(NTotal == (NDMparticles+NStarparticles));

      if(rank == 0) printf("File version: %d \n", h.version);
      int               fileFormatVersion = 0;
      if(h.version & 2) fileFormatVersion = 2;
      int               hasAcceleration = 0;
      if(h.version & 4) hasAcceleration = 1;

      ullong idummy;
      real4  positions;
      real4  velocity;
      real4  acceleration;


      //Rough divide
      uint        perProc = (NTotal / procs) /reduce_bodies_factor;
      if(restart) perProc =  NTotal          /reduce_bodies_factor; //don't subdivide when using restart
      bodyPositions.reserve(perProc+10);
      bodyVelocities.reserve(perProc+10);
      bodyAccelerations.reserve(perProc+10);
      bodiesIDs.reserve(perProc+10);
      perProc -= 1;

      //Start reading
      int procCntr            = 1;
      dark_particleV2 d;
      star_particleV2 s;

      for(int i=0; i < NTotal; i++)
      {
        if(i < NDMparticles)
        {
          inputFile.read((char*)&d, sizeof(d));
          positions.w       = d.mass;
          positions.x       = d.pos[0];
          positions.y       = d.pos[1];
          positions.z       = d.pos[2];
          velocity.x        = d.vel[0];
          velocity.y        = d.vel[1];
          velocity.z        = d.vel[2];
          velocity.w        = 0;
          idummy            = d.getID();

          //Force compatibility with older 32bit ID files by mapping the particle IDs
          if(fileFormatVersion == 0)
          {
            idummy    = d.getID_V1() + DARKMATTERID;
          }
        }
        else
        {
          inputFile.read((char*)&s, sizeof(s));
          positions.w       = s.mass;
          positions.x       = s.pos[0];
          positions.y       = s.pos[1];
          positions.z       = s.pos[2];
          velocity.x        = s.vel[0];
          velocity.y        = s.vel[1];
          velocity.z        = s.vel[2];
          velocity.w        = 0;
          idummy            = s.getID();

          //Force compatibility with older 32bit ID files by mapping the particle IDs
          if(fileFormatVersion == 0)
          {
            if(s.getID_V1() >= 100000000) idummy    = s.getID_V1() + BULGEID; //Bulge particles
            else                          idummy    = s.getID_V1();           //Disk  particles
          }
        }

        //Some input files have bugged z positions, ignore those particles and print a warning
        if(positions.z < -10e10)
        {
           fprintf(stderr," Removing particle %d because of Z is: %f \n", i, positions.z);
           continue;
        }

        //Reduce the number of particles, but increase the mass per particle by the reduction factor
        //We increase i by 1 before the check to retain compatibility with older Bonsai versions.
        if( (i+1) % reduce_bodies_factor == 0 ) positions.w *= reduce_bodies_factor;
        if( (i+1) % reduce_bodies_factor != 0 ) continue;

        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);



        if(!restart)
        {
    #ifdef USE_MPI
          if(bodyPositions.size() > perProc && procCntr != procs)
          {
            ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size(),mpiCommWorld);
            procCntr++;

            bodyPositions.clear();
            bodyVelocities.clear();
            bodiesIDs.clear();
          }
    #endif
        }
      }//end while

      inputFile.close();

      LOGF(stderr,"NTotal: %d\tper proc: ~ %d\tFor ourself: %d\n", NTotal, perProc, (int)bodiesIDs.size());
  }
}
