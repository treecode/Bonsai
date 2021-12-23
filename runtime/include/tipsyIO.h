/*
 * tipsyIO.h
 *
 *  Created on: Sep 6, 2016
 *      Author: jbedorf
 */

#ifndef TIPSYIO_H_
#define TIPSYIO_H_


#ifdef USE_MPI
   #include <mpi.h>
#else
    typedef int MPI_Comm;
#endif

#if 1
    #include "log.h"
#else
    #define LOG printf
    #define LOGF fprintf
#endif

#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>

#include "cuda_runtime.h"

typedef unsigned int uint;

typedef float real;
typedef float2             real2;
typedef float4 real4;
typedef unsigned int       uint;
typedef unsigned long long ullong; //ulonglong1

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000


class tipsyIO
{

private:
/*
 *
 * Tipsy file structure, binary compatible with the original Tipsy files
 * although some fields have been repurposed to write more header information and
 * particle IDS for each particle.
 */

    #define MAXDIM 3
    #define forever for(;;)

    typedef float Real;

    struct gas_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real rho;
        Real temp;
        Real hsmooth;
        Real metals ;
        Real phi ;
    } ;

    //struct gas_particle *gas_particles;

    struct dark_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real acc[MAXDIM];
        Real epot;
        Real eps;
        int phi ;
    public:
      int  getID() const {return phi;}
      void setID(int ID) {  phi = ID; }
    } ;

    struct star_particle {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real acc[MAXDIM];
        Real epot;
        Real metals ;
        Real tform ;
        Real eps;
        int phi ;
    public:
      int  getID() const {return phi;}
      void setID(int ID) {  phi = ID; }
    } ;


    //V2 structures use 64 bit integers for particle storage
    //otherwise they take up the same space for compatibility

    struct dark_particleV2 {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real acc[MAXDIM];
        Real epot;
      private:
        int _ID[2]; //replaces phi and eps
      public:
        unsigned long long getID() const {return *(unsigned long long*)_ID;}
        void setID(unsigned  long long ID) { *(unsigned long long*)_ID = ID; }
        int getID_V1() const {return _ID[1];}
    //    Real eps;
    } ;
    struct star_particleV2 {
        Real mass;
        Real pos[MAXDIM];
        Real vel[MAXDIM];
        Real acc[MAXDIM];
        Real epot;
        Real metals ;
        Real tform ;
    private:
      int _ID[2]; //replaces phi and eps
    public:
      unsigned long long  getID() const {return *(unsigned  long long*)_ID;}
      void setID(unsigned long long ID) { *(unsigned  long long*)_ID = ID; }
      int getID_V1() const {return _ID[1];}
    //    Real eps;
    //    int ID; //replaces phi and eps
    } ;


    struct dump {
        double time ;
        int nbodies ;
        int ndim ;
        int nsph ;
        int ndark ;
        int nstar ;
    } ;

    struct dumpV2 {
        double time ;
        int nbodies ;
        int ndim ;
        int nsph ;
        int ndark ;
        int nstar ;
        int version;
    } ;


    typedef struct dump header ;



    void ICSend(int destination, real4 *bodyPositions, real4 *bodyVelocities, real4 *bodyAccelerations,  ullong *bodiesIDs, int toSend, const MPI_Comm &mpiCommWorld);
    void ICRecv(int recvFrom, int procId, std::vector<real4> &bodyPositions, std::vector<real4> &bodyVelocities, std::vector<real4> &bodyAccelerations,  std::vector<ullong> &bodiesIDs, const MPI_Comm &mpiCommWorld);


public:
/******************************************************************/
/*      Function to read/write the legacy Tipsy format            */
/*                                                                */
/******************************************************************/


/*
 * If 'perProcess' is true then each process writes it's own file
 * Otherwise each process sends it's data to process 0 which then writes a single file
 */
void writeFile(real4 *bodyPositions,
                               real4 *bodyVelocities,
                               real4 *bodyAccelerations,
                               ullong* bodyIds,
                               int n,
                               std::string fileName,
                               float time,
                               const int rank,
                               const int nProcs,
                               const MPI_Comm &mpiCommWorld,
                               bool perProcess);



/*
 * If 'restart' is true then each process will read it'so own file
 * Otherwise each process receives it's data from process 0 which
 * reads he full file and distributes it to the other processes
 */
void readFile(const MPI_Comm &mpiCommWorld,
                              std::vector<real4> &bodyPositions,
                              std::vector<real4> &bodyVelocities,
                              std::vector<real4> &bodyAccelerations,
                              std::vector<ullong> &bodiesIDs,
                              std::string fileName,
                              int rank,
                              int procs,
                              float &snapshotTime,
                              int reduce_bodies_factor,
                              const bool restart);

};


#endif /* TIPSYIO_H_ */
