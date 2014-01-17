#pragma once

#include <cassert>
#include <mpi.h>
#include <vector>
#include <sys/time.h>

struct DENSITY
{
  private:

    #define G_CONST   6.672e-8
    #define M_SUN     1.989e33
    #define PARSEC    3.08567802e18
    #define ONE_YEAR  3.1558149984e7
    #define DMSTARTID 200000000

    #define N_MESH 200

    #define MIN_D 1.0
    #define MAX_D 10000.0

    float perProcRes [N_MESH][2*N_MESH];
    float combinedRes[N_MESH][2*N_MESH];

    const int procId, nProcs, nParticles;

    const double xscale, mscale, xmax;


  private:

    double get_time2()
    {
      struct timeval Tvalue;
      struct timezone dummy;

      gettimeofday(&Tvalue,&dummy);
      return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
    }


    void reduceAndScale(float dataIn [N_MESH][2*N_MESH],
                        float dataOut[N_MESH][2*N_MESH],
                        const float dscale)
    {
      //Sum over all processes
      double t0 = get_time2();
      MPI_Reduce(dataIn, dataOut, 2*(N_MESH*N_MESH), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      double t1 = get_time2();

      if(procId == 0)
      {
        //Scale results
        float bg      = log10(MIN_D);

        for(int i=0;i<N_MESH;i++)
        {
           for(int j=0;j<N_MESH;j++){
               //Normalize top view
               if(dataOut[i][j]>0.0){
                 dataOut[i][j] = log10(dataOut[i][j]*dscale);
               }else{
                 dataOut[i][j] = bg;
               }
               //Normalize front view
               if(dataOut[i][N_MESH+j]>0.0){
                 dataOut[i][N_MESH+j] = log10(dataOut[i][N_MESH+j]*dscale);
               }else{
                 dataOut[i][N_MESH+j] = bg;
               }
           }//for j
        }//for i
      }//if procId == 0
      //if(procId == 0) fprintf(stderr,"MPI Reduce: %lg Scale took: %lg \n", t1-t0, get_time2()-t1);
    }//reduceAndScale

    void writeData(const char *fileName, float data[N_MESH][2*N_MESH],
                   const float tSim)
    {
      FILE *dump = NULL;
            dump = fopen(fileName, "w");

      if(dump)
      {
        //Write some info
        fprintf(dump, "#Tsim= %f\n", tSim);
        fprintf(dump, "#X Y TOP FRONT\n", tSim);

        for(int i=0;i<N_MESH;i++){
            for(int j=0;j<N_MESH;j++){
                fprintf(dump, "%d\t%d\t%f\t%f\n", i, j, data[i][j],data[i][j+N_MESH]);
            }
        }
        fclose(dump);
      }//if dump
    }//writeData
 
  public:

    //Create two density plots and write results to file
    DENSITY(const int _procId, const int _nProc, const int _n,
            const float4 *positions,
            const float4 *velocities,
            const int    *IDs,
            double _xscale, double _mscale, double _xmax,
            const char *baseFilename,
            const double time) :
            procId(_procId), nProcs(_nProc), nParticles(_n),
            xscale(_xscale), mscale(_mscale), xmax(_xmax)
    {
      //Reset the buffers
      for(uint i=0; i < N_MESH; i++)
      {
        for(uint j=0; j < N_MESH; j++)
        {
          perProcRes  [i][j]        = 0.0;
          perProcRes  [i][N_MESH+j] = 0.0;
          combinedRes [i][j]        = 0.0;
        }
      }

      double xmin = -xmax;
      double ymin = -xmax;
      double ymax =  xmax;
      double dx   = (xmax - xmin)/N_MESH;
      double dy   = (ymax - ymin)/N_MESH;


      double x      = xscale*1e3*PARSEC;
      double tmp    = x*x*x/(G_CONST*mscale*M_SUN);
      double tscale = sqrt(tmp)*1e-6/ONE_YEAR;


             tmp    = 1.e3*dx*xscale;
      double dscale = 1./(tmp*tmp);

      double t1 = get_time2();
      //Walk through the particles and sum the densities
      for(uint i = 0; i < nParticles; i++)
      {
        //Only use star particles
        if(IDs[i] < DMSTARTID)
        {
          //Top view
          int x = (int)floor((positions[i].x*xscale-xmin)/dx);
          int y = (int)floor((positions[i].y*xscale-ymin)/dy); //Topview
          int z = (int)floor((positions[i].z*xscale-ymin)/dy); //Front view

          //Top view
          if(x<N_MESH && y<N_MESH && x>0 && y>0)
          {
            perProcRes[x][y] += positions[i].w*mscale;
          }

          //Front view
          if(x<N_MESH && z<N_MESH && x>0 && z>0)
          {
            perProcRes[x][z+N_MESH] += positions[i].w*mscale;
          }//for i
        }//if ID < DMSTARTID
      }//for i < nParticles
      double t2 = get_time2();

      //Combine the results for all processes, top view
      reduceAndScale(perProcRes, combinedRes, dscale);

      double t3 = get_time2();
      //Dump top view results
      char fileName[256];
      sprintf(fileName,"%s-TopFront-%f", baseFilename, time);
      if(procId == 0) writeData(fileName, combinedRes, tscale*time);
      double t3b = get_time2();
      //if(procId == 0) fprintf(stderr, "Compute took: %lg Write took: %lg \n", t2-t1, t3b-t3);
    }//Function
}; //Struct


struct DISKSTATS
{
  private:

    //Some constants
    #define ONE_YEAR  3.1558149984e7
    #define VELOCITY_KMS_CGS    (1.e+5) // [km/s] -> [cm/s]
    #define KPC_CGS     (PC_CGS*1.e+3)      // [cm]
    #define PC_CGS      (3.08568025e+18)    // [cm]
    #define MSUN_CGS    (1.98892e+33)       // [g]
    #define GRAVITY_CONSTANT_CGS 6.6725985e-8     // [dyne m^2/kg^2] = [cm^3/g/s^2]
    #define PI (3.1415926535897932384626433832795)

    #define SQ(x)           ((x)*(x))
    #define CUBE(x)         ((x)*(x)*(x))
    #define MAX(x,y)        (((x)>(y))?(x):(y))

    #define DMSTARTID    200000000
    #define BULGESTARTID 100000000


    const int procId, nProcs, nParticles;

    const double xscale, mscale;

    #define iMax 600
    //nItems should match the items in the enum below
    #define nItems  9

    //Enum contains variables Ns, Sigs, Vrs, Vas, Vzs, Drs, Das, Dzs, zrms
    enum {NS = 0, SIGS,
          VRS, VAS, VZS,  // mean speed
          DRS, DAS,DZS,   // dispersion
          ZRMS};

    float perProcRes[nItems][iMax] ;

    float  RrotEnd;
    float  RrotMin;
    float  dR;
    double UnitVelocity, UnitLength, UnitTime, GravConst, SDUnit, UnitMass;


  private:

    double get_time()
    {
      struct timeval Tvalue;
      struct timezone dummy;

      gettimeofday(&Tvalue,&dummy);
      return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
    }

    void Analysis(const char *fileNameOut, const int procId, const double treal, const double tsim)
    {
      float comProcRes[nItems][iMax] ;

      float Rs[iMax];
      float Qs[iMax],   Gam[iMax], mX[iMax];
      float Omgs[iMax], kapps[iMax];
      float m[iMax],    Mass[iMax];

      float VelUnit = UnitVelocity/VELOCITY_KMS_CGS;

      //Init
      for(int i=0; i<iMax; i++){
          Rs[i]           = RrotMin + (i+0.5)*dR;
          Qs[i]           = Gam[i]        = mX[i]  = 0.0;
          Omgs[i]         = kapps[i]      = 0.0;
          Mass[i]         = 0.0;
      }
      for(int j=0; j<nItems; j++)
        for(int i=0; i<iMax; i++)
          comProcRes[j][i] = 0.0;

     double t0 = get_time();
     //MPI Reduce: Sum results over all processes store in procId == 0
     MPI_Reduce(perProcRes, comProcRes, nItems*iMax, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
     double t1 = get_time();

     if(procId == 0)
     {
        // average
        Mass[0] = comProcRes[SIGS][0];
        for(int i=0; i<iMax; i++)
        {
          if(i>0){
              Mass[i] = Mass[i-1] + comProcRes[SIGS][i];
          }

          if(comProcRes[NS][i] != 0)
          {
            comProcRes[SIGS][i]/= (2.0*PI*Rs[i]*dR);
            comProcRes[VRS][i] /= (float)comProcRes[NS][i];
            comProcRes[VAS][i] /= (float)comProcRes[NS][i];
            comProcRes[VZS][i] /= (float)comProcRes[NS][i];
            comProcRes[DRS][i]  = sqrt(MAX(comProcRes[DRS][i]/(float)comProcRes[NS][i] - SQ(comProcRes[VRS][i]),0.0));
            comProcRes[DAS][i]  = sqrt(MAX(comProcRes[DAS][i]/(float)comProcRes[NS][i] - SQ(comProcRes[VAS][i]),0.0));
            comProcRes[DZS][i]  = sqrt(MAX(comProcRes[DZS][i]/(float)comProcRes[NS][i] - SQ(comProcRes[VZS][i]),0.0));
            Omgs[i]             = comProcRes[VAS][i]/Rs[i];
            comProcRes[ZRMS][i] = sqrt(comProcRes[ZRMS][i]/(float)comProcRes[NS][i]);
          }
        }//for i

        for(int i=1; i<iMax-1; i++)
        {
            kapps[i] = sqrt(MAX(0.5*Rs[i]*((SQ(Omgs[i+1])-SQ(Omgs[i-1]))/dR) + 4.0*SQ(Omgs[i]),0.0));
            Gam[i]   = -(Rs[i]/Omgs[i])*0.5*(Omgs[i+1]-Omgs[i-1])/dR;
        }
        kapps[0]      = 2.0*Omgs[0];
        kapps[iMax-1] = kapps[iMax-2];
        Gam[0]        = Gam[1];
        Gam[iMax-1]   = Gam[iMax-2];

        for(int i=0; i<iMax; i++)
        {
          m[i]                = kapps[i]*kapps[i]/(GravConst*comProcRes[SIGS][i]);
          Qs[i]               = comProcRes[DRS][i]*kapps[i]/(3.36*GravConst*comProcRes[SIGS][i]);
          mX[i]               = SQ(kapps[i])*Rs[i]/(2.0*PI*GravConst*comProcRes[SIGS][i])/4.0;
          comProcRes[VRS][i] *= VelUnit;
          comProcRes[VAS][i] *= VelUnit;
          comProcRes[VZS][i] *= VelUnit;
          comProcRes[DRS][i] *= VelUnit;
          comProcRes[DAS][i] *= VelUnit;
          comProcRes[DZS][i] *= VelUnit;
          kapps[i]           *= VelUnit;
          Omgs[i]            /= UnitTime;
          comProcRes[SIGS][i]*= SDUnit;
          Mass[i]            *= 2.33e9;//UnitMass;
        }
        double t2 = get_time();

        char fileNameOut2[512];
        sprintf(fileNameOut2,"%s-%f", fileNameOut, tsim);

        std::ofstream out(fileNameOut2);
        if(out.is_open())
        {
          out.setf(std::ios::scientific);
          out.precision(6);

          out << "# T = " << treal << " (Gyr) \n";
          out << "#RS Vas Drs Das Dzs Omg Kapp Q Gam mX Sigs Mass m Zrms\n";

          for(int i=0; i<iMax; i++){
              out << Rs[i]              << "  " << comProcRes[VAS][i] << "  "  // 1,2
                  << comProcRes[DRS][i] << "  " << comProcRes[DAS][i] << "  " << comProcRes[DZS][i] << "  "// 3,4,5
                  << Omgs[i]            << "  " << kapps[i]           << "  " << Qs[i] << "  " // 6,7,8
                  << Gam[i]             << "  " << mX[i]              << "  " << comProcRes[SIGS][i] << "  " //9,10,11
                  << Mass[i]            << "  " << m[i]               << "  " << comProcRes[ZRMS][i] << " " << comProcRes[NS][i] << std::endl;  // 12,13,14
          }
          out.close();
        }
        else
        {
          fprintf(stderr,"Failed to open output file for disk-stats: %s \n", fileNameOut);
        }

        double t3 = get_time();

        //fprintf(stderr,"Timing: Reduce: %lg\tComp: %lg\tWrite: %lg\n",t1-t0,t2-t1,t3-t2);

      } //if procId == 0
  }//end analysis


  public:

    //Create two density plots and write results to file
    DISKSTATS(const int _procId, const int _nProc, const int _n,
              const float4 *positions,
              const float4 *velocities,
              const int    *IDs,
              double _xscale, double _mscale,
              const char *baseFilename,
              const double tsim) :
              procId(_procId), nProcs(_nProc), nParticles(_n),
              xscale(_xscale), mscale(_mscale)
      {
        RrotEnd    = 30.0;
        RrotMin    = 0.0;
        dR         = (RrotEnd - RrotMin)/iMax;

        for(int j=0; j<nItems; j++)
          for(int i=0; i<iMax; i++)
            perProcRes[j][i] = 0.0;

        GravConst  = 1.0;
        UnitLength = xscale*KPC_CGS;                      //[kpc]->[cm]
        UnitMass   = mscale*MSUN_CGS;
        SDUnit     = 2.3e+9/1.e6;

        double tmp   = CUBE(UnitLength)/(GRAVITY_CONSTANT_CGS*UnitMass);
        UnitTime     = sqrt(tmp);                       //[s]
        UnitVelocity = UnitLength/UnitTime;             //[cm/s]
        double treal = 1e-9*tsim*UnitTime/ONE_YEAR;

        //Process the particles
        for(int j=0; j < nParticles; j++)
        {
          //if(IDs[j] >= 0 && IDs[j] < BULGESTARTID)
          if(IDs[j] >= 0 && IDs[j] < DMSTARTID)
          {
            double R  =  sqrt(SQ(positions[j].x) + SQ(positions[j].y));
            double z  =  positions[j].z;
            double vr =  velocities[j].x*positions[j].x/R + velocities[j].y*positions[j].y/R;
            double va = -velocities[j].x*positions[j].y/R + velocities[j].y*positions[j].x/R;
            double vz =  velocities[j].z;
            if( R <= RrotMin || R >= RrotEnd )    continue;
            int i     = (int)((R-RrotMin)/dR);

            perProcRes[NS  ][i] += 1;
            perProcRes[SIGS][i] += (float)positions[j].w;
            perProcRes[VRS ][i] += (float)vr;
            perProcRes[VAS ][i] += (float)va;
            perProcRes[VZS ][i] += (float)vz;
            perProcRes[DRS ][i] += SQ((float)vr);
            perProcRes[DAS ][i] += SQ((float)va);
            perProcRes[DZS ][i] += SQ((float)vz);
            perProcRes[ZRMS][i] += SQ(z);
          }
        }//for nParticles

        Analysis(baseFilename, procId, treal, tsim);

      }//DISKSTATS func
}; //DISKSTATS struct

