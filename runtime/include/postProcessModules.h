#pragma once

#include <cassert>

#ifdef USE_MPI
  #include <mpi.h>
#endif

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
    #define BULGESTARTID 100000000

    #define N_MESH      1024
    #define N_MESH_R    20
    #define N_MESH_PHI  128

    #define MIN_D 1.0
    #define MAX_D 10000.0

    float perProcRes [N_MESH][4*N_MESH];
    float combinedRes[N_MESH][4*N_MESH];

    float perProcResRPhi [N_MESH_PHI][2*N_MESH_R]; //First half is np, second half is mass
    float combinedResRPhi[N_MESH_PHI][2*N_MESH_R]; //First half is np, second half is mass

    const int procId, nProcs, nParticles;

    const double xscale, mscale, xmax;

    //For R-Phi computation
    const double Rmin;
    const double Rmax;
    const double pmin;
    const double pmax;

    double dp;
    double dR;


  private:

    double get_time2()
    {
      struct timeval Tvalue;
      struct timezone dummy;

      gettimeofday(&Tvalue,&dummy);
      return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
    }


    void reduceAndScale(float dataIn [N_MESH][4*N_MESH],
                        float dataOut[N_MESH][4*N_MESH],
                        float dataInRPhi [N_MESH_PHI][2*N_MESH_R],
                        float dataOutRPhi[N_MESH_PHI][2*N_MESH_R],
                        const float dscale)
    {
      //Sum over all processes
      double t0 = get_time2();
      #ifdef USE_MPI
        MPI_Reduce(dataIn,     dataOut,     4*(N_MESH*N_MESH),       MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(dataInRPhi, dataOutRPhi, 2*(N_MESH_PHI*N_MESH_R), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
      #else
        memcpy(dataOut,     dataIn,     4*(N_MESH*N_MESH)*sizeof(float));
        memcpy(dataOutRPhi, dataInRPhi, 2*(N_MESH_PHI*N_MESH_R)*sizeof(float));        
      #endif
        
      double t1 = get_time2();

      if(procId == 0)
      {
        //Scale results
//        float bg      = log10(MIN_D);

        //Density
//        for(int i=0;i<N_MESH;i++)
//        {
//           for(int j=0;j<N_MESH;j++)
//           {
//               //Normalize top view
//               if(dataOut[i][j]>0.0){
//                 dataOut[i][j] = log10(dataOut[i][j]*dscale);
//               }else{
//                 dataOut[i][j] = bg;
//               }
//               //Normalize front view
//               if(dataOut[i][N_MESH+j]>0.0){
//                 dataOut[i][N_MESH+j] = log10(dataOut[i][N_MESH+j]*dscale);
//               }else{
//                 dataOut[i][N_MESH+j] = bg;
//               }
//           }//for j
//        }//for i

        //R-Phi scaling
        for(int j=0;j<N_MESH_R;j++)
        {
          float R    = Rmin + (j+0.5)*dR;
          float SigR = 0.0;
          int nR     = 0;
          for(int i=0;i<N_MESH_PHI;i++){
              SigR += dataOutRPhi[i][N_MESH_R+j];
              nR   += dataOutRPhi[i][         j];
          }
          SigR    /= (2.*M_PI*R*dR);
          float ds =  2.*M_PI*R*dR/(float)N_MESH_PHI;

          for(int i=0;i<N_MESH_PHI;i++)
          {
            dataOutRPhi[i][j] = dataOutRPhi[i][N_MESH_R+j]/ds;
            dataOutRPhi[i][j]/= SigR;
          }
        }//for j


      }//if procId == 0
      //if(procId == 0) fprintf(stderr,"MPI Reduce: %lg Scale took: %lg \n", t1-t0, get_time2()-t1);
    }//reduceAndScale

    void writeData(const char *fileName,
                   float data    [N_MESH]    [4*N_MESH],
                   float dataRPhi[N_MESH_PHI][2*N_MESH_R],
                   const double tSim)
    {

#if 1

      ofstream out;
      out.open(fileName);
      if(out.is_open())
      {
        //Write the header for
        int flag    = -1;
        double tmpD = 0.0;
        float  tmpF = 0.0f;

        out.write((char*)&flag, sizeof(int));
        int n = N_MESH;
        out.write((char*)&n, sizeof(int));
        out.write((char*)&n, sizeof(int));
        int Nz = 1;
        out.write((char*)&Nz,   sizeof(int));
        out.write((char*)&tSim, sizeof(double));
        out.write((char*)&tmpD, sizeof(double));
        out.write((char*)&tmpD, sizeof(double));
        out.write((char*)&tmpD, sizeof(double));
        out.write((char*)&tmpD, sizeof(double));
        out.write((char*)&tmpD, sizeof(double));

        //The actual data
        for(int i=0;i<N_MESH;i++){
          for(int j=0;j<N_MESH;j++){

            //Density
            out.write((char*)&data[i][N_MESH * 0 + j], sizeof(float));
            out.write((char*)&data[i][N_MESH * 1 + j], sizeof(float));
            out.write((char*)&data[i][N_MESH * 2 + j], sizeof(float));
            out.write((char*)&data[i][N_MESH * 3 + j], sizeof(float));

            //R-phi
	          if(i <  N_MESH_PHI && j < N_MESH_R)
            {
              out.write((char*)&dataRPhi[i][j], sizeof(float));
            }
            else
            {
              out.write((char*)&tmpF, sizeof(float));
            }
          }//for j
        }//for i
        out.close();

      }//out.is_open

#else
      //Old ASCII format, does not contain velocities
      char filename2[256];
      sprintf(filename2, "TestJB.txt");
      FILE *dump = NULL;
            dump = fopen(filename2, "w");

      if(dump)
      {
        //Write some info
        fprintf(dump, "#Tsim= %f\n", tSim);
        fprintf(dump, "#X Y TOP FRONT R-Phi\n", tSim);

        for(int i=0;i<N_MESH;i++){
            for(int j=0;j<N_MESH;j++){
                if(i <  N_MESH_PHI && j < N_MESH_R)
                {
                  fprintf(dump, "%d\t%d\t%f\t%f\t%f\n", i, j, data[i][j],data[i][j+2*N_MESH], dataRPhi[i][j]);
                }
                else
                {
                  fprintf(dump, "%d\t%d\t%f\t%f\t-\n", i, j, data[i][j],data[i][j+2*N_MESH]);
                }
            }//j
        }//i
        fclose(dump);
      }//if dump
#endif
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
            xscale(_xscale), mscale(_mscale), xmax(_xmax),
            Rmin(0.0), Rmax(20.0), pmin(-180), pmax(180)
    {
      //Reset the buffers
      for(uint i=0; i < N_MESH; i++)
      {
        for(uint j=0; j < N_MESH; j++)
        {
          perProcRes  [i][N_MESH*0 + j] = 0.0;
          perProcRes  [i][N_MESH*1 + j] = 0.0;
          perProcRes  [i][N_MESH*2 + j] = 0.0;
          perProcRes  [i][N_MESH*3 + j] = 0.0;


          if(i < N_MESH_PHI&& j < 2*N_MESH_R)
          {
            perProcResRPhi [i][j] = 0.0;
          }
        }//j
      }//i

      dp = (pmax - pmin)/(double)N_MESH_PHI;
      dR = (Rmax - Rmin)/(double)N_MESH_R;

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
            perProcRes[x][y+0*N_MESH] += positions[i].w*mscale;
            perProcRes[x][y+1*N_MESH] += velocities[i].x*velocities[i].x+velocities[i].y*velocities[i].y;
          }

          //Front view
          if(x<N_MESH && z<N_MESH && x>0 && z>0)
          {
            perProcRes[x][z+2*N_MESH] += positions[i].w*mscale;
            perProcRes[x][z+3*N_MESH] += velocities[i].x*velocities[i].x+velocities[i].z*velocities[i].z;
          }//for i

          //R-Phi projection
          double r2   = positions[i].x*positions[i].x + positions[i].y*positions[i].y;
          double r    = sqrt(r2);
          double sinp = positions[i].y/r;
          double cosp = positions[i].x/r;
          double phi;
          if(positions[i].y>0.0){
              phi = acos(cosp)*180/M_PI;
          }else{
              phi = -acos(cosp)*180/M_PI;
          }
          x = (int)floor((phi-pmin)/dp);
          y = (int)floor((r-Rmin)/dR);

          if(x<N_MESH_PHI && y<N_MESH_R && x>=0 && y>=0)
          {
            perProcResRPhi[x][         y] += 1;
            perProcResRPhi[x][N_MESH_R+y] += positions[i].w;
          }

        }//if ID < DMSTARTID
      }//for i < nParticles
      double t2 = get_time2();

      //Combine the results for all processes, top view
      reduceAndScale(perProcRes, combinedRes, perProcResRPhi, combinedResRPhi, dscale);



      double t3 = get_time2();
      //Dump top view results
      char fileName[256];
      sprintf(fileName,"%s-TopFront-%f", baseFilename, time);
      if(procId == 0) writeData(fileName, combinedRes, combinedResRPhi, tscale*time);
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
          DRS, DAS, DZS,  // dispersion
          ZRMS};

    float perProcRes[nItems][3*iMax] ;

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
      float comProcRes[nItems][iMax*3] ;

      float Rs[iMax*3];
      float Qs[iMax*3],   Gam[iMax*3], mX[iMax*3];
      float Omgs[iMax*3], kapps[iMax*3];
      float m[iMax*3],    Mass[iMax*3];

      float VelUnit = UnitVelocity/VELOCITY_KMS_CGS;

      //Init
      for(int i=0; i<iMax; i++){
          //Rs[i]           = RrotMin + (i+0.5)*dR;
          Qs[i]           = Gam[i]        = mX[i]  = 0.0;
          Omgs[i]         = kapps[i]      = 0.0;
          Mass[i]         = 0.0;
      }
      for(int j=0; j < 3; j++){
        for(int i=0; i<iMax; i++){
            Rs[i+j*iMax]  = RrotMin + (i+0.5)*dR;
      }}
      for(int j=0; j<nItems; j++)
        for(int i=0; i<iMax*3; i++)
          comProcRes[j][i] = 0.0;

     double t0 = get_time();
     #ifdef USE_MPI
       //MPI Reduce: Sum results over all processes store in procId == 0
       MPI_Reduce(perProcRes, comProcRes, nItems*iMax*3, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
     #else
       memcpy(comProcRes, perProcRes, nItems*iMax*3*sizeof(float));
     #endif
     double t1 = get_time();

     if(procId == 0)
     {
       for(int j=0; j < 3; j++)
       {
         const int offset = j*iMax;
          // average
          Mass[offset+0] = comProcRes[SIGS][offset+0];
          for(int i=0; i<iMax; i++)
          {
            if(i>0){
                Mass[offset+i] = Mass[offset+i-1] + comProcRes[SIGS][offset+i];
            }

            if(comProcRes[NS][offset+i] != 0)
            {
              comProcRes[SIGS][offset+i] /= (2.0*PI*Rs[offset+i]*dR);
              comProcRes[VRS] [offset+i] /= (float)comProcRes[NS][offset+i];
              comProcRes[VAS] [offset+i] /= (float)comProcRes[NS][offset+i];
              comProcRes[VZS] [offset+i] /= (float)comProcRes[NS][offset+i];
              comProcRes[DRS] [offset+i]  = sqrt(MAX(comProcRes[DRS][offset+i]/(float)comProcRes[NS][offset+i] - SQ(comProcRes[VRS][offset+i]),0.0));
              comProcRes[DAS] [offset+i]  = sqrt(MAX(comProcRes[DAS][offset+i]/(float)comProcRes[NS][offset+i] - SQ(comProcRes[VAS][offset+i]),0.0));
              comProcRes[DZS] [offset+i]  = sqrt(MAX(comProcRes[DZS][offset+i]/(float)comProcRes[NS][offset+i] - SQ(comProcRes[VZS][offset+i]),0.0));
              Omgs            [offset+i]  = comProcRes[VAS][offset+i]/Rs[offset+i];
              comProcRes[ZRMS][offset+i]  = sqrt(comProcRes[ZRMS][offset+i]/(float)comProcRes[NS][offset+i]);
            }
          }//for i

          for(int i=1; i<iMax-1; i++)
          {
            kapps[offset+i] = sqrt(MAX(0.5*Rs[offset+i]*((SQ(Omgs[offset+i+1])-SQ(Omgs[offset+i-1]))/dR) + 4.0*SQ(Omgs[offset+i]),0.0));
            Gam[offset+i]   = -(Rs[offset+i]/Omgs[offset+i])*0.5*(Omgs[offset+i+1]-Omgs[offset+i-1])/dR;
          }
          kapps[offset+0]      = 2.0*Omgs[offset+0];
          kapps[offset+iMax-1] = kapps[offset+iMax-2];
          Gam[offset+0]        = Gam[offset+1];
          Gam[offset+iMax-1]   = Gam[offset+iMax-2];

          for(int i=0; i<iMax; i++)
          {
            m[offset+i]                = kapps[offset+i]*kapps[offset+i]/(GravConst*comProcRes[SIGS][offset+i]);
            Qs[offset+i]               = comProcRes[DRS][offset+i]*kapps[offset+i]/(3.36*GravConst*comProcRes[SIGS][offset+i]);
            mX[offset+i]               = SQ(kapps[offset+i])*Rs[offset+i]/(2.0*PI*GravConst*comProcRes[SIGS][offset+i])/4.0;
            comProcRes[VRS][offset+i] *= VelUnit;
            comProcRes[VAS][offset+i] *= VelUnit;
            comProcRes[VZS][offset+i] *= VelUnit;
            comProcRes[DRS][offset+i] *= VelUnit;
            comProcRes[DAS][offset+i] *= VelUnit;
            comProcRes[DZS][offset+i] *= VelUnit;
            kapps[offset+i]           *= VelUnit;
            Omgs[offset+i]            /= UnitTime;
            comProcRes[SIGS][offset+i]*= SDUnit;
            Mass[offset+i]            *= 2.33e9;//UnitMass;
          }
        }//For j
        double t2 = get_time();

        char fileNameOut2[512];
        sprintf(fileNameOut2,"%s-%f", fileNameOut, tsim);

        std::ofstream out(fileNameOut2);
        if(out.is_open())
        {
          out.setf(std::ios::scientific);
          out.precision(6);

          out << "# T = " << treal << " (Gyr) \n";
          out << "#RS(D+B) Vas Drs Das Dzs Omg Kapp Q Gam mX Sigs Mass m Zrms RS(B) Vas Drs Das Dzs Sigs Mass RS(D) Vas Drs Das Dzs Sigs Mass\n";

          for(int i=0; i<iMax; i++)
          {
              //Disk + Bulge results
              out << Rs[i]              << "  " << comProcRes[VAS][i] << "  "  // 1,2
                  << comProcRes[DRS][i] << "  " << comProcRes[DAS][i] << "  " << comProcRes[DZS][i] << "  "// 3,4,5
                  << Omgs[i]            << "  " << kapps[i]           << "  " << Qs[i] << "  " // 6,7,8
                  << Gam[i]             << "  " << mX[i]              << "  " << comProcRes[SIGS][i] << "  " //9,10,11
                  << Mass[i]            << "  " << m[i]               << "  " << comProcRes[ZRMS][i] << " " << comProcRes[NS][i]  << "  "  // 12,13,14
              //Bulge only
                  << Rs[i+iMax]               << "  " << comProcRes[VAS][i+iMax] << "  "  // 15,16
                  << comProcRes[DRS][i+iMax]  << "  " << comProcRes[DAS][i+iMax] << "  " << comProcRes[DZS][i+iMax] << "  "// 17,18,19
                  << comProcRes[SIGS][i+iMax] << "  " << Mass[i+iMax]            << "  "  // 20,21
              //Disk only
                  << Rs[i+2*iMax]               << "  " << comProcRes[VAS][i+2*iMax] << "  "  // 22,23
                  << comProcRes[DRS][i+2*iMax]  << "  " << comProcRes[DAS][i+2*iMax] << "  " << comProcRes[DZS][i+2*iMax] << "  "// 24,25,26
                  << comProcRes[SIGS][i+2*iMax] << "  " << Mass[i+2*iMax]            << std::endl;  // 27,28
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
          for(int i=0; i<3*iMax; i++)
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
          if(IDs[j] >= 0 && IDs[j] < DMSTARTID)
          {
            //Bluge+disk particles
            double R  =  sqrt(SQ(positions[j].x) + SQ(positions[j].y));
            double z  =  positions[j].z;
            double vr =  velocities[j].x*positions[j].x/R + velocities[j].y*positions[j].y/R;
            double va = -velocities[j].x*positions[j].y/R + velocities[j].y*positions[j].x/R;
            double vz =  velocities[j].z;
            if( R <= RrotMin || R >= RrotEnd )    continue;
            int i     = (int)((R-RrotMin)/dR);

            //Store the different properties
            //First disk+bulge data
            int offset = 0;
            perProcRes[NS  ][offset+i] += 1;
            perProcRes[SIGS][offset+i] += (float)positions[j].w;
            perProcRes[VRS ][offset+i] += (float)vr;
            perProcRes[VAS ][offset+i] += (float)va;
            perProcRes[VZS ][offset+i] += (float)vz;
            perProcRes[DRS ][offset+i] += SQ((float)vr);
            perProcRes[DAS ][offset+i] += SQ((float)va);
            perProcRes[DZS ][offset+i] += SQ((float)vz);
            perProcRes[ZRMS][offset+i] += SQ(z);

            offset = iMax;
            //Bulge only
            if(IDs[j] >= BULGESTARTID && IDs[j] < DMSTARTID)
            {
              perProcRes[NS  ][offset+i] += 1;
              perProcRes[SIGS][offset+i] += (float)positions[j].w;
              perProcRes[VRS ][offset+i] += (float)vr;
              perProcRes[VAS ][offset+i] += (float)va;
              perProcRes[VZS ][offset+i] += (float)vz;
              perProcRes[DRS ][offset+i] += SQ((float)vr);
              perProcRes[DAS ][offset+i] += SQ((float)va);
              perProcRes[DZS ][offset+i] += SQ((float)vz);
              perProcRes[ZRMS][offset+i] += SQ(z);
            }//Bulge
            offset = 2*iMax;
            //Disk only
            if(IDs[j] >= 0 && IDs[j] < BULGESTARTID)
            {
              perProcRes[NS  ][offset+i] += 1;
              perProcRes[SIGS][offset+i] += (float)positions[j].w;
              perProcRes[VRS ][offset+i] += (float)vr;
              perProcRes[VAS ][offset+i] += (float)va;
              perProcRes[VZS ][offset+i] += (float)vz;
              perProcRes[DRS ][offset+i] += SQ((float)vr);
              perProcRes[DAS ][offset+i] += SQ((float)va);
              perProcRes[DZS ][offset+i] += SQ((float)vz);
              perProcRes[ZRMS][offset+i] += SQ(z);
            }//Disk
          }//if(IDs[j] >= 0 && IDs[j] < DMSTARTID)
        }//for nParticles

        Analysis(baseFilename, procId, treal, tsim);

      }//DISKSTATS func
}; //DISKSTATS struct

