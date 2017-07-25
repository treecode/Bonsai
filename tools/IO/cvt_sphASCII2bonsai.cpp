#include "BonsaiIO.h"
#include "IDType.h"
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>


typedef struct real4
{
  float x,y,z,w;
} real4;

typedef struct real2
{
  float x,y,z,w;
} real2;

typedef float float4[4];
typedef float float3[3];
typedef float float2[2];  

//The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000


static IDType lGetIDType (const long long id)
{
  IDType ID;
  ID.setID(id);
  ID.setType(3);     /* Everything is Dust until told otherwise */
  if(id >= DISKID  && id < BULGEID)       
  {
    ID.setType(2);  /* Disk */
    ID.setID(id - DISKID);
  }
  else if(id >= BULGEID && id < DARKMATTERID)  
  {
    ID.setType(1);  /* Bulge */
    ID.setID(id - BULGEID);
  }
  else if (id >= DARKMATTERID)
  {
    ID.setType(0);  /* DM */
    ID.setID(id - DARKMATTERID);
  }
  return ID;
};


 //#X Y Z MASS VX VY VZ RHO H U 
 void readPhantomSPHFile(std::vector<real4>    &bodyPositions, 
                        std::vector<real4>    &bodyVelocities, 
                        std::vector<IDType>   &bodiesIDs, 
                        std::vector<real2>    &bodyDensRho, 
                        std::vector<real4>    &bodyDrvt, 
                        std::vector<real4>    &bodyHydro, 
                        std::string fileName,
                        const int reduceFactor) {
  
    bodyPositions.clear();
  
    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());

    std::cerr << "Trying to read file: " << fullFileName << std::endl;

    std::ifstream inputFile(fullFileName, std::ios::in);
    
    if(!inputFile.is_open())
    {
      std::cerr << "Can't open input file \n";
      exit(0);
    }
    
    //Skip the  header line
    std::string tempLine;
    std::getline(inputFile, tempLine);
   

    int pid  = 0;
    real4       positions;
    real4       velocity;
    real4       drvt = {0,0,0,0};
    real4       hydro = {0,0,0,0};
    real2       rhoh;
    int         ptype, idummy;

    
    int cntr = 0;

    while(std::getline(inputFile, tempLine))
    {
      if(tempLine.empty()) continue; //Skip empty lines
      
      std::stringstream ss(tempLine);

      ss >>  idummy >> positions.x >> positions.y >> positions.z >> positions.w >>
             velocity.x >> velocity.y >> velocity.z  >>
             rhoh.x  >> rhoh.y >> hydro.z >> ptype;
             
//       fprintf(stderr,"TEST: %d | %f %f %f %f | %f %f %f | %f %f | %f | %d \n",
//               idummy, 
//               positions.x, positions.y, positions.z, positions.w,
//               velocity.x , velocity.y, velocity.z,
//               rhoh.x , rhoh.y , hydro.z, ptype);
//               

      if(reduceFactor > 0)
      {
        if(cntr % reduceFactor == 0)
        {
          positions.w *= reduceFactor;
          bodyPositions.push_back(positions);
          bodyVelocities.push_back(velocity);
          bodyDensRho.push_back(rhoh);
          bodyDrvt.push_back(drvt);
          bodyHydro.push_back(hydro);
          
          //ptype == 1 is default gas particle 
          //ptype == 4 is boundary particle
          int         offset=0;
          if(ptype == 4)
          {
              offset = 100000000; 
          }
      
          //Convert the ID to a star (disk for now) particle
          bodiesIDs.push_back(lGetIDType(pid++ + offset));
        }
      } 
      cntr++;   
  }
  inputFile.close();
  
  fprintf(stderr, "read %d bodies from dump file \n", cntr);
};




#if 1
template<typename IO, size_t N>
static double writeStars(std::vector<real4>     &bodyPositions, 
                         std::vector<real4>     &bodyVelocities, 
                         std::vector<IDType>    &bodiesIDs, 
                         std::vector<real2>    &bodyDensRho, 
                         std::vector<real4>    &bodyDrvt, 
                         std::vector<real4>    &bodyHydro, 
                         IO &out,
                         std::array<size_t,N> &count)
{
    double dtWrite = 0;
    
    const int nS  = bodyPositions.size();
  
  
    BonsaiIO::DataType<IDType> S_id ("Stars:IDType",           nS);
    BonsaiIO::DataType<real4>  S_pos("Stars:POS:real4",        nS);
    BonsaiIO::DataType<float3> S_vel("Stars:VEL:float[3]",     nS);
    BonsaiIO::DataType<float2> S_rhoh("Stars:RHOH:float[2]",   nS);
    BonsaiIO::DataType<float4> S_hydro("Stars:HYDRO:float[4]", nS);
    BonsaiIO::DataType<float4> S_drv  ("Stars:DRVT:float[4]" , nS);  
    

    for (int i = 0; i< nS; i++)
    {
      S_id[i] = bodiesIDs[i];   
      S_pos[i] = bodyPositions[i];
      S_vel[i][0] = bodyVelocities[i].x;
      S_vel[i][1] = bodyVelocities[i].y;
      S_vel[i][2] = bodyVelocities[i].z; 
      
      S_rhoh[i][0] = bodyDensRho[i].x;
      S_rhoh[i][1] = bodyDensRho[i].y;      
      
      S_hydro[i][0] = bodyHydro[i].x;
      S_hydro[i][1] = bodyHydro[i].y;
      S_hydro[i][2] = bodyHydro[i].z; 
      S_hydro[i][3] = bodyHydro[i].w; 
      
      S_drv[i][0] = bodyDrvt[i].x;
      S_drv[i][1] = bodyDrvt[i].y;
      S_drv[i][2] = bodyDrvt[i].z; 
      S_drv[i][3] = bodyDrvt[i].w;       
    }    

  
    out.write(S_id);
    out.write(S_pos);
    out.write(S_vel);
    out.write(S_rhoh);
    out.write(S_hydro);
    out.write(S_drv);    
  
  

  return dtWrite;
}
#endif

int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);


  if (argc < 3)
  {
    if (rank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  baseName outputName [reduceStar]\n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
 
  const std::string baseName(argv[1]);
  const std::string outputName(argv[2]);

  int reduceFactor  = 1;
  
  if(argc > 3) {
    reduceFactor  = atoi(argv[3]);
  }


  if( rank == 0) fprintf(stderr,"Basename: %s  outputname: %s Reducing Stars by factor: %d \n", baseName.c_str(), outputName.c_str(), reduceFactor);
 
  std::vector<real4>    bodyPositions;
  std::vector<real4>    bodyVelocities;
  std::vector<IDType>   bodiesIDs;
  std::vector<real2>   bodyDensRho;
  std::vector<real4>   bodyDrvt;
  std::vector<real4>   bodyHydro;
  
  readPhantomSPHFile(bodyPositions, bodyVelocities, bodiesIDs, bodyDensRho, bodyDrvt, bodyHydro, baseName, reduceFactor);

  if (rank == 0){
    fprintf(stderr, " nTotal= %ld \n", bodyPositions.size());
  }

  const double tAll = MPI_Wtime();
  {
    const double tOpen = MPI_Wtime();
    BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputName);
    if (rank == 0)
      fprintf(stderr, "Time= %g\n", 0.0); //data.time);
    out.setTime(0.0); //start at t=0
    double dtOpenLoc = MPI_Wtime() - tOpen;
    double dtOpenGlb;
    MPI_Allreduce(&dtOpenLoc, &dtOpenGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "open file in %g sec \n", dtOpenGlb);

    double dtWrite = 0;
  
    std::array<size_t,10> ntypeloc, ntypeglb;
    std::fill(ntypeloc.begin(), ntypeloc.end(), 0);

    
    if (rank == 0) fprintf(stderr, " write Stars\n");
    MPI_Barrier(comm);
    dtWrite += writeStars(bodyPositions, bodyVelocities, bodiesIDs,
                          bodyDensRho, bodyDrvt, bodyHydro,
                          out,ntypeloc);
    
    MPI_Reduce(&ntypeloc, &ntypeglb, ntypeloc.size(), MPI_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0)
    {
      size_t nsum = 0;
      for (int type = 0; type < (int)ntypeloc.size(); type++)
      {
        nsum += ntypeglb[type];
        if (ntypeglb[type] > 0)
          fprintf(stderr, "ptype= %d:  np= %zu \n",type, ntypeglb[type]);
      }
    }
  

    double dtWriteGlb;
    MPI_Allreduce(&dtWrite, &dtWriteGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "write file in %g sec \n", dtWriteGlb);
  

    const double tClose = MPI_Wtime();
    out.close();
    double dtCloseLoc = MPI_Wtime() - tClose;
    double dtCloseGlb;
    MPI_Allreduce(&dtCloseLoc, &dtCloseGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "close time in %g sec \n", dtCloseGlb);

    if (rank == 0)
    {
      out.getHeader().printFields();
      fprintf(stderr, " Bandwidth= %g MB/s\n", out.computeBandwidth()/1e6);
    }
  }
  double dtAllLoc = MPI_Wtime() - tAll;
  double dtAllGlb;
  MPI_Allreduce(&dtAllLoc, &dtAllGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
  if (rank == 0)
    fprintf(stderr, "All operations done in   %g sec \n", dtAllGlb);


  MPI_Finalize();


  return 0;
}


