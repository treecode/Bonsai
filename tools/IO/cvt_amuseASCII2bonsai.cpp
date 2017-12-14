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


void readAMUSEFile(std::vector<real4>    &bodyPositions, 
                   std::vector<real4>    &bodyVelocities, 
                   std::vector<IDType>   &bodiesIDs, 
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
    
    //Skip the  header lines
    std::string tempLine;
    std::getline(inputFile, tempLine);
    std::getline(inputFile, tempLine);
    

    int pid  = 0;
    real4       positions;
    real4       velocity;
    int cntr = 0;
//     float r2 = 0;
    while(std::getline(inputFile, tempLine))
    {
      if(tempLine.empty()) continue; //Skip empty lines
      
      std::stringstream ss(tempLine);
      //Amuse format
//       inputFile >> positions.w >> r2 >> r2 >> 
//                velocity.x  >> velocity.y  >> velocity.z  >>
//                positions.x >> positions.y >> positions.z;
      ss >> positions.w >>
              velocity.x >> velocity.y >> velocity.z  >>
              positions.x >> positions.y >> positions.z;
              
//               positions.x /= 1000;
//               positions.y /= 1000;
//               positions.z /= 1000;
          
//       idummy = pid; //particleIDtemp;
      
//       cout << idummy << "\t"<< positions.w << "\t"<<  positions.x << "\t"<<  positions.y << "\t"<< positions.z << "\t"
//       << velocity.x << "\t"<< velocity.y << "\t"<< velocity.z << "\t" << velocity.w << "\n";
              
      if(reduceFactor > 0)
      {
        if(cntr % reduceFactor == 0)
        {
          positions.w *= reduceFactor;
          bodyPositions.push_back(positions);
          bodyVelocities.push_back(velocity);
      
          //Convert the ID to a star (disk for now) particle
          bodiesIDs.push_back(lGetIDType(pid++));
        }
      }      
      cntr++;   
  }
  inputFile.close();
  
  fprintf(stderr, "read %d bodies from dump file \n", cntr);
};




#if 1
template<typename IO, size_t N>
static double writeStars(std::vector<real4>    &bodyPositions, 
                         std::vector<real4>    &bodyVelocities, 
                         std::vector<IDType>   &bodiesIDs, 
                         IO &out,
                         std::array<size_t,N> &count)
{
  double dtWrite = 0;

  const int pCount  = bodyPositions.size();

  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      //ID[i] = lGetIDType(bodiesIDs[i]);
      ID[i] = bodiesIDs[i];
      assert(ID[i].getType() > 0);
      if (ID[i].getType() < count.size())
        count[ID[i].getType()]++;
    }
    double t0 = MPI_Wtime();
    out.write(ID);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write pos */
  {
    BonsaiIO::DataType<real4> pos("Stars:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
      pos[i] = bodyPositions[i];
    double t0 = MPI_Wtime();
    out.write(pos);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write vel */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("Stars:VEL:float[3]",pCount);
    for (int i = 0; i< pCount; i++)
    {
      vel[i][0] = bodyVelocities[i].x;
      vel[i][1] = bodyVelocities[i].y;
      vel[i][2] = bodyVelocities[i].z;
    }
    double t0 = MPI_Wtime();
    out.write(vel);
    dtWrite += MPI_Wtime() - t0;
  }

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
  
  readAMUSEFile(bodyPositions, bodyVelocities, bodiesIDs, baseName, reduceFactor);

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
    dtWrite += writeStars(bodyPositions, bodyVelocities, bodiesIDs,out,ntypeloc);
    
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


