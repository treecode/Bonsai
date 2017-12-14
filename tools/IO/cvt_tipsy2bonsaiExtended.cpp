#include "BonsaiIO.h"
#include "IDType.h"
#include "read_tipsy.h"
#include <array>

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

template<typename IO, size_t N>
static double writeDM(ReadTipsy &data, IO &out,
    std::array<size_t,N> &count)
{
  double dtWrite = 0;
  const int pCount  = data.firstID.size();
  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("DM:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      ID[i] = lGetIDType(data.firstID[i]);
      assert(ID[i].getType() == 0);
      if (ID[i].getType() < count.size())
        count[ID[i].getType()]++;
    }
    double t0 = MPI_Wtime();
    out.write(ID);
    dtWrite += MPI_Wtime() - t0;
  }
  
  /* write pos */
  {
    BonsaiIO::DataType<ReadTipsy::real4> pos("DM:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
      pos[i] = data.firstPos[i];
    double t0 = MPI_Wtime();
    out.write(pos);
    dtWrite += MPI_Wtime() - t0;
  }
    
  /* write vel */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("DM:VEL:float[3]",pCount);
    for (int i = 0; i< pCount; i++)
    {
      vel[i][0] = data.firstVel[i].x;
      vel[i][1] = data.firstVel[i].y;
      vel[i][2] = data.firstVel[i].z;
    }
    double t0 = MPI_Wtime();
    out.write(vel);
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}

template<typename IO, size_t N>
static double writeStars(ReadTipsy &data, IO &out,
    std::array<size_t,N> &count)
{
  double dtWrite = 0;

  const int pCount  = data.secondID.size();

  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("Stars:IDType", pCount);
    for (int i = 0; i< pCount; i++)
    {
      ID[i] = lGetIDType(data.secondID[i]);
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
    BonsaiIO::DataType<ReadTipsy::real4> pos("Stars:POS:real4",pCount);
    for (int i = 0; i< pCount; i++)
      pos[i] = data.secondPos[i];
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
      vel[i][0] = data.secondVel[i].x;
      vel[i][1] = data.secondVel[i].y;
      vel[i][2] = data.secondVel[i].z;
    }
    double t0 = MPI_Wtime();
    out.write(vel);
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}


int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);


  if (argc < 4)
  {
    if (rank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  baseName nDomains outputName reduceDM reduceStar  startLoop endLoop Incr \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
 
  	
  int startLoop = atoi(argv[6]);
  int endLoop   = atoi(argv[7]);
  int increase  = atoi(argv[8]);





for(int i=startLoop; i <= endLoop; i+= increase)
{

  char buff[512];

  sprintf(buff, argv[1], i);
  const std::string baseName(buff);

  sprintf(buff, argv[3], i);
  const std::string outputName(buff);

  //const std::string baseName(argv[1]);
  const int nDomains = atoi(argv[2]);
  //const std::string outputName(argv[3]);


  if( rank == 0)
	  fprintf(stderr,"Basename: %s  outputname: %s \n", baseName.c_str(), outputName.c_str());



  int reduceFactorFirst  = 1;
  int reduceFactorSecond = 1;
  
  if(argc > 3)
  {
	  reduceFactorFirst  = atoi(argv[4]);
	  reduceFactorSecond = atoi(argv[5]);
  }

  if(rank == 0)
	  fprintf(stderr,"Reducing DM: %d  Stars: %d \n", reduceFactorFirst, reduceFactorSecond);



  ReadTipsy data(
      baseName, 
      rank, nranks,
      nDomains, 
      reduceFactorFirst,
      reduceFactorSecond);

  long long nFirstLocal = data.firstID.size();
  long long nSecondLocal = data.secondID.size();

  long long nFirst, nSecond;
  MPI_Allreduce(&nFirstLocal, &nFirst, 1, MPI_LONG, MPI_SUM, comm);
  MPI_Allreduce(&nSecondLocal, &nSecond, 1, MPI_LONG, MPI_SUM, comm);

  if (rank == 0)
  {
    fprintf(stderr, " nFirst = %lld \n", nFirst);
    fprintf(stderr, " nSecond= %lld \n", nSecond);
    fprintf(stderr, " nTotal= %lld \n", nFirst + nSecond);
  }

  const double tAll = MPI_Wtime();
  {
    const double tOpen = MPI_Wtime();
    BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputName);
    if (rank == 0)
      fprintf(stderr, "Time= %g\n", data.time);
    out.setTime(data.time);
    double dtOpenLoc = MPI_Wtime() - tOpen;
    double dtOpenGlb;
    MPI_Allreduce(&dtOpenLoc, &dtOpenGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0)
      fprintf(stderr, "open file in %g sec \n", dtOpenGlb);

    double dtWrite = 0;
  
    std::array<size_t,10> ntypeloc, ntypeglb;
    std::fill(ntypeloc.begin(), ntypeloc.end(), 0);

    
    if (rank == 0)
      fprintf(stderr, " write DM  \n");
    MPI_Barrier(comm);
    dtWrite += writeDM(data,out,ntypeloc);

    if (rank == 0)
      fprintf(stderr, " write Stars\n");
    MPI_Barrier(comm);
    dtWrite += writeStars(data,out,ntypeloc);
    
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
    if (rank == 0)
      fprintf(stderr, "write file in %g sec \n", dtWriteGlb);
  

    const double tClose = MPI_Wtime();
    out.close();
    double dtCloseLoc = MPI_Wtime() - tClose;
    double dtCloseGlb;
    MPI_Allreduce(&dtCloseLoc, &dtCloseGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0)
      fprintf(stderr, "close time in %g sec \n", dtCloseGlb);

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


}

  MPI_Finalize();




  return 0;
}


