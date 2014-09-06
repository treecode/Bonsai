#include "BonsaiIO.h"
#include "IDType.h"
#include "read_tipsy.h"


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
      fprintf(stderr, " %s  output np \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string outputName(argv[1]);
  const int np = atoi(argv[2]);

  const double tAll = MPI_Wtime();
  {
    const double tOpen = MPI_Wtime();
    BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputName);
    double dtOpenLoc = MPI_Wtime() - tOpen;
    double dtOpenGlb;
    MPI_Allreduce(&dtOpenLoc, &dtOpenGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0)
      fprintf(stderr, "open file in %g sec \n", dtOpenGlb);

    double dtWrite = 0;

    /* write IDs */
    {
      BonsaiIO::DataType<IDType> ID("IDType", np);
      for (int i = 0; i< np; i++)
      {
        ID[i].setID(i);
        ID[i].setType(i%3);
      }
      double t0 = MPI_Wtime();
      out.write(ID);
      dtWrite += MPI_Wtime() - t0;
    }
  
    /* write pos */
    {
      BonsaiIO::DataType<ReadTipsy::real4> pos("POS:real4",np);
      for (int i = 0; i< np; i++)
      {
        pos[i].x = drand48();
        pos[i].y = drand48();
        pos[i].z = drand48();
        pos[i].w = 1.0/np;
      }
      double t0 = MPI_Wtime();
      out.write(pos);
      dtWrite += MPI_Wtime() - t0;
    }
    
    /* write vel */
    {
      typedef float vec3[3];
      BonsaiIO::DataType<vec3> vel("VEL:float[3]",np);
      for (int i = 0; i< np; i++)
      {
        vel[i][0] = drand48();
        vel[i][1] = drand48();
        vel[i][2] = drand48();
      }
      double t0 = MPI_Wtime();
      out.write(vel);
      dtWrite += MPI_Wtime() - t0;
    }
    
    /* write rhoh */
    {
      typedef float vec2[2];
      BonsaiIO::DataType<vec2> rhoh("RHOH:float[2]",np);
      for (int i = 0; i< np; i++)
      {
        rhoh[i][0] = drand48();
        rhoh[i][1] = drand48();
      }
      double t0 = MPI_Wtime();
      out.write(rhoh);
      dtWrite += MPI_Wtime() - t0;
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


  MPI_Finalize();

  return 0;
}


