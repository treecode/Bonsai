#include "BonsaiIO.h"
#include "IDType.h"
#include "Tree.h"

typedef float float5[5];
typedef float float4[4];
typedef float float3[3];
typedef float float2[2];
std::vector<Particle> Node::ptcl;
std::vector<Node>     Node::Node_heap;
std::vector<std::pair<Node*, Node*> > Node::pair_list;
  

static double read(
    const int rank, const MPI_Comm &comm,
    const std::vector<BonsaiIO::DataTypeBase*> &data, 
    BonsaiIO::Core &in, 
    const int reduce, 
    const bool restartFlag = true)
{
  double dtRead = 0;
  for (auto &type : data)
  {
    double t0 = MPI_Wtime();
    if (rank == 0)
      fprintf(stderr, " Reading %s ...\n", type->getName().c_str());
    if (in.read(*type, restartFlag, reduce))
    {
      long long int nLoc = type->getNumElements();
      long long int nGlb;
      MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
      if (rank == 0)
      {
        fprintf(stderr, " Read %lld of type %s\n",
            nGlb, type->getName().c_str());
        fprintf(stderr, " ---- \n");
      }
    } 
    else if (rank == 0)
    {
      fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }
      
    dtRead += MPI_Wtime() - t0;
  }

  return dtRead;
}

static double write(
    const int rank, const MPI_Comm &comm,
    const std::vector<BonsaiIO::DataTypeBase*> &data,
    BonsaiIO::Core &out)
{
  double dtWrite = 0;
  for (const auto &type : data)
  {
    double t0 = MPI_Wtime();
    if (rank == 0)
      fprintf(stderr, " Writing %s ... \n", type->getName().c_str());
    long long int nLoc = type->getNumElements();
    long long int nGlb;
    MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (nGlb > 0)
    {
      if (rank == 0)
        fprintf(stderr, " Writing %lld of type %s\n",
            nGlb, type->getName().c_str());
      assert(out.write(*type));
      if (rank == 0)
        fprintf(stderr, " ---- \n");
    }
    else if (rank == 0)
    {
      fprintf(stderr, " %s is empty... not writing \n", type->getName().c_str());
      fprintf(stderr, " ---- \n");
    }
    dtWrite += MPI_Wtime() - t0;
  }

  return dtWrite;
}



int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nRank, myRank;
  MPI_Comm_size(comm, &nRank);
  MPI_Comm_rank(comm, &myRank);

  assert(nRank == 1 && myRank == 0);

  if (argc < 5)
  {
    if (myRank == 0)
    {
      fprintf(stderr, " ------------------------------------------------------------------------\n");
      fprintf(stderr, " Usage: \n");
      fprintf(stderr, " %s  fileIn fileOut reduceDM reduceStars \n", argv[0]);
      fprintf(stderr, " ------------------------------------------------------------------------\n");
    }
    exit(-1);
  }
  
  const std::string fileIn (argv[1]);
  const std::string fileOut(argv[2]);
  const int reduceDM = atoi(argv[3]);
  const int reduceS  = atoi(argv[4]);

  if (myRank == 0)
  {
    fprintf(stderr, " Input file:  %s\n", fileIn.c_str());
    fprintf(stderr, "    reduceStars= %d \n", reduceS);
    fprintf(stderr, "    reduceDM   = %d \n", reduceDM);
    fprintf(stderr, " Output file: %s\n", fileOut.c_str());
  }

  std::vector<BonsaiIO::DataTypeBase*> dataStars, dataDM,data;
  
  /************* read ***********/

  {
    const double tOpen = MPI_Wtime(); 
    BonsaiIO::Core in(myRank, nRank, comm, BonsaiIO::READ,  fileIn);
    double dtOpen = MPI_Wtime() - tOpen;


    if (myRank == 0)
      in.getHeader().printFields();

    double dtRead;
    if (reduceDM > 0)
    {
      assert(0);
      dataDM.push_back(new BonsaiIO::DataType<IDType>("DM:IDType"));
      dataDM.push_back(new BonsaiIO::DataType<float4>("DM:POS:real4"));
      dataDM.push_back(new BonsaiIO::DataType<float3>("DM:VEL:float[3]"));
      dataDM.push_back(new BonsaiIO::DataType<float2>("DM:RHOH:float[2]"));

      dtRead += read(myRank, comm, dataDM, in, reduceDM);

      data.insert(data.end(), dataDM.begin(), dataDM.end());

    }
    if (reduceS > 0)
    {
      dataStars.push_back(new BonsaiIO::DataType<IDType>("Stars:IDType"));
      dataStars.push_back(new BonsaiIO::DataType<float4>("Stars:POS:real4"));
      dataStars.push_back(new BonsaiIO::DataType<float3>("Stars:VEL:float[3]"));
      dataStars.push_back(new BonsaiIO::DataType<float2>("Stars:RHOH:float[2]"));

      dtRead += read(myRank, comm, dataStars, in, reduceS);

      data.insert(data.end(), dataStars.begin(), dataStars.end());
    }

    double readBW = in.computeBandwidth();

    const double tClose = MPI_Wtime(); 
    in.close();
    double dtClose = MPI_Wtime() - tClose;

    double dtOpenGlb = 0;
    double dtReadGlb = 0;
    double dtCloseGlb = 0;
    MPI_Allreduce(&dtOpen, &dtOpenGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&dtRead, &dtReadGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&dtClose, &dtCloseGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (myRank == 0)
    {
      fprintf(stderr, "dtOpen = %g sec \n", dtOpenGlb);
      fprintf(stderr, "dtRead = %g sec \n", dtReadGlb);
      fprintf(stderr, "dtClose= %g sec \n", dtCloseGlb);
      fprintf(stderr, "Read BW= %g MB/s \n", readBW/1e6);
    }
  }

  /************* estimate density **********/


  {
    const auto &posArray = *dynamic_cast<BonsaiIO::DataType<float4>*>(dataStars[1]);
    const auto &velArray = *dynamic_cast<BonsaiIO::DataType<float3>*>(dataStars[2]);
    auto &rhohArray = *dynamic_cast<BonsaiIO::DataType<float2>*>(dataStars[3]);
    const size_t np = posArray.getNumElements();
    Particle::Vector ptcl(np);

    fprintf(stderr, " -- create tree particles -- \n");

    for (size_t i = 0; i < np; i++)
    {
      const auto &pos = posArray[i];
      const auto &vel = velArray[i];
      ptcl[i].ID   = i;
      ptcl[i].pos  = vec3(pos[0], pos[1], pos[2]);
      ptcl[i].mass = pos[3];
      ptcl[i].vel  = vec3(vel[0], vel[1], vel[2]);
    }
    Node::allocate(np,np);
    fprintf(stderr, " -- build tree -- \n");
    Tree tree(ptcl);

    const auto &leafArray = tree.leafArray;
    const int nLeaf = leafArray.size();

    fprintf(stderr, " np= %d  nleaf= %d  NLEAF= %d\n",
        (int)np, nLeaf, Node::NLEAF);

    rhohArray.resize(np);
    
    fprintf(stderr, " -- generate output data  -- \n");

    size_t npMean = 0;
    for (int i = 0; i < nLeaf; i++)
    {
      const auto &leaf = *leafArray[i];
      vec3 pos(0.0);
      vec3 vel(0.0);
      float mass = 0.0;
      npMean += leaf.np;
      for (int j = 0; j < leaf.np; j++)
      {
        const auto &p = Node::ptcl[leaf.pfirst+j];
        mass += p.mass;
        pos += p.pos*p.mass;
        vel += p.vel*p.mass;
      }
      pos *= 1.0/mass;
      vel *= 1.0/mass;
      const float volume = leaf.size*leaf.size*leaf.size;
      const float rho = mass/volume;
      const float h   = leaf.size;
      for (int j = 0; j < leaf.np; j++)
      {
        const auto &p = Node::ptcl[leaf.pfirst+j];
        rhohArray[p.ID][0] = rho;
        rhohArray[p.ID][1] = h;
      }
    }
    fprintf(stderr, "npMean= %g\n", 1.0*npMean/nLeaf);

  }




  /************* write ***********/

  {
    fprintf(stderr, " -- write  output   -- \n");
    const double tOpen = MPI_Wtime(); 
    BonsaiIO::Core out(myRank, nRank, comm, BonsaiIO::WRITE,  fileOut);
    double dtOpen = MPI_Wtime() - tOpen;

    double dtWrite = 0;
    dtWrite += write(myRank, comm, data, out);

    double writeBW = out.computeBandwidth();
    const double tClose = MPI_Wtime(); 
    out.close();
    double dtClose = MPI_Wtime() - tClose;
    
    double dtOpenGlb = 0;
    double dtWriteGlb = 0;
    double dtCloseGlb = 0;
    MPI_Allreduce(&dtOpen, &dtOpenGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&dtWrite, &dtWriteGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    MPI_Allreduce(&dtClose, &dtCloseGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
    if (myRank == 0)
    {
      fprintf(stderr, "dtOpen = %g sec \n", dtOpenGlb);
      fprintf(stderr, "dtWrite = %g sec \n", dtWriteGlb);
      fprintf(stderr, "dtClose= %g sec \n", dtCloseGlb);
      fprintf(stderr, "Write BW= %g MB/s \n", writeBW/1e6);
    }
  }

  /*********** write stats **********/


  MPI_Finalize();




  return 0;
}


