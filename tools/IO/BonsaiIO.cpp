#include "BonsaiIO.h"
#include "IDType.h"

struct real2 { float x,y;};
struct real4 { float x,y,z,w;};
typedef unsigned long long uulong;


#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000

void writeSnapshot(
    real4 *bodyPositions,
    real4 *bodyVelocities,
    uulong* bodyIds,
    const int n,
    const int nDomains,
    const std::string &fileName,
    const float time,
    const MPI_Comm &comm,
    const int nRank, const int myRank)
{
  BonsaiIO::Core out(myRank, nRank, comm, BonsaiIO::WRITE, fileName);

  /* write IDs */
  {
    BonsaiIO::DataType<IDType> ID("IDType", n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      ID[i].setID(bodyIds[i]);
      int type = 0;
      if(bodyIds[i] >= DISKID  && bodyIds[i] < BULGEID)       type = 2;
      if(bodyIds[i] >= BULGEID && bodyIds[i] < DARKMATTERID)  type = 1;
      if(bodyIds[i] >= DARKMATTERID)                          type = 0;
      ID[i].setType(type);
    }
    out.write(ID);
  }

  /* write pos */
  {
    BonsaiIO::DataType<real4> pos("POS",n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
      pos[i] = bodyPositions[i];
    out.write(pos);
  }

  /* write velocities */
  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("VEL",n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      vel[i][0] = bodyVelocities[i].x;
      vel[i][1] = bodyVelocities[i].y;
      vel[i][2] = bodyVelocities[i].z;
    }
    out.write(vel);
  }
  
  /* write rhoh */
  {
    typedef float vec2[2];
    BonsaiIO::DataType<vec2> rhoh("RHOH",n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      rhoh[i][0] = 0; /* rho */
      rhoh[i][1] = 0; /*  h  */
    }
    out.write(rhoh);
  }

  out.close();
}

void readSnapshot(
    std::vector<real4>  &bodyPositions,
    std::vector<real4>  &bodyVelocities,
    std::vector<uulong> &bodyID,
    std::vector<real2>  &rhohList,
    const std::string   &fileName,
    const MPI_Comm       &comm,
    const int nRank, 
    const int myRank,
    int &NTotal2,
    int &NFirst, int &NSecond, int &NThird,
    std::vector<real4> &dustPositions, std::vector<real4> &dustVelocities,
    std::vector<uulong> &dustIDs, 
    const int reduce_bodies_factor,
    const int reduce_dust_factor,
    const bool restart)
{
  NFirst = NSecond = NThird = 0;
  BonsaiIO::Core out(myRank, nRank, comm, BonsaiIO::READ, fileName);

  {
    BonsaiIO::DataType<IDType> IDList("IDType");
    if (!out.read(IDList))
    {
      if (myRank == 0)
        fprintf(stderr, " FATAL: No particle ID data is found. Please make sure you passed the right file \n");
      exit(-1);
    }
    const int n = IDList.getNumElements();
    bodyID.resize(n);
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      bodyID[i] = IDList[i].getID();
      switch (IDList[i].getType())
      {
        case 0:
          NFirst++;
          break;
        case 1:
          NSecond++;
          break;
        case 2:
          NThird++;
          break;
      }
    }
    NTotal2 = NFirst+NSecond+NThird;
    assert(NTotal2 == n);
  }

  {
    BonsaiIO::DataType<real4> pos("pos");
    if (!out.read(pos))
    {
      if (myRank == 0)
        fprintf(stderr, " FATAL: No particle positions data is found. Please make sure you passed the right file \n");
      exit(-1);
    }
    const int n = pos.getNumElements();
    bodyPositions.resize(n);
    assert(bodyPositions.size() == bodyID.size());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
      bodyPositions[i] = pos[i];
  }

  {
    typedef float vec3[3];
    BonsaiIO::DataType<vec3> vel("VEL");
    if (!out.read(vel))
    {
      if (myRank == 0)
        fprintf(stderr, " FATAL: No particle velocity data is found. Please make sure you passed the right file \n");
      exit(-1);
    }
    const int n = vel.getNumElements();
    bodyVelocities.resize(n);
    assert(bodyVelocities.size() == bodyID.size());
#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      bodyVelocities[i].x = vel[i][0];
      bodyVelocities[i].y = vel[i][1];
      bodyVelocities[i].z = vel[i][2];
    }
  }

  {
    typedef float vec2[2];
    BonsaiIO::DataType<vec2> rhoh("RHOH");
    if (out.read(rhoh))
    {
      if (myRank == 0)
        fprintf(stderr , " -- RHOH data is found \n");
      const int n = rhoh.getNumElements();
      rhohList.resize(n);
      assert(rhohList.size() == bodyID.size());
#pragma omp parallel for
      for (int i = 0; i < n; i++)
      {
        rhohList[i].x = rhoh[i][0];
        rhohList[i].y = rhoh[i][1];
      }
    }
    else if (myRank == 0)
      fprintf(stderr , " -- No RHOH data is found \n");
  }

}



