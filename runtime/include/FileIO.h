/*
 *
 * Functions to read / write to various file formats
 *
 *
 *
 */

#pragma once

#include "IDType.h"

#ifdef USE_MPI
    #include "BonsaiIO.h"
#endif    



/*********************************/
/******** Input functions  ******/
/*********************************/

template<typename T>
static void lHandShake(SharedMemoryBase<T> &header)
{
  header.acquireLock();
  header[0].handshake = false;
  header.releaseLock();

  while (!header[0].handshake)
    usleep(10000);

  header.acquireLock();
  header[0].handshake = false;
  header.releaseLock();
}


/*
 *
 * Functions to read the MPI-IO based file format
 *
 *
 */


static IDType lGetIDType(const long long id)
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



#ifdef USE_MPI
    template<typename T>
    static inline T& lBonsaiSafeCast(BonsaiIO::DataTypeBase* ptrBase)
    {
    T* ptr = dynamic_cast<T*>(ptrBase);
    assert(ptr != NULL);
    return *ptr;
    }




    static double lReadBonsaiFields(
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
        else
        {
        if (rank == 0)
        {
            fprintf(stderr, " %s  is not found, skipping\n", type->getName().c_str());
            fprintf(stderr, " ---- \n");
        }
        }

        dtRead += MPI_Wtime() - t0;
    }

    return dtRead;
    }


    static void lReadBonsaiFile(
        std::vector<real4 > &bodyPositions,
        std::vector<real4 > &bodyVelocities,
        std::vector<ullong> &bodyIDs,
        int &NFirst, int &NSecond, int &NThird,
        octree *tree,
        const std::string &fileName,
        const int rank, const int nrank, const MPI_Comm &comm,
        const bool restart = true,
        const int reduceFactor = 1)
    {
    if (rank == 0)
        std::cerr << " >>> Reading Bonsai file format : " << fileName <<  std::endl;

    BonsaiIO::Core *in;
    try
    {
        in = new BonsaiIO::Core(rank, nrank, comm, BonsaiIO::READ, fileName);
    }
    catch (const std::exception &e)
    {
        if (rank == 0)
        fprintf(stderr, "Something went wrong: %s \n", e.what());
        MPI_Finalize();
        ::exit(0);
    }

    if (rank == 0)
        in->getHeader().printFields();

    std::vector<BonsaiIO::DataTypeBase*> data;
    typedef float float3[3];
    typedef float float2[2];

    using IDType = BonsaiIO::DataType<IDType>;
    using Pos    = BonsaiIO::DataType<real4>;
    using Vel    = BonsaiIO::DataType<float3>;
    using RhoH   = BonsaiIO::DataType<float2>;
    data.push_back(new IDType("DM:IDType"));
    data.push_back(new Pos   ("DM:POS:real4"));
    data.push_back(new Vel   ("DM:VEL:float[3]"));
    data.push_back(new IDType("Stars:IDType"));
    data.push_back(new Pos   ("Stars:POS:real4"));
    data.push_back(new Vel   ("Stars:VEL:float[3]"));

    const double dtRead = lReadBonsaiFields(rank, comm, data, *in, reduceFactor, restart);

    const auto &DM_IDType = lBonsaiSafeCast<IDType>(data[0]);
    const auto &DM_Pos    = lBonsaiSafeCast<Pos   >(data[1]);
    const auto &DM_Vel    = lBonsaiSafeCast<Vel   >(data[2]);
    const auto &S_IDType  = lBonsaiSafeCast<IDType>(data[3]);
    const auto &S_Pos     = lBonsaiSafeCast<Pos   >(data[4]);
    const auto &S_Vel     = lBonsaiSafeCast<Vel   >(data[5]);

    const size_t nDM = DM_IDType.size();
    assert(nDM == DM_Pos.size());
    assert(nDM == DM_Vel.size());

    const size_t nS = S_IDType.size();
    assert(nS == S_Pos.size());
    assert(nS == S_Vel.size());


    NFirst  = static_cast<std::remove_reference<decltype(NFirst )>::type>(nDM);
    NSecond = static_cast<std::remove_reference<decltype(NSecond)>::type>(nS);
    NThird  = 0;

    bodyPositions.resize(nDM+nS);
    bodyVelocities.resize(nDM+nS);
    bodyIDs.resize(nDM+nS);

    /* store DM */

    constexpr int ntypecount = 10;
    std::array<size_t,ntypecount> ntypeloc, ntypeglb;
    std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
    for (int i = 0; i < nDM; i++)
    {
        ntypeloc[0]++;
        auto &pos = bodyPositions[i];
        auto &vel = bodyVelocities[i];
        auto &ID  = bodyIDs[i];
        pos = DM_Pos[i];
        pos.w *= reduceFactor;
        vel = make_float4(DM_Vel[i][0], DM_Vel[i][1], DM_Vel[i][2],0.0f);
        ID  = DM_IDType[i].getID() + DARKMATTERID;
    }

    for (int i = 0; i < nS; i++)
    {
        auto &pos = bodyPositions[nDM+i];
        auto &vel = bodyVelocities[nDM+i];
        auto &ID  = bodyIDs[nDM+i];
        pos = S_Pos[i];
        pos.w *= reduceFactor;
        vel = make_float4(S_Vel[i][0], S_Vel[i][1], S_Vel[i][2],0.0f);
        ID  = S_IDType[i].getID();
        switch (S_IDType[i].getType())
        {
        case 1:  /*  Bulge */
            ID += BULGEID;
            break;
        case 2:  /*  Disk */
            ID += DISKID;
            break;
        }
        if (S_IDType[i].getType() < ntypecount)
        ntypeloc[S_IDType[i].getType()]++;
    }

    MPI_Reduce(&ntypeloc, &ntypeglb, ntypecount, MPI_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0)
    {
        size_t nsum = 0;
        for (int type = 0; type < ntypecount; type++)
        {
        nsum += ntypeglb[type];
        if (ntypeglb[type] > 0)
            fprintf(stderr, "bonsai-read: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
        }
        assert(nsum > 0);
    }


    LOGF(stderr,"Read time from snapshot: %f \n", in->getTime());

    if(static_cast<float>(in->getTime()) > 10e10 ||
        static_cast<float>(in->getTime()) < -10e10){
        tree->set_t_current(0);
    }
    else{
        tree->set_t_current(static_cast<float>(in->getTime()));
    }

    in->close();
    const double bw = in->computeBandwidth()/1e6;
    for (auto d : data)
        delete d;
    delete in;
    if (rank == 0)
        fprintf(stderr, " :: dtRead= %g  sec readBW= %g MB/s \n", dtRead, bw);
    }

#endif


/******************************************************************/
/*      Function to read/write the legacy Tipsy format            */
/*                                                                */
/******************************************************************/

static void read_tipsy_file_parallel(const MPI_Comm &mpiCommWorld,
    vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<ullong> &bodiesIDs,  float eps2, string fileName,
                              int rank, int procs, int &NTotal2, int &NFirst,
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<ullong> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor,
                              const bool restart)
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /*
     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the
     location where previously the potential was stored.
  */


  char fullFileName[256];
  if(restart)
    sprintf(fullFileName, "%s%d", fileName.c_str(), rank);
  else
    sprintf(fullFileName, "%s", fileName.c_str());

  LOG("Trying to read file: %s \n", fullFileName);



  ifstream inputFile(fullFileName, ios::in | ios::binary);
  if(!inputFile.is_open())
  {
    LOG("Can't open input file \n");
    ::exit(0);
  }

  dumpV2  h;
  inputFile.read((char*)&h, sizeof(h));

  int NTotal;
  ullong idummy;
  real4 positions;
  real4 velocity;


  //Read Tipsy header
  NTotal        = h.nbodies;
  NFirst        = h.ndark;
  NSecond       = h.nstar;
  NThird        = h.nsph;

  printf("File version: %d \n", h.version);

  int fileFormatVersion = 0;

  if(h.version == 2) fileFormatVersion = 2;



  tree->set_t_current((float) h.time);

  //Rough divide
  uint perProc = (NTotal / procs) /reduce_bodies_factor;
  if(restart) perProc = NTotal /reduce_bodies_factor; //don't subdivide when using restart
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;

  dark_particleV2 d;
  star_particleV2 s;

  int globalParticleCount = 0;
  int bodyCount = 0;
  int dustCount = 0;

  constexpr int ntypecount = 10;
  std::array<size_t,ntypecount> ntypeloc, ntypeglb;
  std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
  for(int i=0; i < NTotal; i++)
  {
    if(i < NFirst)
    {
      inputFile.read((char*)&d, sizeof(d));
      //velocity.w        = d.eps;
      velocity.w        = 0;
      positions.w       = d.mass;
      positions.x       = d.pos[0];
      positions.y       = d.pos[1];
      positions.z       = d.pos[2];
      velocity.x        = d.vel[0];
      velocity.y        = d.vel[1];
      velocity.z        = d.vel[2];
      idummy            = d.getID();

      //printf("%d\t%f\t%f\t%f\n", i, positions.x, positions.y, positions.z);

      //Force compatibility with older 32bit ID files by mapping the particle IDs
      if(fileFormatVersion == 0)
      {
        idummy    = s.getID_V1() + DARKMATTERID;
      }
      //end mapping
    }
    else
    {
      inputFile.read((char*)&s, sizeof(s));
      //velocity.w        = s.eps;
      velocity.w        = 0;
      positions.w       = s.mass;
      positions.x       = s.pos[0];
      positions.y       = s.pos[1];
      positions.z       = s.pos[2];
      velocity.x        = s.vel[0];
      velocity.y        = s.vel[1];
      velocity.z        = s.vel[2];
      idummy            = s.getID();

      //Force compatibility with older 32bit ID files by mapping the particle IDs
      if(fileFormatVersion == 0)
      {
        if(s.getID_V1() >= 100000000)
          idummy    = s.getID_V1() + BULGEID; //Bulge particles
        else
          idummy    = s.getID_V1();
      }
      //end mapping
    }


    if(positions.z < -10e10)
    {
       fprintf(stderr," Removing particle %d because of Z is: %f \n", globalParticleCount, positions.z);
       continue;
    }

    const auto id = idummy;
    if(id >= DISKID  && id < BULGEID)
    {
      ntypeloc[2]++;
    }
    else if(id >= BULGEID && id < DARKMATTERID)
    {
      ntypeloc[1]++;
    }
    else if (id >= DARKMATTERID)
    {
      ntypeloc[0]++;
    }

    globalParticleCount++;

    #ifdef USE_DUST
      if(idummy >= 50000000 && idummy < 100000000)
      {
        dustCount++;
        if( dustCount % reduce_dust_factor == 0 )
          positions.w *= reduce_dust_factor;

        if( dustCount % reduce_dust_factor != 0 )
          continue;
        dustPositions.push_back(positions);
        dustVelocities.push_back(velocity);
        dustIDs.push_back(idummy);
      }
      else
      {
        bodyCount++;
        if( bodyCount % reduce_bodies_factor == 0 )
          positions.w *= reduce_bodies_factor;

        if( bodyCount % reduce_bodies_factor != 0 )
          continue;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);
      }


    #else
      if( globalParticleCount % reduce_bodies_factor == 0 )
        positions.w *= reduce_bodies_factor;

      if( globalParticleCount % reduce_bodies_factor != 0 )
        continue;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(idummy);

    #endif

    particleCount++;




    if(!restart)
    {
#ifdef USE_MPI
      if(bodyPositions.size() > perProc && procCntr != procs)
      {
        tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
        procCntr++;

        bodyPositions.clear();
        bodyVelocities.clear();
        bodiesIDs.clear();
      }
#endif
    }
  }//end while

  inputFile.close();

  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);
//   NTotal2 = particleCount-1;
  NTotal2 = particleCount;
  LOGF(stderr,"NTotal: %d\tper proc: %d\tFor ourself: %d \tNDust: %d \n",
               NTotal, perProc, (int)bodiesIDs.size(), (int)dustPositions.size());

  /* this check was added to test whether particle type was failed to identified.
   * Sometimes DM particles are treated as nonDM on MW+M31 4.6M particle snapshot
   */
  if (restart)
  {
    #ifdef USE_MPI      
        MPI_Reduce(&ntypeloc, &ntypeglb, 10, MPI_LONG_LONG, MPI_SUM, 0, mpiCommWorld);
    #endif
  }
  else
  {
    std::copy(ntypeloc.begin(), ntypeloc.end(), ntypeglb.begin());
  }

  if (rank == 0)
  {
    for (int type = 0; type < ntypecount; type++)
      if (ntypeglb[type] > 0)
        fprintf(stderr, "tispy-read: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
  }
}
