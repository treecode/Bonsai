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

