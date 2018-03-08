#include "octree.h"

#include "FileIO.h"

#include "SharedMemory.h"
#include "BonsaiSharedData.h"
#include <array>

#ifndef WIN32
    #include <sys/time.h>
#endif


#include "IDType.h"

#ifdef USE_MPI
    #include "BonsaiIO.h"
#endif


/*********************************/
/*********************************/
/*********************************/



void octree::set_src_directory(string src_dir) {                                                                                                                                 
    this->src_directory = (char*)src_dir.c_str();                                                                                                                                
}   

double octree::get_time() {
#ifdef WIN32
  if (sysTimerFreq.QuadPart == 0)
  {
    return -1.0;
  }
  else
  {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>( (double)(c.QuadPart - sysTimerAtStart.QuadPart) / sysTimerFreq.QuadPart );
  }
#else
  struct timeval Tvalue;
  struct timezone dummy;
  
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
#endif
}

int octree::getAllignmentOffset(int n)
{
  const int allignBoundary = 128*sizeof(uint); //Fermi,128 bytes 
  
  int offset = 0;
  //Compute the number of bytes  
  offset = n*sizeof(uint); 
  //Compute number of 256 byte blocks  
  offset = (offset / allignBoundary) + (((offset % allignBoundary) > 0) ? 1 : 0); 
  //Compute the number of bytes padded / offset 
  offset = (offset * allignBoundary) - n*sizeof(uint); 
  //Back to the actual number of elements
  offset = offset / sizeof(uint);   
  
  return offset;
}

int octree::getTextureAllignmentOffset(int n, int size)
{
    const int texBoundary = TEXTURE_BOUNDARY; //Fermi
  
    int textOffset = 0;
    //Compute the number of bytes  
    textOffset = n*size; 
    //Compute number of texBoundary byte blocks  
    textOffset = (textOffset / texBoundary) + (((textOffset % texBoundary) > 0) ? 1 : 0); 
    //Compute the number of bytes padded / offset 
    textOffset = (textOffset * texBoundary) - n*size; 
    //Back to the actual number of elements
    textOffset = textOffset / size; 
    
    return textOffset;
}   




/*********************************/
/******** Output functions  ******/
/*********************************/


/*
 * BonsaiIO output routines
 *
 */

using ShmQHeader = SharedMemoryServer<BonsaiSharedQuickHeader>;
using ShmQData   = SharedMemoryServer<BonsaiSharedQuickData>;
using ShmSHeader = SharedMemoryServer<BonsaiSharedSnapHeader>;
using ShmSData   = SharedMemoryServer<BonsaiSharedSnapData>;

static ShmQHeader *shmQHeader = NULL;
static ShmQData   *shmQData   = NULL;
static ShmSHeader *shmSHeader = NULL;
static ShmSData   *shmSData   = NULL;

/*
 *  Signal the IO process that we're finished (this is when tCurrent == -1) *
 *
 */

void octree::terminateIO() const
{
  {
    auto &header = *shmQHeader;
    header.acquireLock();
    header[0].tCurrent = -1;
    header[0].done_writing = false;
    header.releaseLock();
  }
  {
    auto &header = *shmSHeader;
    header.acquireLock();
    header[0].tCurrent = -1;
    header[0].done_writing = false;
    header.releaseLock();
  }
}

#ifdef USE_MPI
/*
 *
 * Send the particle data over MPI to another process.
 * Note only works for nQuickDump data
 */

void octree::dumpDataMPI()
{
  static MPI_Datatype MPI_Header = 0;
  static MPI_Datatype MPI_Data   = 0;
  if (!MPI_Header)
  {
    int ss = sizeof(BonsaiSharedHeader) / sizeof(char);
    assert(0 == sizeof(BonsaiSharedHeader) % sizeof(char));
    MPI_Type_contiguous(ss, MPI_BYTE, &MPI_Header);
    MPI_Type_commit(&MPI_Header);
  }
  if (!MPI_Data)
  {
    int ss = sizeof(BonsaiSharedData) / sizeof(char);
    assert(0 == sizeof(BonsaiSharedData) % sizeof(char));
    MPI_Type_contiguous(ss, MPI_BYTE, &MPI_Data);
    MPI_Type_commit(&MPI_Data);
  }

  BonsaiSharedHeader header;
  std::vector<BonsaiSharedData> data;

  if (t_current >= nextQuickDump && quickDump > 0)
  {
    localTree.bodies_pos.d2h();
    localTree.bodies_vel.d2h();
    localTree.bodies_ids.d2h();
    localTree.bodies_h.d2h();
    localTree.bodies_dens.d2h();

    nextQuickDump += quickDump;
    nextQuickDump  = std::max(nextQuickDump, t_current);

    if (procId == 0) fprintf(stderr, "-- quickdumpMPI: nextQuickDump= %g  quickRatio= %g\n", nextQuickDump, quickRatio);

    const std::string fileNameBase = snapshotFile + "_quickMPI";
    const float ratio = quickRatio;
    assert(!quickSync);

    char fn[1024];
    sprintf(fn, "%s_%010.4f.bonsai", fileNameBase.c_str(), t_current);

    const size_t nSnap = localTree.n;
    const size_t dn    = static_cast<size_t>(1.0/ratio);
    assert(dn >= 1);
    size_t nQuick = 0;
    for (size_t i = 0; i < nSnap; i += dn)
      nQuick++;


    header.tCurrent = t_current;
    header.nBodies  = nQuick;
    for (int i = 0; i < 1024; i++)
    {
      header.fileName[i] = fn[i];
      if (fn[i] == 0)
        break;
    }

    data.resize(nQuick);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nSnap; i += dn)
    {
      auto &p = data[i/dn];
      p.x    = localTree.bodies_pos[i].x;
      p.y    = localTree.bodies_pos[i].y;
      p.z    = localTree.bodies_pos[i].z;
      p.mass = localTree.bodies_pos[i].w;
      p.vx   = localTree.bodies_vel[i].x;
      p.vy   = localTree.bodies_vel[i].y;
      p.vz   = localTree.bodies_vel[i].z;
      p.vw   = localTree.bodies_vel[i].w;
      p.rho  = localTree.bodies_dens[i].x;
      p.h    = localTree.bodies_h[i];
      p.ID   = lGetIDType(localTree.bodies_ids[i]);
    }

    static int worldRank = -1;

    static MPI_Request  req[2];
    static MPI_Status status[2];

    int ready2send = 1;
    if (worldRank != -1)
    {
      int ready2sendHeader;
      int ready2sendData;
      MPI_Test(&req[0], &ready2sendHeader, &status[0]);
      MPI_Test(&req[1], &ready2sendData,   &status[1]);
      ready2send = ready2sendHeader && ready2sendData;
    }
    else
    {
      MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    }
    assert(worldRank%2 == 0);

    const int destRank = worldRank + 1;

    int ready2sendGlobal;
    MPI_Allreduce(&ready2send, &ready2sendGlobal, 1, MPI_INT, MPI_MIN, mpiCommWorld);

    if (ready2sendGlobal)
    {
      static BonsaiSharedHeader            header2send;
      static std::vector<BonsaiSharedData> data2send;

      header2send = std::move(header);
      data2send   = std::move(data);

      static int sendCount = 0;
      const int tagBase    = 42;
      MPI_Isend(&header2send,                 1, MPI_Header, destRank, tagBase+2*sendCount+0, MPI_COMM_WORLD, &req[0]);
      MPI_Isend(&data2send[0], data2send.size(), MPI_Data,   destRank, tagBase+2*sendCount+1, MPI_COMM_WORLD, &req[1]);
      sendCount++;
      sendCount = sendCount % 4 ;  /* permit only 4 buffer */
    }
  }//if (t_current >= nextQuickDump && quickDump > 0)
}




/*
 * Function that is called by the output data routines that write sampled or full snapshots
 * the function puts the data in shared memory buffers which are then written by the IO threads.
 */
template<typename THeader, typename TData>
void octree::dumpDataCommon(
    SharedMemoryBase<THeader> &header, SharedMemoryBase<TData> &data,
    const std::string &fileNameBase,
    const float ratio,
    const bool sync)
{
  /********/

  if (sync)
    while (!header[0].done_writing);
  else
  {
    static bool first = true;
    if (first)
    {
      first = false;
      header[0].done_writing = true;
    }
    int ready = header[0].done_writing;
    int readyGlobal;
    MPI_Allreduce(&ready, &readyGlobal, 1, MPI_INT, MPI_MIN, mpiCommWorld);
    if (!readyGlobal)
      return;
  }

  /* write header */

  char fn[1024];
  sprintf(fn, "%s_%010.4f.bonsai", fileNameBase.c_str(), t_current);

  const size_t nSnap = localTree.n;
  const size_t dn = static_cast<size_t>(1.0/ratio);
  assert(dn >= 1);
  size_t nQuick = 0;
  for (size_t i = 0; i < nSnap; i += dn)
    nQuick++;

  header.acquireLock();
  header[0].tCurrent = t_current;
  header[0].nBodies  = nQuick;
  for (int i = 0; i < 1024; i++)
  {
    header[0].fileName[i] = fn[i];
    if (fn[i] == 0)
      break; //TODO replace by strcopy
  }

  data.acquireLock();
  if (!data.resize(nQuick))
  {
    std::cerr << "rank= "   << procId << ": failed to resize. ";
    std::cerr << "Request " << nQuick << " but capacity is  " << data.capacity() << "." << std::endl;
    MPI_Finalize();
    ::exit(0);
  }
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < nSnap; i += dn)
  {
    auto &p = data[i/dn];
    p.x    = localTree.bodies_pos[i].x;
    p.y    = localTree.bodies_pos[i].y;
    p.z    = localTree.bodies_pos[i].z;
    p.mass = localTree.bodies_pos[i].w;
    p.vx   = localTree.bodies_vel[i].x;
    p.vy   = localTree.bodies_vel[i].y;
    p.vz   = localTree.bodies_vel[i].z;
    p.vw   = localTree.bodies_vel[i].w;
//    p.h    = localTree.bodies_h[i];
    p.rho  = localTree.bodies_dens[i].x;
    p.h    = localTree.bodies_dens[i].y;

    p.hydrox = localTree.bodies_hydro[i].x;
    p.hydroy = localTree.bodies_hydro[i].y;
    p.hydroz = localTree.bodies_hydro[i].z;
    p.hydrow = localTree.bodies_hydro[i].w;

    p.drvx = localTree.bodies_grad[i].x;
    p.drvy = localTree.bodies_grad[i].y;
    p.drvz = localTree.bodies_grad[i].z;
    p.drvw = localTree.bodies_grad[i].w;

    p.ID   = lGetIDType(localTree.bodies_ids[i]);
  }
  data.releaseLock();
  header[0].done_writing = false;
  header.releaseLock();

}

/*
 *
 * Send the particle data to a file
 * Works for both full snapshots and quickdump files
 */
void octree::dumpData()
{
  if (shmQHeader == NULL)
  {
    const size_t capacity  = min(4*localTree.n, 24*1024*1024);

    shmQHeader = new ShmQHeader(ShmQHeader::type::sharedFile(procId,sharedPID), 1);
    shmQData   = new ShmQData  (ShmQData  ::type::sharedFile(procId,sharedPID), capacity);

    shmSHeader = new ShmSHeader(ShmSHeader::type::sharedFile(procId,sharedPID), 1);
    shmSData   = new ShmSData  (ShmSData  ::type::sharedFile(procId,sharedPID), capacity);
  }

  if ((t_current >= nextQuickDump && quickDump    > 0) ||
      (t_current >= nextSnapTime  && snapshotIter > 0))
  {
    localTree.bodies_pos.d2h();
    localTree.bodies_vel.d2h();
    localTree.bodies_ids.d2h();
    localTree.bodies_h.d2h();

    localTree.bodies_dens. d2h();
    localTree.bodies_hydro.d2h();
    localTree.bodies_grad. d2h();
  }


  if (t_current >= nextQuickDump && quickDump > 0)
  {
    static bool handShakeQ = false;
    if (!handShakeQ && quickDump > 0 && quickSync)
    {
      auto &header = *shmQHeader;
      header[0].tCurrent     = t_current;
      header[0].done_writing = true;
      lHandShake(header);
      handShakeQ = true;
    }

    nextQuickDump += quickDump;
    nextQuickDump = std::max(nextQuickDump, t_current);
    if (procId == 0) fprintf(stderr, "-- quickdump: nextQuickDump= %g  quickRatio= %g\n", nextQuickDump, quickRatio);
    dumpDataCommon(*shmQHeader, *shmQData, snapshotFile + "_quick", quickRatio, quickSync);
  }

  if (t_current >= nextSnapTime && snapshotIter > 0)
  {
    static bool handShakeS = false;
    if (!handShakeS && snapshotIter > 0)
    {
      auto &header           = *shmSHeader;
      header[0].tCurrent     = t_current;


      // Domain information
      header[0].xmin = this->periodicDomainInfo.minrange.x;
      header[0].ymin = this->periodicDomainInfo.minrange.y;
      header[0].zmin = this->periodicDomainInfo.minrange.z;
      header[0].xmax = this->periodicDomainInfo.maxrange.x;
      header[0].ymax = this->periodicDomainInfo.maxrange.y;
      header[0].zmax = this->periodicDomainInfo.maxrange.z;
      header[0].periodicity = this->periodicDomainInfo.domainSize.w;

      header[0].done_writing = true;
      lHandShake(header);
      handShakeS             = true;
    }

    nextSnapTime += snapshotIter;
    nextSnapTime = std::max(nextSnapTime, t_current);
    if (procId == 0)  fprintf(stderr, "-- snapdump: nextSnapDump= %g  %d\n", nextSnapTime, localTree.n);

    dumpDataCommon(
        *shmSHeader, *shmSData,
        snapshotFile,
        1.0,  /* fraction of particles to store */
        true  /* force sync between IO and simulator */);
  }
}




    /*
     *
     * Functions to read the MPI-IO based file format
     *
     *
     */

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
                    fprintf(stderr, " Read %lld of type %s\n", nGlb, type->getName().c_str());
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

    void octree::lReadBonsaiFile(std::vector<real4 > &bodyPositions,
                                 std::vector<real4 > &bodyVelocities,
                                 std::vector<ullong> &bodyIDs,
                                 std::vector<real2>  &bodyDensity,
                                 std::vector<real4>  &bodyHydro,
                                 std::vector<real4>  &bodyDrv,
                                 float &t_current,
                                 float3 &domain_low,
                                 float3 &domain_high,
                                 int &periodicity,
                                 const std::string &fileName,
                                 const int rank, const int nrank,
                                 const MPI_Comm &comm,
                                 const bool restart,
                                 const int reduceFactor)
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
        using Hydro  = BonsaiIO::DataType<float4>;
        using Drvt   = BonsaiIO::DataType<float4>;

        data.push_back(new IDType("DM:IDType"));
        data.push_back(new Pos   ("DM:POS:real4"));
        data.push_back(new Vel   ("DM:VEL:float[3]"));
        data.push_back(new RhoH  ("DM:RHOH:float[2]"));
        data.push_back(new Hydro ("DM:HYDRO:float[4]"));
        data.push_back(new Drvt  ("DM:DRVT:float[4]"));

        data.push_back(new IDType("Stars:IDType"));
        data.push_back(new Pos   ("Stars:POS:real4"));
        data.push_back(new Vel   ("Stars:VEL:float[3]"));
        data.push_back(new RhoH  ("Stars:RHOH:float[2]"));
        data.push_back(new Hydro ("Stars:HYDRO:float[4]"));
        data.push_back(new Drvt  ("Stars:DRVT:float[4]"));

        const double dtRead = lReadBonsaiFields(rank, comm, data, *in, reduceFactor, restart);

        const auto &DM_IDType = lBonsaiSafeCast<IDType>(data[0]);
        const auto &DM_Pos    = lBonsaiSafeCast<Pos   >(data[1]);
        const auto &DM_Vel    = lBonsaiSafeCast<Vel   >(data[2]);
        const auto &DM_RhoH   = lBonsaiSafeCast<RhoH  >(data[3]);
        const auto &DM_Hydro  = lBonsaiSafeCast<Hydro >(data[4]);
        const auto &DM_Drvt   = lBonsaiSafeCast<Drvt  >(data[5]);

        const auto &S_IDType  = lBonsaiSafeCast<IDType>(data[6]);
        const auto &S_Pos     = lBonsaiSafeCast<Pos   >(data[7]);
        const auto &S_Vel     = lBonsaiSafeCast<Vel   >(data[8]);
        const auto &S_RhoH    = lBonsaiSafeCast<RhoH  >(data[9]);
        const auto &S_Hydro   = lBonsaiSafeCast<Hydro >(data[10]);
        const auto &S_Drvt    = lBonsaiSafeCast<Drvt  >(data[11]);

        const size_t nDM = DM_IDType.size();
        assert(nDM == DM_Pos.size());
        assert(nDM == DM_Vel.size());

        const size_t nS = S_IDType.size();
        assert(nS == S_Pos.size());
        assert(nS == S_Vel.size());


        //NFirst  = static_cast<std::remove_reference<decltype(NFirst )>::type>(nDM);
        //NSecond = static_cast<std::remove_reference<decltype(NSecond)>::type>(nS);


        bodyPositions.resize(nDM+nS);
        bodyVelocities.resize(nDM+nS);
        bodyIDs.resize(nDM+nS);
        bodyDensity.resize(nDM+nS);
        bodyHydro.resize(nDM+nS);
        bodyDrv.resize(nDM+nS);

        /* store DM */

        constexpr int ntypecount = 10;
        std::array<size_t,ntypecount> ntypeloc, ntypeglb;
        std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
        for (int i = 0; i < nDM; i++)
        {
            ntypeloc[0]++;
            auto &pos  = bodyPositions[i];
            auto &vel  = bodyVelocities[i];
            auto &ID   = bodyIDs[i];
            auto &RhoH = bodyDensity[i];
            auto &Hydr = bodyHydro[i];
            auto &Drvt = bodyDrv[i];

            pos       = DM_Pos[i];
            pos.w    *= reduceFactor;
            vel       = make_float4(DM_Vel[i][0], DM_Vel[i][1], DM_Vel[i][2],0.0f);
            ID        = DM_IDType[i].getID() + DARKMATTERID;

            RhoH      = make_float2(DM_RhoH[i][0], DM_RhoH[i][1]);
            Hydr      = DM_Hydro[i];
            Drvt      = DM_Drvt[i];
        }

        for (int i = 0; i < nS; i++)
        {
            auto &pos = bodyPositions[nDM+i];
            auto &vel = bodyVelocities[nDM+i];
            auto &ID  = bodyIDs[nDM+i];
            auto &RhoH = bodyDensity[nDM+i];
            auto &Hydr = bodyHydro[nDM+i];
            auto &Drvt = bodyDrv[nDM+i];

            pos    = S_Pos[i];
            pos.w *= reduceFactor;
            vel    = make_float4(S_Vel[i][0], S_Vel[i][1], S_Vel[i][2],0.0f);

            RhoH   = make_float2(S_RhoH[i][0],S_RhoH[i][1]);
            Hydr   = S_Hydro[i];
            Drvt   = S_Drvt[i];

            ID     = S_IDType[i].getID();
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
            //tree->set_t_current(0);
            t_current = 0;
        }
        else{
            //tree->set_t_current(static_cast<float>(in->getTime()));
            t_current = static_cast<float>(in->getTime());
        }

        //Read boundaries, if those are not available defaults will be set automatically
        in->getDomainInfo(domain_low.x,  domain_low.y,  domain_low.z,
             domain_high.x, domain_high.y, domain_high.z, periodicity);


        in->close();
        const double bw = in->computeBandwidth()/1e6;
        for (auto d : data)
            delete d;
        delete in;
        if (rank == 0)
            fprintf(stderr, " :: dtRead= %g  sec readBW= %g MB/s \n", dtRead, bw);
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
        long long int nLoc = type->getNumElements();
        long long int nGlb;
        MPI_Allreduce(&nLoc, &nGlb, 1, MPI_DOUBLE, MPI_SUM, comm);
        if (nGlb > 0)
        {
          bool res = out.write(*type);
          assert(res);
          //fprintf(stderr,"Wrote %lld %lld %d\n", nGlb, nLoc, res);
        }
        dtWrite += MPI_Wtime() - t0;
      }

      return dtWrite;
    }



    template<typename ShmHeader, typename ShmData>
    bool writeBonsaiFileLoop(ShmHeader &header, ShmData &data, const int rank, const int nrank, const MPI_Comm &comm)
    {
      double tLast = -1;

      /* handshake */
      header.acquireLock();
      header[0].handshake = true;
      header.releaseLock();

      while (header[0].handshake)
        usleep(1000);

      header.acquireLock();
      header[0].handshake = true;
      header.releaseLock();
      /* handshake complete */

      while (1)
      {
        while (header[0].done_writing);

        assert(tLast != header[0].tCurrent);
        tLast = header[0].tCurrent;

        if (header[0].tCurrent == -1.0)
          break;

        header.acquireLock();
        const float tCurrent = header[0].tCurrent;
        const size_t nBodies = header[0].nBodies;
        char fn[1024];
        for (int i = 0; i < 1024; i++)
        {
          fn[i] = header[0].fileName[i];
          if (fn[i] == 0)
            break;
        }

        data.acquireLock();
        const size_t size = data.size();
        assert(size == nBodies);

        /* write data */
        {
          const double tBeg  = MPI_Wtime();
          const double tOpen = MPI_Wtime();
          BonsaiIO::Core out(rank, nrank, comm, BonsaiIO::WRITE, fn);
          double dtOpen = MPI_Wtime() - tOpen;

          out.setTime(tCurrent);
          out.setDomainInfo(header[0].xmin,header[0].ymin,header[0].zmin,
                            header[0].xmax,header[0].ymax,header[0].zmax,
                            header[0].periodicity);
          tLast = tCurrent;

          /* prepare data */

          size_t nDM = 0, nS = 0;
          constexpr int ntypecount = 10;
          std::array<size_t,ntypecount> ntypeloc, ntypeglb;
          std::fill(ntypeloc.begin(), ntypeloc.end(), 0);
          for (size_t i = 0; i < size; i++)
          {
            const int type = data[i].ID.getType();
            if  (type < ntypecount)
              ntypeloc[type]++;
            switch(type)
            {
              case 0:
                nDM++;
                break;
              default:
                nS++;
            }
          }

          MPI_Reduce(&ntypeloc, &ntypeglb, ntypecount, MPI_LONG_LONG, MPI_SUM, 0, comm);
          if (rank == 0)
          {
            fprintf(stderr, "BonsaiIO:: writing to %s \n", fn);
            for (int type = 0; type < ntypecount; type++)
              if (ntypeglb[type] > 0)
                fprintf(stderr, " BonsaiIO:: ptype= %d:  np= %zu \n",type, ntypeglb[type]);
          }

          typedef float float4[4];
          typedef float float3[3];
          typedef float float2[2];
          BonsaiIO::DataType<IDType> DM_id ("DM:IDType",        nDM);
          BonsaiIO::DataType<float4> DM_pos("DM:POS:real4",     nDM);
          BonsaiIO::DataType<float3> DM_vel("DM:VEL:float[3]",  nDM);
          BonsaiIO::DataType<float2> DM_rhoh("DM:RHOH:float[2]", nDM);
          BonsaiIO::DataType<float4> DM_hydro("DM:HYDRO:float[4]", nDM);
          BonsaiIO::DataType<float4> DM_drv("DM:DRVT:float[4]"   , nDM);


          BonsaiIO::DataType<IDType> S_id ("Stars:IDType",           nS);
          BonsaiIO::DataType<float4> S_pos("Stars:POS:real4",        nS);
          BonsaiIO::DataType<float3> S_vel("Stars:VEL:float[3]",     nS);
          BonsaiIO::DataType<float2> S_rhoh("Stars:RHOH:float[2]",   nS);
          BonsaiIO::DataType<float4> S_hydro("Stars:HYDRO:float[4]", nS);
          BonsaiIO::DataType<float4> S_drv  ("Stars:DRVT:float[4]" , nS);

          size_t iDM = 0, iS = 0;
          for (size_t i = 0; i < size; i++)
          {
            assert(iDM + iS == i);
            switch (data[i].ID.getType())
            {
              case 0:
                DM_id  [iDM]    = data[i].ID;
                DM_pos [iDM][0] = data[i].x;
                DM_pos [iDM][1] = data[i].y;
                DM_pos [iDM][2] = data[i].z;
                DM_pos [iDM][3] = data[i].mass;
                DM_vel [iDM][0] = data[i].vx;
                DM_vel [iDM][1] = data[i].vy;
                DM_vel [iDM][2] = data[i].vz;
                DM_rhoh[iDM][0] = data[i].rho;
                DM_rhoh[iDM][1] = data[i].h;

                DM_hydro[iDM][0] = data[i].hydrox;
                DM_hydro[iDM][1] = data[i].hydroy;
                DM_hydro[iDM][0] = data[i].hydroz;
                DM_hydro[iDM][1] = data[i].hydrow;
                DM_drv[iDM][0]   = data[i].drvx;
                DM_drv[iDM][1]   = data[i].drvy;
                DM_drv[iDM][0]   = data[i].drvz;
                DM_drv[iDM][1]   = data[i].drvw;


                iDM++;
                assert(iDM <= nDM);
                break;
              default:
                S_id  [iS]    = data[i].ID;
                S_pos [iS][0] = data[i].x;
                S_pos [iS][1] = data[i].y;
                S_pos [iS][2] = data[i].z;
                S_pos [iS][3] = data[i].mass;
                S_vel [iS][0] = data[i].vx;
                S_vel [iS][1] = data[i].vy;
                S_vel [iS][2] = data[i].vz;
                S_rhoh[iS][0] = data[i].rho;
                S_rhoh[iS][1] = data[i].h;

                S_hydro[iS][0] = data[i].hydrox;
                S_hydro[iS][1] = data[i].hydroy;
                S_hydro[iS][2] = data[i].hydroz;
                S_hydro[iS][3] = data[i].hydrow;
                S_drv[iS][0]   = data[i].drvx;
                S_drv[iS][1]   = data[i].drvy;
                S_drv[iS][2]   = data[i].drvz;
                S_drv[iS][3]   = data[i].drvw;

                iS++;
                assert(iS <= nS);
            }
          }

          for (int i = 0; i < nDM; i++)
            assert(DM_id[i].getType() == 0);
          for (int i = 0; i < nS; i++)
            assert(S_id[i].getType() > 0);

          const double dtWrite = write(rank, comm,
              {
                &DM_id, &DM_pos, &DM_vel, &DM_rhoh, &DM_hydro, &DM_drv,
                &S_id, &S_pos, &S_vel, &S_rhoh, &S_hydro, &S_drv
              }, out);

          const double tClose = MPI_Wtime();
          out.close();
          double dtClose = MPI_Wtime() - tClose;

          const double writeBW = out.computeBandwidth();
          const double tEnd = MPI_Wtime();

          long long nGlb[2], nLoc[2];
          nLoc[0] = nDM;
          nLoc[1] = nS;
          MPI_Reduce(nLoc, nGlb, 2, MPI_LONG_LONG, MPI_SUM, 0, comm);


          if (rank == 0)
            fprintf(stderr, " BonsaiIO:: total= %g sec nDM= %gM  nS= %gM [open= %g  write= %g close= %g] BW= %g MB/s \n",
                tEnd-tBeg, nGlb[0]/1e6, nGlb[1]/1e6, dtOpen, dtWrite, dtClose, writeBW/1e6);
        }
        header[0].done_writing = true;

        data.releaseLock();
        header.releaseLock();
      }
    }

    void octree::writeSharedMemoryLoop(const int procId, const int nProcs, const int shrMemPID, const MPI_Comm &ioComm)
    {
        using ShmSHeader = SharedMemoryClient<BonsaiSharedSnapHeader>;
        using ShmSData   = SharedMemoryClient<BonsaiSharedSnapData>;
        ShmSHeader shmSHeader(ShmSHeader::type::sharedFile(procId, shrMemPID));
        ShmSData   shmSData  (ShmSData  ::type::sharedFile(procId, shrMemPID));
        writeBonsaiFileLoop(shmSHeader, shmSData, procId, nProcs, ioComm);
    }



    #endif

void octree::dumpTreeStructureToFile(tree_structure &tree)
{

    //Write the tree structure to file
    tree.multipole.d2h();
    tree.boxSizeInfo.d2h();
    tree.boxCenterInfo.d2h();
    char fileName[256];
    for(int i=0; i < tree.n_levels; i++)
    {

      sprintf(fileName, "fullTreeStructure-%d-level-%d.txt", mpiGetRank(), i);
      ofstream nodeFile;
      nodeFile.open(fileName);
      nodeFile << "NODES" << endl;
      for(int j=tree.level_list[i].x; j < tree.level_list[i].y; j++)
      {
          nodeFile << tree.boxCenterInfo[j].x << "\t" << tree.boxCenterInfo[j].y << "\t" << tree.boxCenterInfo[j].z;
          nodeFile << "\t"  << tree.boxSizeInfo[j].x << "\t" << tree.boxSizeInfo[j].y << "\t" << tree.boxSizeInfo[j].z << "\n";
      }//for j
      nodeFile.close();
    }//for i

    //Write the source particles
    sprintf(fileName, "fullTreeStructureParticles-%d.txt", mpiGetRank());
    ofstream partFile;
    partFile.open(fileName);
    tree.bodies_Ppos.d2h();
    partFile << "POINTS\n";
    for(int i=0; i < tree.n; i++)
    {
      float4  pos =  tree.bodies_Ppos[i];
      partFile << pos.x << "\t" << pos.y << "\t" << pos.z << endl;
    }
    partFile.close();

    //Write the groups
    tree.groupCenterInfo.d2h();
    tree.groupSizeInfo.d2h();
    sprintf(fileName, "fullTreeStructureGroups-%d.txt", mpiGetRank());
    FILE *out = fopen(fileName, "w");
    fprintf(out,"NODES\n");
    for(int j= 0; j < tree.n_groups; j++)
    {
      fprintf(out, "%f %f %f %f %f %f \n",
          tree.groupCenterInfo[j].x,tree.groupCenterInfo[j].y,tree.groupCenterInfo[j].z,
          tree.groupSizeInfo[j].x,tree.groupSizeInfo[j].y,tree.groupSizeInfo[j].z);
    }
    fclose(out);

#if 0
  tree.boxSizeInfo.d2h();
  tree.boxCenterInfo.d2h();

  FILE *f = fopen("nodes.txt", "w");
  for(int i=0; i < tree.n_nodes; i++)
  {

      if(
          tree.boxCenterInfo[i].x-tree.boxSizeInfo[i].x <= 8 &&
          tree.boxCenterInfo[i].x+tree.boxSizeInfo[i].x >= 8 &&
          tree.boxCenterInfo[i].y-tree.boxSizeInfo[i].y <= 8 &&
          tree.boxCenterInfo[i].y+tree.boxSizeInfo[i].y >= 8 &&
          tree.boxCenterInfo[i].z-tree.boxSizeInfo[i].z <= 77 &&
          tree.boxCenterInfo[i].z+tree.boxSizeInfo[i].z >= 77)
      {
    fprintf(f,"FOUND Box: %d  [ %lg %lg %lg ] size: [ %lg %lg %lg ] node: %d \n",
            i,
            tree.boxCenterInfo[i].x,
            tree.boxCenterInfo[i].y,
            tree.boxCenterInfo[i].z,
            tree.boxSizeInfo[i].x,
            tree.boxSizeInfo[i].y,
            tree.boxSizeInfo[i].z,
            tree.boxCenterInfo[i].w > 0.0);
  }





    fprintf(f,"Box: %d  [ %lg %lg %lg ] size: [ %lg %lg %lg ] node: %d \n",
            i,
            tree.boxCenterInfo[i].x,
            tree.boxCenterInfo[i].y,
            tree.boxCenterInfo[i].z,
            tree.boxSizeInfo[i].x,
            tree.boxSizeInfo[i].y,
            tree.boxSizeInfo[i].z,
            tree.boxCenterInfo[i].w > 0.0);
  }
  fclose(f);
#endif

#if 0


if(iter == 20)
{
   char fileName[256];
    sprintf(fileName, "groups-%d.bin", mpiGetRank());
    ofstream nodeFile;
    nodeFile.open(fileName, ios::out | ios::binary);
    if(nodeFile.is_open())
    {
      nodeFile.write((char*)&tree.n_groups, sizeof(int));

      for(int i=0; i < tree.n_groups; i++)
      {
        nodeFile.write((char*)&tree.groupSizeInfo[i],  sizeof(real4)); //size
        nodeFile.write((char*)&tree.groupCenterInfo[i], sizeof(real4)); //center
      }
    }
  }

 //Write the tree-structure
 if(iter == 20)
 {
   tree.multipole.d2h();
  tree.boxSizeInfo.d2h();
  tree.boxCenterInfo.d2h();
  tree.bodies_Ppos.d2h();

    char fileName[256];
    sprintf(fileName, "fullTreeStructure-%d.bin", mpiGetRank());
    ofstream nodeFile;
    //nodeFile.open(nodeFileName.c_str());
    nodeFile.open(fileName, ios::out | ios::binary);
    if(nodeFile.is_open())
    {
      uint2 node_begend;
      int level_start = tree.startLevelMin;
      node_begend.x   = tree.level_list[level_start].x;
      node_begend.y   = tree.level_list[level_start].y;

      nodeFile.write((char*)&node_begend.x, sizeof(int));
      nodeFile.write((char*)&node_begend.y, sizeof(int));
      nodeFile.write((char*)&tree.n_nodes, sizeof(int));
      nodeFile.write((char*)&tree.n, sizeof(int));

      for(int i=0; i < tree.n; i++)
      {
        nodeFile.write((char*)&tree.bodies_Ppos[i], sizeof(real4));
      }

      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.multipole[3*i+0], sizeof(real4));
        nodeFile.write((char*)&tree.multipole[3*i+1], sizeof(real4));
        nodeFile.write((char*)&tree.multipole[3*i+2], sizeof(real4));;
      }

      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.boxSizeInfo[i], sizeof(real4));
      }
      for(int i=0; i < tree.n_nodes; i++)
      {
        nodeFile.write((char*)&tree.boxCenterInfo[i], sizeof(real4));
      }

      nodeFile.close();
    }
}
#endif

}




