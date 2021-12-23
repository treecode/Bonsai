#include "octree.h"

#include "FileIO.h"

#include "SharedMemory.h"
#include "BonsaiSharedData.h"

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
    p.rho  = localTree.bodies_dens[i].x;
    p.h    = localTree.bodies_h[i];
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
    localTree.bodies_dens.d2h();
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
     * Functions to read the MPI-IO based file format     [GL ignoring this for accelerations]
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
                                 float &t_current,
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


        //NFirst  = static_cast<std::remove_reference<decltype(NFirst )>::type>(nDM);
        //NSecond = static_cast<std::remove_reference<decltype(NSecond)>::type>(nS);


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
            //tree->set_t_current(0);
            t_current = 0;
        }
        else{
            //tree->set_t_current(static_cast<float>(in->getTime()));
            t_current = static_cast<float>(in->getTime());
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


