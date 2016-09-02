#include "octree.h"

#include "FileIO.h"

#include "SharedMemory.h"
#include "BonsaiSharedData.h"

#ifndef WIN32
#include <sys/time.h>
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

#endif

void octree::write_snapshot_per_process(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds,
                                        int n, string fileName, float time)
{
    NTotal = n;
    NFirst = NSecond = NThird = 0;

    for(int i=0; i < n; i++)
    {
      if(bodyIds[i] >= DISKID  && bodyIds[i] < BULGEID)       NThird++;
      if(bodyIds[i] >= BULGEID && bodyIds[i] < DARKMATTERID)  NSecond++;
      if(bodyIds[i] >= DARKMATTERID)                          NFirst++;
    }

    //Sync them to process 0
    int NCombTotal, NCombFirst, NCombSecond, NCombThird;
    NCombTotal  = (NTotal);
    NCombFirst  = (NFirst);
    NCombSecond = (NSecond);
    NCombThird  = (NThird);

    //Since the bulge and disk particles are star particles, lets add them
    NCombSecond += NCombThird;


    ofstream outputFile;
    outputFile.open(fileName.c_str(), ios::out | ios::binary);

    dumpV2  h;

    if(!outputFile.is_open())
    {
      cout << "Can't open output file: "<< fileName << std::endl;
      ::exit(0);
    }

    //Create Tipsy header
    h.time    = time;
    h.nbodies = NCombTotal;
    h.ndim    = 3;
    h.ndark   = NCombFirst;
    h.nstar   = NCombSecond;    //In case of disks we have to finish this
    h.nsph    = 0;
    h.version = 2;
    outputFile.write((char*)&h, sizeof(h));

    //First write the dark matter particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(bodyIds[i] >= DARKMATTERID)
      {
        //Set particle properties
        dark_particleV2 d;
        //d.eps = bodyVelocities[i].w;
        d.mass = bodyPositions[i].w;
        d.pos[0] = bodyPositions[i].x;
        d.pos[1] = bodyPositions[i].y;
        d.pos[2] = bodyPositions[i].z;
        d.vel[0] = bodyVelocities[i].x;
        d.vel[1] = bodyVelocities[i].y;
        d.vel[2] = bodyVelocities[i].z;
        d.setID(bodyIds[i]);      //Custom change to Tipsy format

        outputFile.write((char*)&d, sizeof(d));
      } //end if
    } //end i loop

    //Next write the star particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(bodyIds[i] < DARKMATTERID)
      {
        //Set particle properties
        star_particleV2 s;
        //s.eps = bodyVelocities[i].w;
        s.mass = bodyPositions[i].w;
        s.pos[0] = bodyPositions[i].x;
        s.pos[1] = bodyPositions[i].y;
        s.pos[2] = bodyPositions[i].z;
        s.vel[0] = bodyVelocities[i].x;
        s.vel[1] = bodyVelocities[i].y;
        s.vel[2] = bodyVelocities[i].z;
        s.setID(bodyIds[i]);      //Custom change to tipsy format

        s.metals = 0;
        s.tform = 0;
        outputFile.write((char*)&s, sizeof(s));
      }

    } //end i loop

    outputFile.close();

    LOGF(stderr,"Wrote %d bodies to tipsy file \n", NCombTotal);

}

void octree::write_dumbp_snapshot_parallel_tipsyV2(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds, int n, string fileName,
                                                 int NCombTotal, int NCombFirst, int NCombSecond, int NCombThird, float time)
{
  //Rank 0 does the writing
  if(mpiGetRank() == 0)
  {
    ofstream outputFile;
    outputFile.open(fileName.c_str(), ios::out | ios::binary);

    dumpV2  h;

    if(!outputFile.is_open())
    {
      cout << "Can't open output file: "<< fileName << std::endl;
      ::exit(0);
    }

    //Create Tipsy header
    h.time    = time;
    h.nbodies = NCombTotal;
    h.ndim    = 3;
    h.ndark   = NCombFirst;
    h.nstar   = NCombSecond;    //In case of disks we have to finish this
    h.nsph    = 0;
    h.version = 2;

    outputFile.write((char*)&h, sizeof(h));

    //Buffer to store complete snapshot
    vector<real4>   allPositions;
    vector<real4>   allVelocities;
    vector<ullong>  allIds;

    allPositions.insert(allPositions.begin(), &bodyPositions[0], &bodyPositions[n]);
    allVelocities.insert(allVelocities.begin(), &bodyVelocities[0], &bodyVelocities[n]);
    allIds.insert(allIds.begin(), &bodyIds[0], &bodyIds[n]);

    //Now receive the data from the other processes
    vector<real4>   extPositions;
    vector<real4>   extVelocities;
    vector<ullong>  extIds;

    for(int recvFrom=1; recvFrom < mpiGetNProcs(); recvFrom++)
    {
      ICRecv(recvFrom, extPositions, extVelocities,  extIds);

      allPositions.insert(allPositions.end(), extPositions.begin(), extPositions.end());
      allVelocities.insert(allVelocities.end(), extVelocities.begin(), extVelocities.end());
      allIds.insert(allIds.end(), extIds.begin(), extIds.end());
    }


    //First write the dark matter particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(allIds[i] >= DARKMATTERID)
      {
        //Set particle properties
        dark_particleV2 d;
        d.mass   = allPositions[i].w;
        d.pos[0] = allPositions[i].x;
        d.pos[1] = allPositions[i].y;
        d.pos[2] = allPositions[i].z;
        d.vel[0] = allVelocities[i].x;
        d.vel[1] = allVelocities[i].y;
        d.vel[2] = allVelocities[i].z;
        d.setID(allIds[i]);
        outputFile.write((char*)&d, sizeof(d));
      } //end if
    } //end i loop


    //Next write the star particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(allIds[i] < DARKMATTERID)
      {
        //Set particle properties
        star_particleV2 s;
        s.mass = allPositions[i].w;
        s.pos[0] = allPositions[i].x;
        s.pos[1] = allPositions[i].y;
        s.pos[2] = allPositions[i].z;
        s.vel[0] = allVelocities[i].x;
        s.vel[1] = allVelocities[i].y;
        s.vel[2] = allVelocities[i].z;
        s.setID(allIds[i]);
        s.metals = 0;
        s.tform = 0;
        outputFile.write((char*)&s, sizeof(s));
     } //end if
    } //end i loop

    outputFile.close();
  }
  else
  {
    //All other ranks send their data to process 0
    ICSend(0,  bodyPositions, bodyVelocities,  bodyIds, n);
  }
}

void octree::write_dumbp_snapshot_parallel(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds, int n, string fileName, float time)
{
    NTotal = n;
    NFirst = NSecond = NThird = 0;
    
    for(int i=0; i < n; i++)
    { 
      //Specific for JB his files
      if(bodyIds[i] >= DISKID  && bodyIds[i] < BULGEID)       NThird++;
      if(bodyIds[i] >= BULGEID && bodyIds[i] < DARKMATTERID)  NSecond++;
      if(bodyIds[i] >= DARKMATTERID)                          NFirst++;
    }
    
    //Sync them to process 0
    int NCombTotal, NCombFirst, NCombSecond, NCombThird;
    NCombTotal  = SumOnRootRank(NTotal);
    NCombFirst  = SumOnRootRank(NFirst);
    NCombSecond = SumOnRootRank(NSecond);
    NCombThird  = SumOnRootRank(NThird);
    
    //Since the dust and disk particles are star particles
    //lets add them
    NCombSecond += NCombThird;    

    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());
    string tempName; tempName.assign(fullFileName);
    write_dumbp_snapshot_parallel_tipsyV2(bodyPositions, bodyVelocities, bodyIds, n, tempName,
                                        NCombTotal, NCombFirst, NCombSecond, NCombThird, time);
    return;

};


