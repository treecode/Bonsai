#pragma once


#include <algorithm>
#include <cassert>
#include <vector>

#ifdef USE_MPI
  #include <mpi.h>
#else
//  #define MPI_Comm int
    typedef int MPI_Comm;
#endif

struct LOGFILEWRITER
{
  private:

  int procId, nProc, fileId;
  const int fullNProc;

#ifdef USE_MPI
  const MPI_Comm mpi_comm;
#else
  const int mpi_comm;
#endif


  ofstream logFile;


  private:
  
    void GatherLogData(const std::string &logData, std::string &fullLog)
    {
      #ifdef USE_MPI
            //Gather sizes
            std::vector<int> logSizes(nProc);
            int logSize = logData.size();
            MPI_Gather(&logSize, 1, MPI_INT, &logSizes[0], 1, MPI_INT, 0, mpi_comm);

            //Compute displacements
            std::vector<int> logDispl(nProc+1,0);
            for (int i = 0; i < nProc; i++)
              logDispl[i+1] = logDispl[i] + logSizes[i];

            //Receive the data
            const int logRecv = logDispl[nProc];
            fullLog.resize(logRecv);
            MPI_Gatherv(
                (void*)&logData[0], logSize, MPI_BYTE,
                (void*)&fullLog[0], &logSizes[0], &logDispl[0], MPI_BYTE, 0, mpi_comm);
      #else
            fullLog = logData;
      #endif
    }


  public:

  /* sample_keys must be sorted by Key in an increaing order, otherwise
   * assignKeyToProc will fail  */
  LOGFILEWRITER(const int _fullNProc,
                const MPI_Comm &_mpi_comm,
                const MPI_Comm &_mpi_comm2) :
                fullNProc(_fullNProc),mpi_comm(_mpi_comm)
  {
    //Get the local rank and number of ranks
    #ifdef USE_MPI
        MPI_Comm_rank (_mpi_comm, &procId);
        MPI_Comm_size (_mpi_comm, &nProc);
        MPI_Comm_rank (_mpi_comm2, &fileId);
    #else
        procId = 0;
        nProc  = 1;
        fileId = 0;
    #endif

    char fileName[64];
    sprintf(fileName, "gpuLog.log-%d-%d", fullNProc, fileId);

    logFile.open(fileName);
  }

  ~LOGFILEWRITER()
  {
    logFile.close();
  }

  void updateLogData(const std::string &logData)
  {
    //Write the data
    std::string fullLog;
    GatherLogData(logData, fullLog);
    if(procId == 0) {
      logFile <<  fullLog;
    }
  }


};
