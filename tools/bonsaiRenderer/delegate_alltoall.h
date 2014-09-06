#pragma once

#include <mpi.h>
#include <assert.h>

#define GANGED_ALLTOALL
#ifdef GANGED_ALLTOALL
#endif
// #include "BHtree.h"

#if 0
#ifndef BHTREE_H
struct float4{
  typedef float  v4sf __attribute__ ((vector_size(16)));
  v4sf val;
};

struct nbody_particle{
  typedef float  v4sf __attribute__ ((vector_size(16)));
  v4sf dum[6];
};
#endif
#endif

struct MYMPI
{

  int nproc_pre_node = 4;

  MPI_Comm comm_world;
  MPI_Comm comm_global;
  MPI_Comm comm_local;

  int myid_world;
  bool isroot;
  int myid_global;
  int myid_local;

  int size_world;
  int size_global;
  int size_local;

  enum {
    MAX_WORLD = 2048,
    MAX_GLOBAL =  256, // num of node
    MAX_LOCAL  =    8  // nprocs per node
  };
#if 0
  const int MAX_WORLD  = 2048; // total procs
  const int MAX_GLOBAL =  256; // num of node
  const int MAX_LOCAL  =    8; // nprocs per node
#endif

  MPI_Datatype MPI_FLOAT3   = 0;
  MPI_Datatype MPI_FLOAT4   = 0;
  MPI_Datatype MPI_PARTICLE = 0;

  int firstcall = true;

  void initialize(
      const int nproc = 4,
      const MPI_Comm comm = MPI_COMM_WORLD)
  {
    nproc_pre_node = nproc;

    //initialize communicators
    comm_world = comm;
    MPI_Comm_rank(comm_world, &myid_world);
    MPI_Comm_size(comm_world, &size_world);
    isroot = (myid_world == 0);
    assert(size_world % nproc_pre_node == 0);
    assert(size_world <= MAX_WORLD);

    MPI_Comm_split(
        comm_world, 
        myid_world / nproc_pre_node, // color
        myid_world % nproc_pre_node, // key
        &comm_local);
    MPI_Comm_rank(comm_local, &myid_local);
    MPI_Comm_size(comm_local, &size_local);
    assert(size_local <= MAX_LOCAL);

    MPI_Comm_split(
        comm_world, 
        myid_world % nproc_pre_node, // color
        myid_world / nproc_pre_node, // key
        &comm_global);

    MPI_Comm_rank(comm_global, &myid_global);
    MPI_Comm_size(comm_global, &size_global);
    assert(size_global <= MAX_GLOBAL);

    assert(size_world == size_global * size_local);

#if 0
    //initialize datatypes
    MPI_Type_contiguous(3, MPI_FLOAT, &MPI_FLOAT3);
    MPI_Type_commit(&MPI_FLOAT3);

    MPI_Type_contiguous(4, MPI_FLOAT, &MPI_FLOAT4);
    MPI_Type_commit(&MPI_FLOAT4);

    int ss = sizeof(nbody_particle) / sizeof(double);
    assert(0 == sizeof(nbody_particle) % sizeof(double));
    MPI_Type_contiguous(ss, MPI_DOUBLE, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);
#endif

    firstcall = false;
  }
  template <typename T>
    MPI_Datatype datatype();


#if 0
  template <> MPI_Datatype datatype<int> (){ return MPI_INT; }

  template <> MPI_Datatype datatype<float3> (){ return MPI_FLOAT3; }

  template <> MPI_Datatype datatype<float4> (){ return MPI_FLOAT4; }

  template <> MPI_Datatype datatype<nbody_particle> (){ return MPI_PARTICLE; }
#endif

#if 0
  template <typename T>
    int ganged_alltoall(){
      if(firstcall) initialize();
      int ierr;
      return ierr;
    }
#endif

#include <vector>

  // static std::ofstream nullstr("/dev/null");

  template <typename T>
    int detuned_alltoallv(
        std::vector<T> sendvec[],
        std::vector<T> recvvec[],
        const int block = 2){
      if(firstcall) initialize();

      int ierr = 0;
      int myid   = myid_world;
      int nprocs = size_world;
      static MPI_Status stat[MAX_WORLD];
      static MPI_Request req[MAX_WORLD*2];
      static int nrecv[MAX_WORLD];

      for(int is=0; is<block; is++){
        for(int id=0; id<block; id++){
          static char str[128];
          sprintf(str, "alltoall %d %d : ", is, id);
//          RAII_Timer timer(str, NULL, std::cout, isroot);
          int nreq = 0;
          // first exchange the count
          if(is == myid%block){
            for(int p=id; p<nprocs; p+=block){
              int nsend = sendvec[p].size();
              ierr |= MPI_Isend(&nsend, 1, MPI_INT, p, 0, comm_world, &req[nreq++]);
            }
          }
          if(id == myid%block){
            for(int p=is; p<nprocs; p+=block){
              ierr |= MPI_Irecv(&nrecv[p], 1, MPI_INT, p, 0, comm_world, &req[nreq++]);
            }
          }
          ierr |= MPI_Waitall(nreq, req, stat);

          // then exchange the data
          nreq = 0;
          if(is == myid%block){
            for(int p=id; p<nprocs; p+=block){
              int nsend = sendvec[p].size();
              ierr |= MPI_Isend(&sendvec[p][0], nsend, datatype<T>(), p, 0, comm_world, &req[nreq++]);
            }
          }
          if(id == myid%block){
            for(int p=is; p<nprocs; p+=block){
              // printf("%d %d %d %d %d\n", myid_world, is, id, p, nrecv[p]);
              // fflush(stdout);
              recvvec[p].resize(nrecv[p]);
              ierr |= MPI_Irecv(&recvvec[p][0], nrecv[p], datatype<T>(), p, 0, comm_world, &req[nreq++]);
            }
          }
          ierr |= MPI_Waitall(nreq, req, stat);
          ierr |= MPI_Barrier(comm_world);
        } // for(id)
      } // for(is)
      return 0;
    }

#ifdef GANGED_ALLTOALL
//  typedef boost::multi_array<int, 2> int2D;
  using int2D = std::vector<std::vector<int>>;
  template <typename T>
    void gather_vector(std::vector<T> dst[], std::vector<T> src[]){
      // NOT YET OPTIMIEZED
      static int count[MAX_LOCAL];
      static int off[MAX_LOCAL+1] = {0, };
      for(int p=0; p<size_world; p++){
        int ssize = src[p].size();
        MPI_Gather(&ssize, 1, MPI_INT, count, 1, MPI_INT, 0, comm_local);
        for(int pp=0; pp<size_local; pp++){
          off[pp+1] = off[pp] + count[pp];
        }
        if(myid_local == 0){
          dst[p].resize(off[size_local]);
        }
        MPI_Gatherv(&src[p][0], ssize, datatype<T>(), 
            &dst[p][0], count, off, datatype<T>(), 0, comm_local);
      }
    }

  template <typename T>
    void connect_vector(std::vector<T> dst[], std::vector<T> src[]){
      for(int ii=0, iii=0; ii<size_global; ii++){
        dst[ii].clear();
        for(int i=0; i<size_local; i++, iii++){
          dst[ii].insert(dst[ii].end(), src[iii].begin(), src[iii].end());
        }
      }
    }

  template <typename T>
    void alltoall_safe(std::vector<T> rbuf[], std::vector<T> sbuf[]){
#if 1
      for(int dist=1; dist<size_global; dist++){
        int src = (size_global + myid_global - dist) % size_global;
        int dst = (size_global + myid_global + dist) % size_global;
        int scount = sbuf[dst].size();
        int rcount;
        MPI_Status stat;
        MPI_Sendrecv(&scount, 1, MPI_INT, dst, 0,
            &rcount, 1, MPI_INT, src, 0, comm_global, &stat);
        rbuf[src].resize(rcount);
        MPI_Sendrecv(&sbuf[dst][0], scount, datatype<T>(), dst, 1,
            &rbuf[src][0], rcount, datatype<T>(), src, 1, comm_global, &stat);
      }
#else // yet another implementation for 2^n processors
      assert(__builtin_popcount(size_global) == 1);
      for(int dist=1; dist<size_global; dist++){
        int dst = myid_global ^ dist;
        int scount = sbuf[dst].size();
        int rcount;
        MPI_Status stat;
        MPI_Sendrecv(&scount, 1, MPI_INT, dst, 0,
            &rcount, 1, MPI_INT, dst, 0, comm_global, &stat);
        rbuf[dst].resize(rcount);
        MPI_Sendrecv(&sbuf[dst][0], scount, datatype<T>(), dst, 1,
            &rbuf[dst][0], rcount, datatype<T>(), dst, 1, comm_global, &stat);
      }
#endif
      rbuf[myid_global] = sbuf[myid_global]; // just copy it
    }

  template <typename T>
    void alltoall_danger(std::vector<T> rbuf[], std::vector<T> sbuf[]){
      static MPI_Status stat[MAX_GLOBAL];
      static MPI_Request req[MAX_GLOBAL*2];
      static int nrecv[MAX_GLOBAL];

      int nreq = 0;
      for(int p=0; p<size_global; p++){
        int nsend = sbuf[p].size();
        MPI_Isend(&nsend, 1, MPI_INT, p, 0, comm_global, &req[nreq++]);
      }
      for(int p=0; p<size_global; p++){
        MPI_Irecv(&nrecv[p], 1, MPI_INT, p, 0, comm_global, &req[nreq++]);
      }
      MPI_Waitall(nreq, req, stat);

      nreq = 0;
      for(int p=0; p<size_global; p++){
        int nsend = sbuf[p].size();
        MPI_Isend(&sbuf[p][0], nsend, datatype<T>(), p, 0, comm_global, &req[nreq++]);
      }
      for(int p=0; p<size_global; p++){
        rbuf[p].resize(nrecv[p]);
        MPI_Irecv(&rbuf[p][0], nrecv[p], datatype<T>(), p, 0, comm_global, &req[nreq++]);
      }
      MPI_Waitall(nreq, req, stat);
    }

  template <typename T>
    void scatter_vector(std::vector<T> dst[], std::vector<T> src[], int2D &counts){
      // NOT YET OPTIMIEZED
      for(int ii=0; ii<size_global; ii++){
        int *count = &counts[ii][0];
        static int off[MAX_LOCAL+1] = {0, }; 
        for(int pp=0; pp<size_local; pp++){
          off[pp+1] = off[pp] + count[pp];
        }
        int rcount = count[myid_local];
        dst[ii].resize(rcount);
        MPI_Scatterv(&src[ii][0], count, off, datatype<T>(),
            &dst[ii][0], rcount, datatype<T>(), 0, comm_local);
      }
    }

  template <typename T>
    int delegate_alltoallv(
        std::vector<T> sendvec[],
        std::vector<T> recvvec[]){
      if(firstcall) initialize();

      // phase-1 (local comm)
      static std::vector<T> sendbuf1[MAX_WORLD];
      {
//        RAII_Timer timer("phase-1 : ", NULL, std::cout, isroot);
        gather_vector(sendbuf1, sendvec);
      }

      // phase-2 (global comm)
#if 0
      int2D scount(boost::extents[size_global][size_local]);
      int2D rcount(boost::extents[size_global][size_local]);
#else
      int2D scount,rcount;
      scount.resize(size_global);
      rcount.resize(size_global);
      for (int i = 0; i < size_global; i++)
      {
        scount[i].resize(size_local);
        rcount[i].resize(size_local);
      }
#endif

      {
//        RAII_Timer timer("phase-2 : ", NULL, std::cout, isroot);
        if(myid_local == 0){
          for(int ii=0, iii=0; ii<size_global; ii++){
            for(int i=0; i<size_local; i++, iii++){
              scount[ii][i] = sendbuf1[iii].size();
            }
          }
          MPI_Alltoall(&scount[0][0], size_local, MPI_INT,
              &rcount[0][0], size_local, MPI_INT, comm_global);
        }
        // local comm
        MPI_Bcast(&rcount[0][0], size_world, MPI_INT, 0, comm_local);
      }

      // phase-3 (no comm)
      static std::vector<T> sendbuf2[MAX_GLOBAL];
      {
//        RAII_Timer timer("phase-3 : ", NULL, std::cout, isroot);
        connect_vector(sendbuf2, sendbuf1);
      }

      // phase-4 (global comm)
      static std::vector<T> recvbuf[MAX_GLOBAL];
      {
//        RAII_Timer timer("phase-4 : ", NULL, std::cout, isroot);
        if(myid_local == 0){
#if 1
          if(isroot) std::cout << "phase-4 : safe version" << std::endl;
          alltoall_safe(recvbuf, sendbuf2);
#else
          if(isroot) std::cout << "phase-4 : danger version" << std::endl;
          alltoall_danger(recvbuf, sendbuf2);
#endif
          MPI_Barrier(comm_global);
        }
        MPI_Barrier(comm_local);
      }

      // phase-5 (local comm)
      {
//        RAII_Timer timer("phase-5 : ", NULL, std::cout, isroot);
        for(int i=0; i<size_global; i++){
          recvvec[i].clear();
        }
        scatter_vector(recvvec, recvbuf, rcount);
      }
      MPI_Barrier(comm_world);
      return 0;
    }
#endif

  int local_size(){
    return size_local;
  }
  int local_id(){
    return myid_local;
  }
  void sync_local(){
    MPI_Barrier(comm_local);
  }
};
