#pragma once


#include <mpi.h>
#include <iostream>
#include "delegate_alltoall.h"

struct MP
{

  int rank, nrank;
  MPI_Comm comm;
  MYMPI mympi;


  struct float4
  {
    typedef float  v4sf __attribute__ ((vector_size(16)));
    typedef double v2df __attribute__ ((vector_size(16)));
    static v4sf v4sf_abs(v4sf x){
      typedef int v4si __attribute__ ((vector_size(16)));
      v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
      return __builtin_ia32_andps(x, (v4sf)mask);
    }
    union{
      v4sf v;
      struct{
        float x, y, z, w;
      };
    };
    float4() : v((v4sf){0.f, 0.f, 0.f, 0.f}) {}
    float4(float x, float y, float z, float w) : v((v4sf){x, y, z, w}) {}
    float4(float x) : v((v4sf){x, x, x, x}) {}
    float4(v4sf _v) : v(_v) {}
    float4 abs(){
      typedef int v4si __attribute__ ((vector_size(16)));
      v4si mask = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
      return float4(__builtin_ia32_andps(v, (v4sf)mask));
    }
    void dump(){
      std::cerr << x << " "
        << y << " "
        << z << " "
        << w << std::endl;
    }
#if 1
    v4sf operator=(const float4 &rhs){
      v = rhs.v;
      return v;
    }
    float4(const float4 &rhs){
      v = rhs.v;
    }
#endif
#if 0
    const v4sf_stream operator=(const v4sf_stream &s){
      __builtin_ia32_movntps((float *)&v, s.v);
      return s;
    }
    float4(const v4sf_stream &s){
      __builtin_ia32_movntps((float *)&v, s.v);
    }
#endif
  };


  using vector3 = std::array<double,3>;
  int MP_myprocid() const
  {
    return rank;
  }
  int MP_proccount() const
  {
    return nrank;
  }
  bool MP_root() {return (0 == MP_myprocid()); }
  void MP_Abort(int err)
  {
    MPI_Abort(MPI_COMM_WORLD, err);
  }
  void MP_int_sum(int& i)
  {
    int tmp;
    MPI_Reduce(&i,&tmp,1, MPI_INT, MPI_SUM,0,MPI_COMM_WORLD);
    if(rank == 0) i = tmp;
  }
  void MP_sum(double& r)
  {
    double tmp;
    MPI_Reduce(&r,&tmp,1, MPI_DOUBLE, MPI_SUM,0,MPI_COMM_WORLD);
    if(rank == 0) r = tmp;
  }
  void MP_double_sum(double r[], int count, bool allreduce = false)
  {
    static double _buf[1024];
    double *buf = count > 1024 ? new double[count] : _buf;
    if(allreduce){
      MPI_Allreduce(r, buf, count, MPI_DOUBLE, MPI_SUM,  MPI_COMM_WORLD);
    }else{
      MPI_Reduce(r, buf, count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if(allreduce || MP_root()){
      for(int i=0; i<count; i++){
        r[i] = buf[i];
      }
    }
    if(buf != _buf) free(buf);
  }

  void MP_int_bcast(int& i)
  {
    MPI_Bcast(&i,1,MPI_INT,0,MPI_COMM_WORLD);
  }
  void MP_double_bcast(double* data, int nwords)
  {
    MPI_Bcast(data,nwords,MPI_DOUBLE,0,MPI_COMM_WORLD);
  }

  void MP_gather_sample_coords(std::vector<vector3> &sample_array)
  {
    int nsample = sample_array.size();
    using real = double;
    const int local_proc_id = rank;
    const int total_proc_count = nrank;
    MPI_Status status;
#if 1 // original version
    if(local_proc_id != 0){
#ifndef TCPLIB	
      // send samples and return
      MPI_Send( &nsample, 1, MPI_INT, 0,local_proc_id*2 , MPI_COMM_WORLD);
      MPI_Send( (real*)&sample_array[0], nsample*3, MPI_DOUBLE, 0,local_proc_id*2+1,
          MPI_COMM_WORLD);
#else
      tcp_transfer_data_by_MPIname(0,TCPLIB_SEND,sizeof(int),&nsample);
      tcp_transfer_data_by_MPIname(0,TCPLIB_SEND,sizeof(real)*nsample*3,
          (void*)sample_array);
#endif
    }else{
      for(int i=1;i<total_proc_count; i++){
        int nreceive;
#ifndef TCPLIB	    
        MPI_Recv( &nreceive, 1, MPI_INT, i,i*2, MPI_COMM_WORLD,&status);
        MPI_Recv((real*)(&sample_array[nsample]), 3*nreceive, MPI_DOUBLE,
            i,i*2+1, MPI_COMM_WORLD,&status);
#else
        tcp_transfer_data_by_MPIname(i,TCPLIB_RECV,sizeof(int),&nreceive);
        tcp_transfer_data_by_MPIname(i,TCPLIB_RECV,sizeof(real)*nreceive*3,
            (void*)(sample_array+nsample));
#endif	    
        nsample+=nreceive;
      }
    }
#else  // using MPI_Gathrev
    int np = MP_proccount();
    static int count[NMAXPROC], displs[NMAXPROC+1];
    MPI_Gather(&nsample, 1, MPI_INT, count, 1, MPI_INT, 0, MPI_COMM_WORLD);
    displs[0] = 0;
    for(int i=0; i<np; i++){
      displs[i+1] = displs[i] + count[i];
    }
    assert(displs[np] <= NMAXSAMPLE);
    for(int i=0; i<np; i++){
      count[i] *= 3;
      displs[i+1] *= 3;
    }
    MPI_Gatherv(sample_array, nsample, MPI_DOUBLE, sample_array, count, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
#endif
  }

  template <typename T>
    void MP_alltoallv(std::vector<T> sendbuf[], std::vector<T> recvbuf[])
    {
#if 0
      mympi::detuned_alltoallv(sendbuf, recvbuf, 4);
#else
      mympi.delegate_alltoallv(sendbuf, recvbuf);
#endif
    }

};

