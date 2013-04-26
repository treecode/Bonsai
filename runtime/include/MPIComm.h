#pragma once

#include "mpi.h"
#include <vector>
#include <sys/time.h>

template<class T>
MPI_Datatype MPIComm_datatype();


#if 0 /*data type specification, must be in *.cpp file */

/* default data-types */

template <> MPI_Datatype MPIComm_datatype<char  >() {return MPI_BYTE;   }
template <> MPI_Datatype MPIComm_datatype<int   >() {return MPI_INT;    }
template <> MPI_Datatype MPIComm_datatype<float >() {return MPI_FLOAT;  }
template <> MPI_Datatype MPIComm_datatype<double>() {return MPI_DOUBLE; }

static MPI_Datatype MPI_MDATA = 0;

  template <> 
MPI_Datatype MPIComm_datatype<MYDATA>() 
{
  if (MPI_MYDATA) return MPI_MYDATA;
  else {
    int ss = sizeof(MYDATA) / sizeof(float);
    assert(0 == sizeof(MYDATA) % sizeof(float));
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_MYDATA);
    MPI_Type_commit(&MPI_MYDATA);
    return MPI_MYDATA;
  }
}
#endif

#if 0
#define MPICOMMDEBUG
#endif

#if 0
#define USE_ALL2ALLV
#endif

void MPIComm_free_type();
struct MPIComm
{

  MPI_Comm MPI_COMM_I;
  MPI_Comm MPI_COMM_J;

  int myid, n_proc;

  int n_proc_i;
  int n_proc_j;

  int i_color;
  int j_color;

  double get_time() {
  	struct timeval Tvalue;
	struct timezone dummy;
  
	gettimeofday(&Tvalue,&dummy);
	  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
	}

  MPIComm(const int _myid, const int _nproc) : myid(_myid), n_proc(_nproc)
  {
    //// ij-parallized ////

    int n_tmp = (int)sqrt(n_proc+0.1);
    while(n_proc%n_tmp){
      n_tmp--;
    }

    n_proc_j = n_tmp;  // number of pe in i-comm
    n_proc_i = n_proc/n_proc_j; // number of pe in j-comm
    j_color = myid%n_proc_i; // = rank in i-comm
    i_color = myid/n_proc_i; // = rank in j-comm

    if (myid == 0)
    {
      fprintf(stderr, " MPIComm :: ni= %d  nj= %d  :: ni x nj= %d  nproc= %d\n",
          n_proc_i, n_proc_j, n_proc_i * n_proc_j, n_proc);
    }

    MPI_Comm_split(MPI_COMM_WORLD, i_color, myid, &MPI_COMM_I);
    MPI_Comm_size(MPI_COMM_I, &n_proc_i);

    MPI_Comm_split(MPI_COMM_WORLD, j_color, myid, &MPI_COMM_J);
    MPI_Comm_size(MPI_COMM_J, &n_proc_j);
  } 

  ~MPIComm()
  {
    MPI_Comm_free(&MPI_COMM_I);
    MPI_Comm_free(&MPI_COMM_J);
    MPIComm_free_type();
  }

  template<typename T>
    void all2all(
        const int nproc, const int color,
	T sbuf[], const int sendcnts[], const int sdispl[],
	T rbuf[],const  int recvcnts[], const int rdispl[],
	const MPI_Comm comm)
    {
	for (int dist = 0; dist < nproc; dist++) {
		const int src = (nproc + color - dist) % nproc;
		const int dst = (nproc + color + dist) % nproc;
		const int scount = sendcnts[dst];
		const int rcount = recvcnts[src];
		MPI_Status stat;
	MPI_Sendrecv(&sbuf[sdispl[dst]],
	      scount, MPIComm_datatype<T>(), dst, 1,
		&rbuf[rdispl[src]], rcount, MPIComm_datatype<T>(), src, 1, comm, &stat);
	}
    }

  template<typename T>
    void sort_array(std::vector<T> &p, int sub_counts[], int scounts[])
    {
#if 0
      if(myid==0) cout << myid << "sort array" << endl;
#endif  
      int np = p.size();
      std::vector<T> p_new(np);
      std::vector<int> sub_disp(n_proc+1);
      sub_disp[0] = 0;
      for(int i=0;i<n_proc;i++)
        sub_disp[i+1] = sub_disp[i]+sub_counts[i];

      int ncount = 0;
      for(int i=0;i<n_proc_i;i++)
      {
        scounts[i] = 0;
        for(int j=0;j<n_proc_j;j++)
        {
          int kstart = sub_disp[j*n_proc_i+i];
          for(int k=0;k<sub_counts[j*n_proc_i+i];k++)
          {
            p_new[ncount] = p[kstart+k];
            ncount++;
            scounts[i]++;   
          }
        }
      }
      p = p_new;
    }

  template<typename T>
    void all2allv_2D_main(std::vector<T> &p, int scounts[], int sdispls[])
    {
	//MPI_Barrier(MPI_COMM_WORLD);
	    double t0 = get_time();
      std::vector<int> sub_counts(n_proc);
      //// exchange n of particles within j-comm -> sub counts ////
      MPI_Alltoall(&scounts[0], n_proc_i, MPI_INT,
          &sub_counts[0], n_proc_i, MPI_INT, MPI_COMM_J);    
	double t1 = get_time();
      std::vector<int> scounts_j(n_proc_j);
      std::vector<int> rcounts_j(n_proc_j);
      std::vector<int> sdispls_j(n_proc_j+1);
      std::vector<int> rdispls_j(n_proc_j+1);

      //// from sub-counts, generates scounts for j-comm ////
      rdispls_j[0] = 0;
      sdispls_j[0] = 0;
      for(int i=0;i<n_proc_j;i++)
      {
        rcounts_j[i] = 0;
        scounts_j[i] =  sdispls[(i+1)*n_proc_i] - sdispls[i*n_proc_i];

        for(int k=0;k<n_proc_i;k++)
          rcounts_j[i] += sub_counts[n_proc_i*i+k];

        rdispls_j[i+1] = rdispls_j[i] + rcounts_j[i];
        sdispls_j[i+1] = sdispls_j[i] + scounts_j[i];
      }
      //// alltoallv within j-comm ////
      std::vector<T> p_new(rdispls_j[n_proc_j]); 
	double t2 = get_time();
#ifdef USE_ALL2ALLV
      MPI_Alltoallv(&p[0], &scounts_j[0], &sdispls_j[0], MPIComm_datatype<T>(), &p_new[0], &rcounts_j[0], &rdispls_j[0], MPIComm_datatype<T>(), MPI_COMM_J);
#else
      all2all<T>(n_proc_j,i_color,&p[0], &scounts_j[0], &sdispls_j[0],  &p_new[0], &rcounts_j[0], &rdispls_j[0],  MPI_COMM_J);
#endif
#if 0
      if(myid==0) cout << i_color << " " << j_color << "alltoallv comm-j np=" << p.size()<< endl;
#endif

	double t3= get_time();
      std::vector<int> scounts_i(n_proc_i);
      std::vector<int> rcounts_i(n_proc_i);
      std::vector<int> sdispls_i(n_proc_i+1);
      std::vector<int> rdispls_i(n_proc_i+1);


      //// swap particles for communication in i-comm //// 
      sort_array<T>(p_new, &sub_counts[0], &scounts_i[0]);
	double t4 = get_time();
      //// Alltoall in i-comm ////
      MPI_Alltoall(&scounts_i[0], 1,  MPI_INT, &rcounts_i[0], 1, MPI_INT, MPI_COMM_I);
	double t5 = get_time();
#if 0
      if(myid==0) cout << "alltoall in comm-i" << endl;
#endif

      rdispls_i[0] = 0;
      sdispls_i[0] = 0;
      for(int i=0;i<n_proc_i;i++)
      {
        rdispls_i[i+1] = rdispls_i[i] + rcounts_i[i];
        sdispls_i[i+1] = sdispls_i[i] + scounts_i[i];
      }

      //// Alltoallv within i-comm ////
#if 0
      std::vector<T> p_new2(rdispls_i[n_proc_i]);
#endif
      p.resize(rdispls_i[n_proc_i]);
double t6 = get_time();
#ifdef USE_ALL2ALLV
      MPI_Alltoallv(&p_new[0], &scounts_i[0], &sdispls_i[0], MPIComm_datatype<T>(), 
          &p[0], &rcounts_i[0], &rdispls_i[0], MPIComm_datatype<T>(), MPI_COMM_I);
#else
      all2all<T>(n_proc_i,j_color,&p_new[0], &scounts_i[0], &sdispls_i[0],
          &p[0], &rcounts_i[0], &rdispls_i[0], MPI_COMM_I);
#endif
#if 0
      if(myid==0) cout << "alltoallv in comm-i" << endl;
      p.swap(p_new2);
#endif
      double t7 = get_time();
      if(myid == 0)
	fprintf(stderr, "Proc: %d  a2a: %lg offsets: %lg a2av: %lg sort: %lg  a2a: %lg res: %lg a2av: %lg Total: %lg \n",
		myid, t1-t0, t2-t1,t3-t2, t4-t3, t5-t4, t6-t5, t7-t6, t7-t0);	 
    }

  template<typename T>
    void all2allv_2D(std::vector<T> &p, int scounts[])
    {
      std::vector<int> sdispls(n_proc+1);
      sdispls[0] = 0;
      for(int i=0;i<n_proc;i++)
        sdispls[i+1] = sdispls[i] + scounts[i];

      //MPI_Barrier(MPI_COMM_WORLD); //// for test 
      //
      all2allv_2D_main(p, scounts, &sdispls[0]);
    }

  template<typename T>
    void all2allv_1D(std::vector<T> &p, int scounts[])
    {
      std::vector<int> sdispls(n_proc+1);
      std::vector<int> rcounts(n_proc);
      std::vector<int> rdispls(n_proc+1);

      MPI_Alltoall(scounts, 1, MPI_INT, 
          &rcounts[0], 1, MPI_INT, MPI_COMM_WORLD);
      rdispls[0] = 0;
      sdispls[0] = 0;
      for(int i=0;i<n_proc;i++)
      {
        sdispls[i+1] = sdispls[i] + scounts[i];
        rdispls[i+1] = rdispls[i] + rcounts[i];
        //cout << myid << " " << i << "  " << scounts[i] << endl;
      }
      //MPI_Barrier(MPI_COMM_WORLD); //// for test 
      std::vector<T> p_new(rdispls[n_proc]);
#ifdef USE_ALL2ALLV
      MPI_Alltoallv(&p[0], scounts, &sdispls[0], MPIComm_datatype<T>(), &p_new[0], &rcounts[0], &rdispls[0], MPIComm_datatype<T>(), MPI_COMM_WORLD);
#else
      all2all<T>(n_proc,myid,&p[0], scounts, &sdispls[0],  &p_new[0], &rcounts[0], &rdispls[0], MPI_COMM_WORLD);
#endif
      p.swap(p_new);
    }

  void ugly_all2allv_char(const float *sendbuf, const int *sendcnts, float *recvbuf)
  {
double t0 =get_time();
    std::vector<int> scounts(n_proc);
    int nsend= 0;
    for (int i = 0; i < n_proc; i++)
    {
      assert(sendcnts[i] % sizeof(float) == 0);
      scounts[i] = sendcnts[i] / sizeof(float);
      nsend += scounts[i];
    }
double t1 = get_time();	
    std::vector<float> sendrecv_buf(nsend);
    for (int i = 0; i < nsend; i++)
      sendrecv_buf[i] = sendbuf[i];

double t2 = get_time();	
#if 1
    all2allv_2D(sendrecv_buf, &scounts[0]);
#else
    all2allv_1D(sendrecv_buf, &scounts[0]);
#endif

double t3 = get_time();	
    const int nrecv = sendrecv_buf.size();
    for (int i = 0; i < nrecv; i++)
      recvbuf[i] = sendrecv_buf[i];
double t4 = get_time();	

fprintf(stderr,"Proc: %d a2a prep:  t1 : %lg  t2: %lg t3: %lg t4: %lg total: %lg Items: %d \n" ,
myid, t1-t0, t2-t1, t3-t2, t4-t3, t4-t0, nsend);


  }

};
