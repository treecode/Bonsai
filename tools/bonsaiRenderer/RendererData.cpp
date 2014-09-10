#include "RendererData.h"
    
    
void RendererDataDistribute::create_division()
{ 
  int &nx = npx;
  int &ny = npy;
  int &nz = npz;
  ////////
  const int n = nrank;
  int n0, n1; 
  n0 = (int)pow(n+0.1,0.33333333333333333333);
  while(n%n0)n0--;
  if (isMaster())
    fprintf(stderr, "n= %d  n0= %d \n", n, n0);
  nx = n0;
  n1 = n/nx;
  n0 = (int)sqrt(n1+0.1);
  while(n1%n0)n0++;
  if(isMaster()){
    fprintf(stderr, "n1= %d  n0= %d \n", n1, n0);
  }
  ny = n0; nz = n1/n0;
  // int ntmp;
  if (nz > ny){
    // ntmp = nz; nz = ny; ny = ntmp;
    std::swap(ny, nz);
  }
  if (ny > nx){
    // ntmp = nx; nx = ny; ny = ntmp;
    std::swap(nx, ny);
  }
  if (nz > ny){
    // ntmp = nz; nz = ny; ny = ntmp;
    std::swap(ny, nz);
  }
  if (nx*ny*nz != n){
    std::cerr << "create_division: Intenal Error " << n << " " << nx
      << " " << ny << " " << nz <<std::endl;
    MPI_Abort(comm, 1);
  }
  if(isMaster()) {
    fprintf(stderr, "[nx, ny, nz] = %d %d %d\n", nx, ny, nz);
  }
}

int RendererDataDistribute::determine_sample_freq()
{
  const int _n = data.size();
  const int nbody = _n;
#if 0
  int nreal = nbody;
  MPI_int_sum(nreal);
  int maxsample = (int)(NMAXSAMPLE*0.8); // 0.8 is safety factor
  int sample_freq = (nreal+maxsample-1)/maxsample;
#else
  double nglb = nbody, nloc = nbody;
  MPI_Allreduce(&nloc, &nglb, 1, MPI_DOUBLE, MPI_SUM, comm);
  // double maxsample = (NMAXSAMPLE*0.8); // 0.8 is safety factor
  double maxsample = (NMAXSAMPLE*0.8); // 0.8 is safety factor
  int sample_freq = int((nglb+maxsample)/maxsample);
#endif
  MPI_Bcast(&sample_freq,1,MPI_INT,getMaster(),comm);
  return sample_freq;
}
    
void RendererDataDistribute::initialize_division()
{
  static bool initcall = true;
  if(initcall)
  {
    sample_freq = determine_sample_freq();
    create_division();
    initcall = false;
  }
}

void RendererDataDistribute::collect_sample_particles(std::vector<vector3> &sample_array, const int sample_freq)
{
  const int _n = data.size();
  const int nbody = _n;
  sample_array.clear();
  for(int i=0,  ii=0; ii<nbody; i++, ii+=sample_freq)
    sample_array.push_back(vector3{{posx(i), posy(i), posz(i)}});

  /* gather sample coords */
  int nsample = sample_array.size();
  if (!isMaster())
  {
    MPI_Send(&nsample,         1,         MPI_INT,    getMaster(), rank*2,   comm);
    MPI_Send(&sample_array[0], 3*nsample, MPI_DOUBLE, getMaster(), rank*2+1, comm);
  }
  else
  {
    MPI_Status status;
    for (int p = 0; p < nrank; p++)
      if (p != getMaster())
      {
        int nrecv;
        MPI_Recv(&nrecv, 1, MPI_INT, p, p*2, comm, &status);
        sample_array.resize(nsample+nrecv);
        MPI_Recv(&sample_array[nsample], 3*nrecv, MPI_DOUBLE, p, p*2+1, comm, &status);
        nsample += nrecv;
      }

  }
}

void RendererDataDistribute::determine_division( // nitadori's version
    std::vector<float4>  &pos,
    const float rmax,
    vector3  xlow[],  // left-bottom coordinate of divisions
    vector3 xhigh[])  // size of divisions
{
  const int nx = npx;
  const int ny = npy;
  const int nz = npz;
  const int np = pos.size();

  struct Address{
    const int nx, ny, nz;
    std::vector<int> offset;

    Address(int _nx, int _ny, int _nz, int np) :
      nx(_nx), ny(_ny), nz(_nz), offset(1 + nx*ny*nz)
    {
      const int n = nx*ny*nz;
      for(int i=0; i<=n; i++){
        offset[i] = (i*np)/n;
      }
    }
    int idx(int ix, int iy, int iz){
      return ix + nx*(iy + ny*(iz));
    }
    int xdi(int ix, int iy, int iz){
      return iz + nz*(iy + ny*(ix));
    }
    int off(int ix, int iy, int iz){
      return offset[xdi(ix, iy, iz)];
    }
  };

  const int n = nx*ny*nz;
  assert(n <= NMAXPROC);

  Address addr(nx, ny, nz, np);


  double buf[NMAXPROC+1];
  // divide on x
  {
    double *xoff = buf; // xoff[nx+1]
    __gnu_parallel::sort(&pos[addr.off(0, 0, 0)], &pos[addr.off(nx, 0, 0)], 
        [](const float4 &rhs, const float4 &lhs)  {
          const int mask = 1;
          return mask & __builtin_ia32_movmskps(
            (float4::v4sf)__builtin_ia32_cmpltps(lhs.v, rhs.v));
        });
    for(int ix=0; ix<nx; ix++)
    {
      const int ioff = addr.off(ix, 0, 0);
      xoff[ix] = 0.5 * (pos[ioff].x + pos[1+ioff].x);
      // PRC(xoff[ix]);
    }
    // cerr << endl;
    xoff[0]  = -rmax;
    xoff[nx] = +rmax;
    for(int ix=0; ix<nx; ix++)
      for(int iy=0; iy<ny; iy++)
        for(int iz=0; iz<nz; iz++)
        {
          const int ii = addr.xdi(ix, iy, iz);
          // PRC(ix); PRC(iy); PRC(iz); PRL(ii);
          xlow [ii][0] = xoff[ix];
          xhigh[ii][0] = xoff[ix+1];
        }
  }

  // divide on y
  {
    double *yoff = buf; // yoff[ny+1];
    for(int ix=0; ix<nx; ix++)
    {
      __gnu_parallel::sort(&pos[addr.off(ix, 0, 0)], &pos[addr.off(ix, ny, 0)], 
        [](const float4 &rhs, const float4 &lhs)  {
          const int mask = 2;
          return mask & __builtin_ia32_movmskps(
            (float4::v4sf)__builtin_ia32_cmpltps(lhs.v, rhs.v));
        });
      for(int iy=0; iy<ny; iy++)
      {
        const int ioff = addr.off(ix, iy, 0);
        yoff[iy] = 0.5 * (pos[ioff].y + pos[1+ioff].y);
        // PRC(yoff[iy]);
      }
      // cerr << endl;
      yoff[0]  = -rmax;
      yoff[ny] = +rmax;
      for(int iy=0; iy<ny; iy++)
        for(int iz=0; iz<nz; iz++)
        {
          const int ii = addr.xdi(ix, iy, iz);
          xlow [ii][1] = yoff[iy];
          xhigh[ii][1] = yoff[iy+1];
        }
    }
  }
  // divide on z
  {
    double *zoff = buf; // zoff[nz+1];
#pragma omp parallel for schedule(guided) collapse(2)
    for(int ix=0; ix<nx; ix++)
      for(int iy=0; iy<ny; iy++)
      {
        std::sort(&pos[addr.off(ix, iy, 0)], &pos[addr.off(ix, iy, nz)], 
        [](const float4 &rhs, const float4 &lhs)  {
          const int mask = 4;
          return mask & __builtin_ia32_movmskps(
            (float4::v4sf)__builtin_ia32_cmpltps(lhs.v, rhs.v));
        });
        for(int iz=0; iz<nz; iz++)
        {
          const int ioff = addr.off(ix, iy, iz);
          zoff[iz] = 0.5 * (pos[ioff].z + pos[1+ioff].z);
        }
        // cerr << endl;
        zoff[0]  = -rmax;
        zoff[nz] = +rmax;
        for(int iz=0; iz<nz; iz++){
          const int ii = addr.xdi(ix, iy, iz);
          xlow [ii][2] = zoff[iz];
          xhigh[ii][2] = zoff[iz+1];
        }
      }
  }
}
    
inline int RendererDataDistribute::which_box(
        const vector3 &pos,
        const vector3 xlow[],
        const vector3 xhigh[])
    {
      int p = 0;
      if(pos[0] < xlow[p][0]) return -1;
      for(int ix=0; ix<npx; ix++, p+=npy*npz){
        if(pos[0] < xhigh[p][0]) break;
      }
      if(pos[0] > xhigh[p][0]) return -1;

      if(pos[1] < xlow[p][1]) return -1;
      for(int iy=0; iy<npy; iy++, p+=npz){
        if(pos[1] < xhigh[p][1]) break;
      }
      if(pos[1] > xhigh[p][1]) return -1;

      if(pos[2] < xlow[p][2]) return -1;
      for(int iy=0; iy<npy; iy++, p++){
        if(pos[2] < xhigh[p][2]) break;
      }
      if(pos[2] > xhigh[p][2]) return -1;

      return p;
    }

inline void RendererDataDistribute::which_boxes(
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> &boxes)
{
  for (int p = 0; p < nrank; p++)
  {
    if (
        pos[0]+h >= xlow[p][0]  && pos[0]-h <= xhigh[p][0] &&
        pos[1]+h >= xlow[p][1]  && pos[1]-h <= xhigh[p][1] &&
        pos[2]+h >= xlow[p][2]  && pos[2]-h <= xhigh[p][2])
      boxes.push_back(p);
  }
}
    
void RendererDataDistribute::alltoallv(std::vector<particle_t> psend[], std::vector<particle_t> precv[])
{
  static MPI_Datatype MPI_PARTICLE = 0;
  if (!MPI_PARTICLE)
  {
    int ss = sizeof(particle_t) / sizeof(float);
    assert(0 == sizeof(particle_t) % sizeof(float));
    MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_PARTICLE);
    MPI_Type_commit(&MPI_PARTICLE);
  }

  static std::vector<int> nsend(nrank), senddispl(nrank+1,0);
  int nsendtot = 0;
  for (int p = 0; p < nrank; p++)
  {
    nsend[p] = psend[p].size();
    senddispl[p+1] = senddispl[p] + nsend[p];
    nsendtot += nsend[p];
  }

  static std::vector<int> nrecv(nrank), recvdispl(nrank+1,0);
  MPI_Alltoall(&nsend[0], 1, MPI_INT, &nrecv[0], 1, MPI_INT, comm);

  int nrecvtot = 0;
  for (int p = 0; p < nrank; p++)
  {
    recvdispl[p+1] = recvdispl[p] + nrecv[p];
    nrecvtot += nrecv[p];
  }


  static std::vector<particle_t> sendbuf, recvbuf;
  sendbuf.resize(nsendtot); 
  recvbuf.resize(nrecvtot);

  int iloc = 0;
  for (int p = 0; p < nrank; p++)
    for (int i = 0; i < nsend[p]; i++)
      sendbuf[iloc++] = psend[p][i];

  assert(senddispl[nrank] == nsendtot);
  assert(recvdispl[nrank] == nrecvtot);

  MPI_Alltoallv(
      &sendbuf[0], &nsend[0], &senddispl[0], MPI_PARTICLE,
      &recvbuf[0], &nrecv[0], &recvdispl[0], MPI_PARTICLE, 
      comm);

  for (int p = 0; p < nrank; p++)
  {
    precv[p].resize(nrecv[p]);
    for (int i = 0; i < nrecv[p]; i++)
      precv[p][i] = recvbuf[recvdispl[p] + i];
  }
}


void RendererDataDistribute::exchange_particles_alltoall_vector(
    const vector3  xlow[],
    const vector3 xhigh[])
{
  int myid = rank;
  int nprocs = nrank;

  static std::vector<particle_t> psend[NMAXPROC];
  static std::vector<particle_t> precv[NMAXPROC];
  const int nbody = data.size();

  static bool initcall = true;
  if(initcall)
  {
    initcall = false;
    for(int p=0; p<nprocs; p++)
    {
      psend[p].reserve(64);
      precv[p].reserve(64);
    }
  }

  int iloc = 0;
  Boundary boundary(xlow[myid], xhigh[myid]);
  for(int i=0; i<nbody; i++)
    if(boundary.isinbox(vector3{{data[i].posx, data[i].posy, data[i].posz}}))
      std::swap(data[i],data[iloc++]);

  for(int p=0; p<nprocs; p++)
  {
    psend[p].clear();
    precv[p].clear();
  }

#if 0
  for(int i=0; i<nbody; i++)
  {
    int ibox = which_box(vector3{{data[i].posx,data[i].posy,data[i].posz}}, xlow, xhigh);
    if (nbody < iloc)
      assert(ibox == rank);
    if(ibox < 0)
    {
      std::cerr << myid <<" exchange_particle error: particle in no box..." << std::endl;
      vector3 fpos{{data[i].posx,data[i].posy,data[i].posz}};
      // cerr << pb[i].get_pos() << endl;
      std::cout // << boost::format("[%f %f %f], [%lx %lx %lx]")
        << fpos[0] << " " << fpos[1] << " " << fpos[2] << std::endl;
      //        pb[i].dump();

      for (int p = 0; p < nrank; p++)
      {
        fprintf(stderr," rank= %d: xlow= %g %g %g  xhigh= %g %g %g \n", p,
            xlow[p][0],  xlow[p][1],  xlow[p][2],
            xhigh[p][0], xhigh[p][1], xhigh[p][2]);
      }
      MPI_Abort(comm,1);
    }
    else
    {
      psend[ibox].push_back(data[i]);
    }
  }
#else
  std::vector<int> boxes;
  boxes.reserve(nrank);
  for(int i=0; i<nbody; i++)
  {
    boxes.clear();
    which_boxes(vector3{{data[i].posx,data[i].posy,data[i].posz}}, 1.1*data[i].attribute[Attribute_t::H], xlow, xhigh, boxes);
    //        which_boxes(vector3{{data[i].posx,data[i].posy,data[i].posz}}, 0, xlow, xhigh, boxes);
    assert(!boxes.empty());
    for (auto ibox : boxes)
      psend[ibox].push_back(data[i]);
  }
#endif

  double dtime = 1.e9;
  {
    const double t0 = MPI_Wtime();
    alltoallv(psend, precv);
    const double t1 = MPI_Wtime();
    dtime = t1 - t0;
    if (isMaster())
      fprintf(stderr, "alltoallv= %g sec \n", t1-t0);
  }

#if 0 
  {
    assert(precv[rank].size() == precv[rank].size());
    for(int p=0; p<nrank; p++)
      for (int i = 0; i < (int)precv[p].size(); i++)
        assert(boundary.isinbox(vector3{{precv[p][i].posx, precv[p][i].posy, precv[p][i].posz}}));
  }
#endif


  assert(precv[rank].size() == psend[rank].size());

  int nsendtot = 0, nrecvtot = 0;
  for(int p=0; p<nprocs; p++)
  {
    nsendtot += psend[p].size();
    nrecvtot += precv[p].size();
  }
  int nsendloc = nsendtot, nrecvloc = nrecvtot;
  MPI_Allreduce(&nsendloc,&nsendtot,1, MPI_INT, MPI_SUM,comm);
  MPI_Allreduce(&nrecvloc,&nrecvtot,1, MPI_INT, MPI_SUM,comm);
  double bw = 2.0 * double(sizeof(particle_t) * nsendtot) / dtime * 1.e-9;
  if (isMaster())
  {
    assert(nsendtot == nrecvtot);
    std::cout << "Exchanged particles = " << nsendtot << ", " << dtime << "sec" << std::endl;
    std::cout << "Global Bandwidth " << bw << " GB/s" << std::endl;
  }
  data.clear();
  data.resize(nrecvloc);
  int ip = 0;
  for(int p=0; p<nprocs; p++)
  {
    int size = precv[p].size();
    for(int i=0; i<size; i++)
      data[ip++] = precv[p][i];
  }
  assert(ip == nrecvloc);


  computeMinMax();
}

////// public
//

void RendererDataDistribute::distribute()
{
  initialize_division();
  std::vector<vector3> sample_array;
  collect_sample_particles(sample_array, sample_freq);

  /* determine division */
  vector3  xlow[NMAXPROC];
  vector3 xhigh[NMAXPROC];
  const float rmax = std::max(std::abs(_rmin), std::abs(_rmax)) * 1.0001;

  const int nsample = sample_array.size();
  if (rank == 0)
    fprintf(stderr, " -- nsample= %d\n", nsample);
  std::vector<float4> pos(nsample);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nsample; i++)
    pos[i] = float4(sample_array[i][0], sample_array[i][1], sample_array[i][2],0.0f);

  if (rank == 0)
    determine_division(pos, rmax, xlow, xhigh);


  const int nwords=nrank*3;
  MPI_Bcast(& xlow[0],nwords,MPI_DOUBLE,getMaster(),comm);
  MPI_Bcast(&xhigh[0],nwords,MPI_DOUBLE,getMaster(),comm);

#if 0
  if (isMaster())
  {
    for (int p = 0; p < nrank; p++)
    {
      fprintf(stderr," rank= %d: xlow= %g %g %g  xhigh= %g %g %g \n", p,
          xlow[p][0],  xlow[p][1],  xlow[p][2],
          xhigh[p][0], xhigh[p][1], xhigh[p][2]);
    }

  }
#endif
  exchange_particles_alltoall_vector(xlow, xhigh);

  for (int k = 0; k < 3; k++)
  {
    this-> xlow[k] =  xlow[rank][k];
    this->xhigh[k] = xhigh[rank][k];
  }

#if 0
  {
    const int nbody = data.size();
    Boundary boundary(xlow[rank], xhigh[rank]);
    for(int i=0; i<nbody; i++)
      assert(boundary.isinbox(vector3{{data[i].posx,data[i].posy,data[i].posz}}));
  }
#endif
  distributed = true;
}
