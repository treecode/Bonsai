#include "RendererData.h"
#include <omp.h>


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


  // divide on x
  {
    double buf[NMAXPROC+1];
    double *xoff = buf; // xoff[nx+1]
    __gnu_parallel::sort(&pos[addr.off(0, 0, 0)], &pos[addr.off(nx, 0, 0)], 
        [](const float4 &lhs, const float4 &rhs)  {
        constexpr int mask = 1;
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
#pragma omp parallel for schedule(static)
    for(int ix=0; ix<nx; ix++)
    {
      double buf[NMAXPROC+1];
      double *yoff = buf; // yoff[ny+1];
      std::sort(&pos[addr.off(ix, 0, 0)], &pos[addr.off(ix, ny, 0)], 
          [](const float4 &lhs, const float4 &rhs)  {
          constexpr int mask = 2;
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
#pragma omp parallel for schedule(static) collapse(2)
    for(int ix=0; ix<nx; ix++)
      for(int iy=0; iy<ny; iy++)
      {
        double buf[NMAXPROC+1];
        double *zoff = buf; // zoff[nz+1];

        std::sort(&pos[addr.off(ix, iy, 0)], &pos[addr.off(ix, iy, nz)], 
            [](const float4 &lhs, const float4 &rhs)  {
            constexpr int mask = 4;
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


void RendererDataDistribute::which_boxes_z(
    int p,
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> &boxes)
{

  int npz = this->npz;
  for(int iz=0; iz<npz; iz++, p++){
    if(pos[2]+h >= xlow[p][2]  && pos[2]-h <= xhigh[p][2]){
      boxes.push_back(p);
    }
  }

}

void RendererDataDistribute::which_boxes_y(
    int p,
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> &boxes)
{
  int npy = this->npy;
  for(int iy=0; iy<npy; iy++, p+=npz){
    if(pos[1]+h >= xlow[p][1]  && pos[1]-h <= xhigh[p][1]){
      which_boxes_z(p, pos, h, xlow, xhigh, boxes);
    }
  }

}

void RendererDataDistribute::which_boxes_x(
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> &boxes)
{
  int p=0;
  int npx = this->npx;
  for(int ix=0; ix<npx; ix++, p+=npy*npz){
    if(pos[0]+h >= xlow[p][0]  && pos[0]-h <= xhigh[p][0]){
      which_boxes_y(p, pos, h, xlow, xhigh, boxes);
    }
  }
}

void RendererDataDistribute::which_boxes(
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> &boxes)
{
#if 0  /* naive: O(nrank) */
  for (int p = 0; p < nrank; p++)
  {
    if (
        pos[0]+h >= xlow[p][0]  && pos[0]-h <= xhigh[p][0] &&
        pos[1]+h >= xlow[p][1]  && pos[1]-h <= xhigh[p][1] &&
        pos[2]+h >= xlow[p][2]  && pos[2]-h <= xhigh[p][2])
      boxes.push_back(p);
  }
#else  /* optimized: O(nrank^{1/3}) */
  which_boxes_x(pos, h, xlow, xhigh, boxes);
#endif
}

void RendererDataDistribute::which_boxes_z(
    int p, const int i,
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> boxes[NMAXPROC])
{

  int npz = this->npz;
  for(int iz=0; iz<npz; iz++, p++)
    if(pos[2]+h >= xlow[p][2]  && pos[2]-h <= xhigh[p][2])
      boxes[p].push_back(i);
}

void RendererDataDistribute::which_boxes_y(
    int p,  const int i,
    const vector3 &pos,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> boxes[NMAXPROC])
{
  int npy = this->npy;
  for(int iy=0; iy<npy; iy++, p+=npz)
    if(pos[1]+h >= xlow[p][1]  && pos[1]-h <= xhigh[p][1])
      which_boxes_z(p, i, pos, h, xlow, xhigh, boxes);
}

void RendererDataDistribute::which_boxes_x(
    const vector3 &pos,
    const int i,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> boxes[NMAXPROC])
{
  int p=0;
  int npx = this->npx;
  for(int ix=0; ix<npx; ix++, p+=npy*npz)
    if(pos[0]+h >= xlow[p][0]  && pos[0]-h <= xhigh[p][0])
      which_boxes_y(p, i, pos, h, xlow, xhigh, boxes);
}

void RendererDataDistribute::which_boxes(
    const vector3 &pos, 
    const int i,
    const float h,
    const vector3 xlow[],
    const vector3 xhigh[],
    std::vector<int> boxes[NMAXPROC])
{
  which_boxes_x(pos,i, h, xlow, xhigh, boxes);
}

void RendererDataDistribute::exchange_particles_alltoall_vector(
    const vector3  xlow[],
    const vector3 xhigh[])

{
  const double t00 = MPI_Wtime();

  static std::vector<particle_t> sendbuf;
  static std::vector<int> sendcount(nrank  ), recvcount(nrank  );
  static std::vector<int> senddispl(nrank+1), recvdispl(nrank+1);

  constexpr int NTHREADMAX = 8;
  static std::vector<int> sendidx[NTHREADMAX][NMAXPROC];

  const int np = data.size();

  const double t10 = MPI_Wtime();

  static int nthreadsHW = 0;
  if (nthreadsHW == 0)
#pragma omp critical
    nthreadsHW++;
  nthreadsHW++;

  const static int nthreads = std::min(NTHREADMAX,nthreadsHW);

#pragma omp parallel num_threads(nthreads)
  {
    const int tid = omp_get_thread_num();
    for (int p = 0; p < nrank; p++)
    {
      sendidx[tid][p].clear();
      sendidx[tid][p].reserve(4096);
    }

#pragma omp for schedule(static)
    for (int i = 0; i < np; i++)
    {
      which_boxes(
          {data[i].posx,data[i].posy,data[i].posz},
          i,
          hfac*data[i].attribute[Attribute_t::H], 
          xlow, xhigh, sendidx[tid]);
    }
  }
  
  senddispl[0] = 0;
  int sendcountmax = 0;
  for (int p = 0; p < nrank; p++)
  {
    sendcount[p] = 0;
    for (int tid = 0; tid < nthreads; tid++)
      sendcount[p] += sendidx[tid][p].size();
    senddispl[p+1] = senddispl[p] + sendcount[p];
    sendcountmax = std::max(sendcountmax, sendcount[p]);
  }
  
  const double t20 = MPI_Wtime();

  MPI_Alltoall(&sendcount[0], 1, MPI_INT, &recvcount[0], 1, MPI_INT, comm);

  recvdispl[0] = 0;
  for (int p = 0; p < nrank; p++)
    recvdispl[p+1] = recvdispl[p] + recvcount[p];

  const double t30 = MPI_Wtime();

  sendbuf.resize(senddispl[nrank]);
#pragma omp parallel for schedule(static)
  for (int p = 0; p < nrank; p++)
  {
    size_t base = senddispl[p];
    for (int tid = 0; tid < nthreads; tid++)
    {
      for (size_t i = 0; i < sendidx[tid][p].size(); i++)
        sendbuf[base+i] = data[sendidx[tid][p][i]];
      base += sendidx[tid][p].size();
      assert(base <= sendbuf.size());
    }
  }
  
  const double t40 = MPI_Wtime();

  data.resize(recvdispl[nrank]);
  auto recvbuf = &data[0];
  {
    const double t0 = MPI_Wtime();
    static MPI_Datatype MPI_PARTICLE = 0;
    if (!MPI_PARTICLE)
    {
      int ss = sizeof(particle_t) / sizeof(float);
      assert(0 == sizeof(particle_t) % sizeof(float));
      MPI_Type_contiguous(ss, MPI_FLOAT, &MPI_PARTICLE);
      MPI_Type_commit(&MPI_PARTICLE);
    }
    MPI_Alltoallv(
        &sendbuf[0], &sendcount[0], &senddispl[0], MPI_PARTICLE,
        &recvbuf[0], &recvcount[0], &recvdispl[0], MPI_PARTICLE,
        comm);
    const double t1 = MPI_Wtime();
    const double dtime = t1 - t0;

    size_t nsendrecvloc = senddispl[nrank] + recvdispl[nrank];
    size_t nsendrecv;
    MPI_Allreduce(&nsendrecvloc,&nsendrecv,1, MPI_LONG_LONG, MPI_SUM,comm);
    double bw =  double(sizeof(particle_t))*nsendrecv / dtime * 1.e-9;
    if (isMaster())
    {
      std::cerr 
        << "Exchanged particles= " << nsendrecv / 1e6 << "M, " << dtime << " sec || "
        << "Global Bandwidth " << bw << " GB/s" << std::endl;
    }
  }

  const double t50 = MPI_Wtime();

  computeMinMax();

  const double t60 = MPI_Wtime();

#if 1
  fprintf(stderr, "xchg: rank= %d: dt= %g [ %g %g %g %g %g %g  ]\n", rank, t60-t00,
      t10-t00,t20-t10,t30-t20,t40-t30,t50-t40,t60-t50);
#endif
}

////// public
//

void RendererDataDistribute::distribute()
{
  const double t00 = MPI_Wtime();
  initialize_division();
  const double t10 = MPI_Wtime();
  std::vector<vector3> sample_array;
  collect_sample_particles(sample_array, sample_freq);
  const double t20 = MPI_Wtime();

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
  const double t30 = MPI_Wtime();

  if (rank == 0)
    determine_division(pos, rmax, xlow, xhigh);

  const double t40 = MPI_Wtime();

  const int nwords=nrank*3;
  MPI_Bcast(& xlow[0],nwords,MPI_DOUBLE,getMaster(),comm);
  MPI_Bcast(&xhigh[0],nwords,MPI_DOUBLE,getMaster(),comm);

  bounds.resize(nrank);
  for (int p = 0; p < nrank; p++)
    bounds[p] = {xlow[p][0],xlow[p][1],xlow[p][2]};

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
  const double t50 = MPI_Wtime();
  exchange_particles_alltoall_vector(xlow, xhigh);
  const double t60 = MPI_Wtime();

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
  const double t70 = MPI_Wtime();
  MPI_Barrier(comm);
#if 1
  fprintf(stderr, "dist: rank= %d: dt= %g [ %g %g %g %g %g %g %g ]\n", rank, t70-t00,
      t10-t00,t20-t10,t30-t20,t40-t30,t50-t40,t60-t50,t70-t60);
#endif
}

std::vector<int> RendererDataDistribute::getVisibilityOrder(const std::array<float,3> camPos) const
{
  if (!isDistributed())
  {
    std::vector<int> order(nrank);
    std::iota(order.begin(), order.end(), 0);
    return order;
  }
  const double t0 = MPI_Wtime();
  assert(static_cast<int>(bounds.size()) == nrank);

  auto xdi = [=](int ix, int iy, int iz)
  {
    return iz + npz*(iy + npy*(ix));
  };

  auto locate = [](float splits[], const int n, const float val)
  {
    auto up = std::upper_bound(splits, splits+n, val);
    const int idx = up - splits;
    return std::max(0, std::min(n-1,idx-1));
  };

  auto map = [](const int i, const int pxc, const int npx)
  {
    const int px = i <= pxc ? pxc-i : i;
    assert(px >= 0 && px < npx);
    return px;
  };


  static std::vector<int> compositingOrder(nrank);
  {
    constexpr int NRANKMAX = 1024;
    assert(npx <= NRANKMAX);
    assert(npy <= NRANKMAX);
    assert(npz <= NRANKMAX);

    float xsplits[NRANKMAX];
    for (int px = 0; px < npx; px++)
      xsplits[px] = bounds[xdi(px,0,0)][0];
    const int pxc = locate(xsplits, npx, camPos[0]);

#pragma omp parallel for schedule(static)
    for (int i = 0; i < npx; i++)
    {
      const int px = map(i,pxc,npx);

      float ysplits[NRANKMAX];
      for (int py = 0; py < npy; py++)
        ysplits[py] = bounds[xdi(px,py,0)][1];
      const int pyc = locate(ysplits, npy, camPos[1]);

      for (int j = 0; j < npy; j++)
      {
        const int py = map(j,pyc,npy);

        float zsplits[NRANKMAX];
        for (int pz = 0; pz < npz; pz++)
          zsplits[pz] = bounds[xdi(px,py,pz)][2];
        const int pzc = locate(zsplits, npz, camPos[2]);

        for (int k = 0; k < npz; k++)
        {
          const int pz = map(k,pzc,npz);
          compositingOrder[xdi(i,j,k)] = xdi(px,py,pz);
        }
      }
    }
  }

  assert(!compositingOrder.empty());
  assert(static_cast<int>(compositingOrder.size()) == nrank);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < nrank; i++)
    assert(compositingOrder[i] >= 0 && compositingOrder[i] < nrank);

  const double t2 = MPI_Wtime();
  if (isMaster())
    fprintf(stderr, " globalOrder: algorithm %g sec \n", t2-t0);

  return compositingOrder;
}
