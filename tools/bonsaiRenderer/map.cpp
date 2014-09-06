/* (x0<=x<x1, y0<=y<y1: an image in viewport */

struct int2 { int x,y; };
struct float4 {float x,y,z,w;};
static float4 make_float4(const float f) { return (float4){f,f,f,f}; }
#include <mpi.h>
#include <vector>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <algorithm>

void lCompose(
    const float4* imgSrc,
    const float*  depthSrc,
    float4* imgDst,
    const int rank, const int nrank, const MPI_Comm &comm,
    const int2 imgCrd,
    const int2 imgSize,
    const int2 viewportSize,
    const std::vector<int> &compositeOrder)
{
  constexpr int master = 0;

  using imgData_t = std::array<float,5>;
  constexpr int mpiImgDataSize = sizeof(imgData_t)/sizeof(float);
  static std::vector<imgData_t> sendbuf;

  /* copy img pixels ot send buffer */

  const int imgNPix = imgSize.x*imgSize.y;
  if (imgNPix > 0)
  {
    sendbuf.resize(imgNPix);
    if (compositeOrder.empty() && depthSrc)
    {
      /* if there is no global composition order and depth is != NULL */
#pragma omp parallel for schedule(static)
      for (int i = 0; i < imgNPix; i++)
        sendbuf[i] = imgData_t{{imgSrc[i].x, imgSrc[i].y, imgSrc[i].z, imgSrc[i].w, depthSrc[i]}};
    }
    else if (compositeOrder.empty())
    {
      /* if there is no global composition order and no depth is passed */
#pragma omp parallel for schedule(static)
      for (int i = 0; i < imgNPix; i++)
        sendbuf[i] = imgData_t{{imgSrc[i].x, imgSrc[i].y, imgSrc[i].z, imgSrc[i].w, static_cast<float>(rank)}};
    }
    else
    {
      /* if there is global composition order */
#pragma omp parallel for schedule(static)
      for (int i = 0; i < imgNPix; i++)
        sendbuf[i] = imgData_t{{imgSrc[i].x, imgSrc[i].y, imgSrc[i].z, imgSrc[i].w, static_cast<float>(compositeOrder[rank])}};
    }
  }

  /* compute which parts of img are sent to which rank */
  
  const int nPixels = viewportSize.x*viewportSize.y;
  const int nPixelsPerRank = (nPixels+nrank-1)/nrank; 

  const int x0 = imgCrd.x;
  const int y0 = imgCrd.y;
  const int x1 = imgCrd.x + imgSize.x;
  const int y1 = imgCrd.y + imgSize.y;

  const int w = viewportSize.x;

  const int imgBeg   =  y0   *w + x0;
  const int imgEnd   = (y1-1)*w + x1-1;
  const int imgWidth = imgSize.x;

  using imgMetaData_t = std::array<int,5>;
  constexpr  int mpiImgMetaDataSize = sizeof(imgMetaData_t)/sizeof(int);

  static std::vector<imgMetaData_t> srcMetaData(nrank);
  int totalSendCount = 0;
  for (int p = 0; p < nrank; p++)
  {
    /* domain scanline beginning & end */
    const int pbeg =    p * nPixelsPerRank;
    const int pend = pbeg + nPixelsPerRank-1;

    /* clip image with the domain scanline */
    const int clipBeg = std::min(pend, std::max(imgBeg,pbeg));
    const int clipEnd = std::max(pbeg, std::max(imgEnd,pend));

    int sendcount = 0;
    if (clipBeg < clipEnd)
    {
      const int i0 = clipBeg % w;
      const int i1 = clipEnd % w;
      const int j0 = clipBeg / w;
      const int j1 = clipEnd / w;
      assert(j0 >= y0);
      assert(j1 <  y1);

      /* compute number of pixels to send: 
       * multiply hight (j1-j0+1) by image width */
      /* but subtract top and bottom corners */
      sendcount = 
        + (j1-j0+1)*imgWidth
        - std::max(0,std::min(i0-x0,   imgWidth))
        - std::max(0,std::min(x1-i1-1, imgWidth));
    }
    
    srcMetaData[p] = imgMetaData_t{{x0,y0,x1,y1,sendcount}};
    totalSendCount += sendcount;
  }
  assert(totalSendCount == (x1-x0)*(y1-y0));

  /* exchange metadata info */

  static std::vector<imgMetaData_t> rcvMetaData(nrank);
  MPI_Alltoall(&srcMetaData[0], mpiImgMetaDataSize, MPI_INT, &rcvMetaData[0], mpiImgMetaDataSize, MPI_INT, comm);

  /* prepare counts & displacements for alltoallv */

  static std::vector<int> sendcount(nrank), senddispl(nrank+1);
  static std::vector<int> recvcount(nrank), recvdispl(nrank+1);
  senddispl[0] = recvdispl[0] = 0;
  for (int p = 0; p < nrank; p++)
  {
    sendcount[p  ] = srcMetaData[p][4] * mpiImgDataSize;
    recvcount[p  ] = rcvMetaData[p][4] * mpiImgDataSize;
    senddispl[p+1] = senddispl[p] + sendcount[p];
    recvdispl[p+1] = recvdispl[p] + recvcount[p];
  }

  static std::vector<imgData_t> recvbuf;
  {
    if (recvdispl[nrank] > 0)
      recvbuf.resize(recvdispl[nrank] / mpiImgDataSize);
    const double t0 = MPI_Wtime();
    MPI_Alltoallv(
        &sendbuf[0], &sendcount[0], &senddispl[0], MPI_FLOAT,
        &recvbuf[0], &recvcount[0], &recvdispl[0], MPI_FLOAT,
        comm);
    double nsendrecvloc = (senddispl[nrank] + recvdispl[nrank])*sizeof(float);
    double nsendrecv;
    MPI_Allreduce(&nsendrecvloc, &nsendrecv, 1, MPI_DOUBLE, MPI_SUM, comm);
    const double t1 = MPI_Wtime();
    if (rank == master)
    {
      const double dt = t1-t0;
      const double bw = nsendrecv / dt;
      fprintf(stderr, " MPI_Alltoallv: dt= %g  BW= %g MB/s  mem= %g MB\n", dt, bw/1e6, nsendrecv/1e6);
    }
  }

  /* pixel composition */

  const int pixelBeg =              rank * nPixelsPerRank;
  const int pixelEnd = std::min(pixelBeg + nPixelsPerRank, nPixels);

  constexpr int NRANKMAX = 1024;
  assert(nrank <= NRANKMAX);
    
  for (int p = 0; p < nrank+1; p++)
    recvdispl[p] /= mpiImgDataSize;

  static std::vector<float4> imgLoc;
  imgLoc.resize(nPixelsPerRank);
#pragma omp parallel for schedule(static)
  for (int idx = pixelBeg; idx < pixelEnd; idx++)
  {
    int pcount = 0;
    imgData_t imgData[NRANKMAX];

    const int i = idx % viewportSize.x;
    const int j = idx / viewportSize.x;

    for (int p = 0; p < nrank; p++)
    {
      const int x0 = rcvMetaData[p][0];
      const int y0 = rcvMetaData[p][1];
      const int x1 = rcvMetaData[p][2];
      const int y1 = rcvMetaData[p][3];
      if (rcvMetaData[p][5] && x0 <= i && i < x1  && y0 <= j && j < y1)
        imgData[pcount++] = recvbuf[recvdispl[p] + (j - x0)*(x1-x0) + (i - x0)];
    }

    std::sort(imgData, imgData+pcount, 
        [](const imgData_t &a, const imgData_t &b) { return a[4] < b[4]; });

    float4 dst = make_float4(0.0f);
    for (int p = 0; p < pcount; p++)
    {
      auto &src = imgData[p];
      src[0] *= 1.0f - dst.w;
      src[1] *= 1.0f - dst.w;
      src[2] *= 1.0f - dst.w;
      src[3] *= 1.0f - dst.w;

      dst.x += src[0];
      dst.y += src[1];
      dst.z += src[2];
      dst.w += src[3];

      dst.w = std::min(dst.w, 1.0f);
    }
    imgLoc[idx - pixelBeg] = dst;
  }

  /* gather composited part of images into a single image on the master rank */
  {
    const double t0 = MPI_Wtime();
    MPI_Gather(&imgLoc[0], nPixelsPerRank*4, MPI_FLOAT, imgDst, 4*nPixelsPerRank, MPI_FLOAT, master, comm);
    const double t1 = MPI_Wtime();
    if (master == rank)
    {
      const double dt        = t1 - t0;
      const double nsendrecv = nPixelsPerRank*4*nrank*sizeof(float);
      const double bw        = nsendrecv / dt;
      fprintf(stderr, " MPI_Gather: dt= %g  BW= %g MB/s  mem= %g MB\n", dt, bw/1e6, nsendrecv/1e6);
    }
  }
}

