#pragma once

#if 0
#define PARALLELSORT
#endif

#ifdef PARALLELSORT
#include <parallel/algorithm>
#else
#include <algorithm>
#endif
#include <cassert>
#include <mpi.h>
#include <vector>

struct DD2D
{
  struct Key
  {
    enum { SIZEFLT = 2}; 
    unsigned long long key;

    Key() { assert(sizeof(Key) == SIZEFLT*sizeof(float)); }
    Key(const unsigned long long _key) : key(_key) {}

    bool operator< (const Key &a) const { return key <  a.key; }
    bool operator<=(const Key &a) const { return key <= a.key; }
    bool operator> (const Key &a) const { return key >  a.key; }
    bool operator>=(const Key &a) const { return key >= a.key; }
    bool operator==(const Key &a) const { return key == a.key; }

    static Key min() { return Key(0); }
    static Key max() { return Key(0xFFFFFFFFFFFFFFFFULL); }


    bool operator () (const Key &a, const Key &b) {  return a.key < b.key; }
  };

  private:

  const int procId, npx, nProc;
  const std::vector<Key> &key_sample1d;
  const std::vector<Key> &key_sample2d;
  const MPI_Comm mpi_comm;

  std::vector<Key> boundaries;

  public:

  const Key& keybeg(const int proc) const {return boundaries[proc  ];}
  const Key& keyend(const int proc) const {return boundaries[proc+1];}

  private:
  
  void GatherAllKeys(const std::vector< std::vector<Key> > &keys2send, std::vector<Key> &keys2recv)
  {
    const int np = keys2send.size();
#if 0
    for (int root = 0; root < np; root++)
    {
      std::vector<int> keys_sizes(nProc);
      int keys_size = keys2send[root].size() * Key::SIZEFLT;
      MPI_Gather(&keys_size, 1, MPI_INT, &keys_sizes[0], 1, MPI_INT, root, mpi_comm);

      std::vector<int> keys_displ(nProc+1,0);
      for (int i = 0; i < nProc; i++)
        keys_displ[i+1] = keys_displ[i] + keys_sizes[i];

      const int keys_recv = keys_displ[nProc] / Key::SIZEFLT;
      assert(keys_recv * Key::SIZEFLT == keys_displ[nProc]);
      const int keys2recv_size = keys2recv.size();
      keys2recv.resize(keys2recv_size + keys_recv);
      MPI_Gatherv(
          (void*)&keys2send[root][0], keys_size, MPI_FLOAT,
          (void*)&keys2recv[keys2recv_size], &keys_sizes[0], &keys_displ[0], MPI_FLOAT, root, mpi_comm);
    } 
#else
    std::vector<int> keys2send_size(nProc,0), keys2recv_size(nProc), keys2recv_displ(nProc+1,0);

    for (int p = 0; p < np; p++)
      keys2send_size[p] = keys2send[p].size();
    MPI_Alltoall(&keys2send_size[0], 1, MPI_INT, &keys2recv_size[0], 1, MPI_INT, mpi_comm);
    
    for (int p = 0; p < nProc; p++)
      keys2recv_displ[p+1] = keys2recv_displ[p] + keys2recv_size[p];

    keys2recv.resize(keys2recv_displ[nProc]);

    std::vector<MPI_Request> req(nProc*2);

    int nreq = 0;
    for (int p = 0; p < nProc; p++)
    {
      const int nsend = keys2send_size[p];
      const int nrecv = keys2recv_size[p];
      if (p >= npx) assert(nsend == 0);
      if (nsend > 0)
        MPI_Isend((void*)&keys2send[p][0], nsend*Key::SIZEFLT, MPI_FLOAT, p, 1, mpi_comm, &req[nreq++]);
      if (nrecv > 0)
        MPI_Irecv((void*)&keys2recv[keys2recv_displ[p]], nrecv*Key::SIZEFLT, MPI_FLOAT, p, 1, mpi_comm, &req[nreq++]);
    }
    std::vector<MPI_Status> status(nProc*2);
    MPI_Waitall(nreq, &req[0], &status[0]);
#endif
  }

  void GatherKeys(const int root, const std::vector<Key> &keys2send, std::vector<Key> &keys2recv)
  {
    std::vector<int> keys_sizes(nProc);

#if 1
    int keys_size = keys2send.size() * Key::SIZEFLT;
    MPI_Gather(&keys_size, 1, MPI_INT, &keys_sizes[0], 1, MPI_INT, root, mpi_comm);

    std::vector<int> keys_displ(nProc+1,0);
    for (int i = 0; i < nProc; i++)
      keys_displ[i+1] = keys_displ[i] + keys_sizes[i];

    const int keys_recv = keys_displ[nProc] / Key::SIZEFLT;
    assert(keys_recv * Key::SIZEFLT == keys_displ[nProc]);
    const int keys2recv_size = keys2recv.size();
    keys2recv.resize(keys2recv_size + keys_recv);
    MPI_Gatherv(
        (void*)&keys2send[0], keys_size, MPI_FLOAT,
        (void*)&keys2recv[keys2recv_size], &keys_sizes[0], &keys_displ[0], MPI_FLOAT, root, mpi_comm);
#else
    if (procId != root) 
    {
      int keys_size = keys2send.size();
      MPI_Send(       &keys_size,                           1, MPI_INT,   root, procId*2,     mpi_comm);
      MPI_Send((void*)&keys2send[0], keys_size * Key::SIZEFLT, MPI_FLOAT, root, procId*2 + 1, mpi_comm);
    } 
    else 
    {
      keys2recv.insert(keys2recv.end(), keys2send.begin(), keys2send.end());
      int keys_size = keys2recv.size();
      for (int p = 0; p < nProc; p++) 
        if (p != root)
        {
          MPI_Status status;
          int nreceive;
          MPI_Recv(&nreceive, 1, MPI_INT, p, p*2, mpi_comm, &status);
          //          assert(status == MPI_SUCCESS);
          keys2recv.resize(keys_size + nreceive);
          MPI_Recv(&keys2recv[keys_size], nreceive*Key::SIZEFLT, MPI_FLOAT, p, p*2 + 1, mpi_comm, &status);
          //          assert(status == MPI_SUCCESS);
          keys_size += nreceive;
        } 
    }
#endif
  }

  void chopSortedKeys(const unsigned long long np, const Key &minkey, const std::vector<Key> &keys2chop, std::vector<Key> &boundaries)
  {
    boundaries.resize(np);
    const unsigned long long size = keys2chop.size();

    boundaries[0] = minkey;
    for (unsigned long long i = 1;  i < np; i++)
    {
      const int idx = static_cast<int>(i*size/np);
      boundaries[i] = keys2chop[idx];
    }
    for (int i = 0; i < np-1; i++)
      assert(boundaries[i] < boundaries[i+1]);
  }

  /* key_sample must be sorted in an increasing Key order, otherwise it will fail */
  void assignKeysToProc(
      const std::vector<Key> &key_sample,  
      const std::vector<Key> &boundaries,  
      std::vector< std::vector<Key> > &keys)
  {
    const int np = boundaries.size();
    keys.resize(np);
    const int sample_size = key_sample.size();

    std::vector<int> firstKey(np+2);

    int location       = 0;
    firstKey[location] = 0;

    for (int i = 0; i < sample_size; i++)
    {
      const Key key = key_sample[i];

      bool assigned = false;
      while(!assigned)
      {
        const Key lowerBoundary = boundaries[location  ];
        const Key upperBoundary = boundaries[location+1];

        assert(key >= lowerBoundary);

        if(key < upperBoundary) 
        {    
          assigned = true;    /* is in box */
        }
        else
        {
          firstKey[++location] = i;    /* outside the box */
          assert(location < np);
        }
      }
    }

    //Fill remaining processes
    while(location <= np)
      firstKey[++location] = sample_size;

    for (int p = 0; p < np; p++)
      if (firstKey[p+1] > firstKey[p])
        keys[p].insert(keys[p].begin(), key_sample.begin()+firstKey[p], key_sample.begin()+firstKey[p+1]);

#if 0  /*** diagnostic ***/
    for (int p = 0; p < np; p++)
    {
      fprintf(stderr, " dd2d:: procId= %d  sends to %d a total of %d keys out of %d\n",
          procId, p, (int)keys[p].size(), sample_size);
    }
#endif

  }

  public:

  /* sample_keys must be sorted by Key in an increaing order, otherwise
   * assignKeyToProc will fail  */
  DD2D(const int _procId, const int _npx, const int _nProc, const std::vector<Key> &_key_sample1d, const std::vector<Key> &_key_sample2d, const MPI_Comm &_mpi_comm) :
    procId(_procId), npx(_npx), nProc(_nProc), key_sample1d(_key_sample1d), key_sample2d(_key_sample2d),mpi_comm(_mpi_comm)
  {
    assert(nProc % npx == 0);
    const int npy = nProc / npx;

    /*** do first 1D domain decomposition ***/

    /* sample list of key,
     * each proc samples nsamples_tot/nProc keys
     */

    /* gather keys to proc 0 */

    std::vector<Key> keys1d_recv;
    GatherKeys(0, key_sample1d, keys1d_recv);
    if (procId == 0)
      fprintf(stderr, " dd2d:: 1st step gathered= %d keys \n", (int)keys1d_recv.size());

    /* compute npx boundaries, from nsamples_tot keys */

    std::vector<Key> boundaries1d(npx, Key::min());
    if (procId == 0)
    {
#ifdef PARALLELSORT
      __gnu_parallel::sort(keys1d_recv.begin(), keys1d_recv.end(), Key());
#else
      std::sort(keys1d_recv.begin(), keys1d_recv.end(), Key());
#endif
      chopSortedKeys(npx, Key::min(), keys1d_recv, boundaries1d);
    }

    /* boradcast 1d boundaries to all procs */

    MPI_Bcast(&boundaries1d[0], npx*Key::SIZEFLT, MPI_FLOAT, 0, mpi_comm);
    boundaries1d.push_back(Key::max());

    /*** proceed with 2D decomposition ***/

    /* at this point each proc has informationa about npx sorting bins 
     * and locally each proc populate those bins with samples key and export
     * the each of the corresponding npx procs 
     */

    /* sanity test, make sure 1d boundaries are sorted */

    for (int i = 0; i < npx; i++)
      assert(boundaries1d[i] < boundaries1d[i+1]);

    /* each proc resample local particle, however,
     * we can increase local sampling rate by npx 
     * local sampling rate becomes nsamples_loc = (nsamples_tot / nProc) * npx  
     */

    /* assign each of the sampled keys to appropriate proc */

    std::vector< std::vector<Key> > keys2d_send;
    assignKeysToProc(key_sample2d, boundaries1d, keys2d_send);

    /* gather keys from remote procs to the first npx sorting procs */

    std::vector<Key> keys2d_recv;
#if 0
    for (int p = 0; p < npx; p++)
      GatherKeys(p, keys2d_send[p], keys2d_recv);
#else
    GatherAllKeys(keys2d_send, keys2d_recv);
#endif


    if (procId < npx)
      fprintf(stderr, "dd2d:: 2nd step proc= %d gathered= %d keys \n", procId, (int)keys2d_recv.size());

    /* sanity test, only first npx must recieve data */
    if (procId >= npx) assert(keys2d_recv.empty());

    /* each of npx sorting proc, generate 2d boundaries */
    std::vector<Key> boundaries2d(npy);
    if (procId < npx)
    {
#ifdef PARALLELSORT
      __gnu_parallel::sort(keys2d_recv.begin(), keys2d_recv.end(), Key());
#else
      std::sort(keys2d_recv.begin(), keys2d_recv.end(), Key());
#endif
      const Key minkey = boundaries1d[procId];
      chopSortedKeys(npy, minkey, keys2d_recv, boundaries2d);
    }

    /* gather 2d boundaries */
    /* there could be a better way to do the following two steps... */

    /* first gather npx boundaries on proc 0, to merge into a global boundary */
    boundaries.resize(nProc+1);
    boundaries[nProc] = Key::max();

    std::vector<int> boundaries_size(nProc,0), boundaries_displ(nProc+1, 0);
    for (int i = 0; i < npx; i++)
    {
      boundaries_size [i  ] = npy * Key::SIZEFLT;
      boundaries_displ[i+1] = boundaries_displ[i] + boundaries_size[i];
    }
    assert(boundaries_displ[npx] == nProc *Key::SIZEFLT);
    for (int i = npx; i < nProc; i++)
      boundaries_displ[i] = boundaries_displ[npx-1];

    MPI_Gatherv(
        (void*)&boundaries2d[0], procId < npx ? npy*Key::SIZEFLT : 0, MPI_FLOAT,
        (void*)&boundaries  [0], &boundaries_size[0], &boundaries_displ[0], MPI_FLOAT, 0, mpi_comm);

    /* then broadcast boundaries from proc 0 to all */

    MPI_Bcast(&boundaries[0], nProc*Key::SIZEFLT, MPI_FLOAT, 0, mpi_comm);

    /* sanity checks */

    assert(boundaries[    0] == Key::min());
    assert(boundaries[nProc] == Key::max());
  }
};
