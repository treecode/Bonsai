#pragma once

#include <sys/types.h> /* pid_t */
#include <unistd.h>  /* _exit, fork */
#include <sys/wait.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>


extern "C"
{
  void gen_disk(int nptcl, int myseed, float *buffer, int verbose);
  void gen_bulge(int nptcl, int myseed, float *buffer, int verbose);
  void gen_halo (int nptcl, int myseed, float *buffer, int verbose);
}

struct Galactics
{
  struct Particle
  {
    int id;
    float mass;
    float x,y,z;
    float vx,vy,vz;
  };

  private:
  int ndisk, nbulge, nhalo;
  int childId;
  std::vector<Particle> ptcl;

  public:

  int get_ndisk() const { return ndisk; }
  int get_nbulge() const { return nbulge; }
  int get_nhalo() const { return nhalo; }
  int get_ntot() const { return ndisk+nbulge+nhalo; }
  const Particle& operator[](const int i) const {return ptcl[i];}

  Galactics(const int procId, const int _ndisk, int _nbulge, int _nhalo, const int NCHILD = 4) : 
    ndisk(_ndisk/NCHILD), nbulge(_nbulge/NCHILD), nhalo(_nhalo/NCHILD), childId(0)
  {
    const int nptcl = ndisk + nbulge + nhalo;

    const int nByteMap = nptcl*NCHILD*sizeof(Particle);

    /* allocate shared memory */

    pid_t pid;
    char shfn[256];
    sprintf(shfn, "/BONSAISHARED-PROC-%d", procId);
    int shmfd = shm_open(shfn, O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
    if (shmfd == -1)
    {
      fprintf(stderr, "procId= %d :: creator:shm_open failed \n", procId);
      perror("creator:shm_open");
      exit(EXIT_FAILURE);
    }
    if (ftruncate(shmfd,nByteMap))
    {
      fprintf(stderr, "procId= %d :: creator:ftruncate failed \n", procId);
      perror("creator:ftruncate");
      exit(EXIT_FAILURE);
    }

    void *data_ptr = mmap(NULL,nByteMap,PROT_READ|PROT_WRITE, MAP_SHARED,shmfd,0);
    assert(data_ptr != MAP_FAILED);

    /* fork process */

    for (int i = 0; i < NCHILD; i++)
    {
      pid = fork();
      if (pid < 0)
      {
        fprintf(stderr, " procId= %d  parent :: failed to fork ... \n", procId);
        exit(EXIT_FAILURE);
      } 
      else if (pid == 0)
        break;
      childId++;
    }

    if (pid > 0 && procId == 0)
      fprintf(stderr," parent :: done forking ... \n");

    assert(pid >= 0);

#if 0
    if (pid == 0)  
      fprintf(stderr,"I am a child process: id= %d  pid = %d\n", childId, getpid());
    else if (pid  > 0)
      fprintf(stderr,"I am a parent process: pid = %d\n", getpid());
    else
      assert(0);
#endif

    if (pid == 0)
    {
      /* child processes generating a small galaxy each */
      int shmfd = shm_open(shfn, O_RDWR,0);
      if(shmfd == -1)
      {
        fprintf(stderr, "procId= %d childId= %d:: child:ftruncate failed \n", procId, childId);
        perror("child:shm_open");
        exit(EXIT_FAILURE);
      }
      void *data_ptr = mmap(NULL,nByteMap,PROT_READ|PROT_WRITE, MAP_SHARED,shmfd,0);
      assert(data_ptr != MAP_FAILED);

      Particle *data = &((Particle*)data_ptr)[nptcl*childId];

      /* generate galaxy */

      const int verbose = (childId == 0) && (procId == 0);
      const int a = procId;
      const int b = childId;
      gen_disk (ndisk,  131072*a+34*b, (float*)&data[0           ], verbose);
      gen_bulge(nbulge, 262144*a+67*b, (float*)&data[ndisk       ], verbose);
      gen_halo (nhalo,  524288*a+97*b, (float*)&data[ndisk+nbulge], verbose);

      exit(EXIT_SUCCESS);
    }

    /* from here only parent process is active */

    if (pid > 0)
    {
      int status;
      for (int i = 0; i < NCHILD; i++)
      {
        int wpid = wait(&status);
        if (procId == 0)
          fprintf(stderr,"Child pid= %d done with status= %d \n", wpid, status);
      }
      if (procId == 0)
        fprintf(stderr,"\n parent :: Galaxy generation complete \n");
      const Particle *ptcl_list = (Particle*)data_ptr;

      /* collect the results from child processes */

      const int ntot = nptcl*NCHILD;
      if (procId == 0)
        fprintf(stderr, "nptcl_per_proc= %d  nproc= %d  ntot= %d\n",
            nptcl, NCHILD, ntot);
      const float mscale = 1.0/NCHILD;
      ptcl.insert(ptcl.begin(), ptcl_list, ptcl_list+ntot);
      for (int i = 0; i < ntot; i++)
      {
        assert(ptcl[i].mass > 0);
        assert(!std::isnan(ptcl[i].x));
        assert(!std::isnan(ptcl[i].y));
        assert(!std::isnan(ptcl[i].z));
        assert(!std::isnan(ptcl[i].vx));
        assert(!std::isnan(ptcl[i].vy));
        assert(!std::isnan(ptcl[i].vz));
        ptcl[i].mass *= mscale;
      }
    }

    close(shmfd);
    munmap((void*)data_ptr,nByteMap);
    shm_unlink(shfn);
    ndisk  *= NCHILD;
    nbulge *= NCHILD;
    nhalo  *= NCHILD;
  }
};
