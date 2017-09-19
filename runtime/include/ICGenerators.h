#pragma once

#include "plummer.h"
//#include "disk_shuffle.h"

struct DiskShuffle
{

  private:
    void read_file(
        FILE *in,
        std::vector<star_particle> &sp,
        std::vector<dark_particle> &dp,
        dump &h)
    {
      sp.clear();
      dp.clear();

      size_t res = fread(&h, sizeof(h), 1, in);

      const float time = h.time;
      const int nstar = h.nstar;
      const int ndark = h.ndark;

      for(int i=0;i<ndark;i++)
      {
        dark_particle _dp;
        res = fread(&_dp, sizeof(dark_particle), 1, in);
        dp.push_back(_dp);
      }
      for(int i=0;i<nstar;i++){
        star_particle _sp;
        res = fread(&_sp, sizeof(star_particle), 1, in);
        sp.push_back(_sp);
      }
    }

    void rotate_xy(const float angl, float val[3]){
      float rot[3];
      rot[0] = cos(angl)*val[0] - sin(angl)*val[1];
      rot[1] = sin(angl)*val[0] + cos(angl)*val[1];
      rot[2] = val[2];

      val[0] = rot[0];
      val[1] = rot[1];
      val[2] = rot[2];
      //return rot;
    }

    void rotate_yz(const float angl, float val[3]){
      float rot[3];
      rot[1] = cos(angl)*val[1] - sin(angl)*val[2];
      rot[2] = sin(angl)*val[1] + cos(angl)*val[2];
      rot[0] = val[0];

      val[0] = rot[0];
      val[1] = rot[1];
      val[2] = rot[2];
      //return rot;
    }

    void rotate_one_particle(float pos[3], float vel[3])
    {
      const double angl = (2*drand48()-1.)*M_PI;
      rotate_xy(angl, pos);
      rotate_xy(angl, vel);
      const float f = 1.0e-3;
      pos[0] *= 1 + f * (2.0*drand48()- 1.0);
      pos[1] *= 1 + f * (2.0*drand48()- 1.0);
      pos[2] *= 1 + f * (2.0*drand48()- 1.0);
      vel[0] *= 1 + f * (2.0*drand48()- 1.0);
      vel[1] *= 1 + f * (2.0*drand48()- 1.0);
      vel[2] *= 1 + f * (2.0*drand48()- 1.0);
    }

    std::vector<dvec3> _pos, _vel;
    std::vector<double> _mass;
    int nstar, ndark;

  public:
    DiskShuffle(const std::string &fileName)
    {
      FILE *fin = fopen(fileName.c_str(), "rb");
      if (!fin)
      {
        fprintf(stderr, "DiskShuffle:: file %s not found\n", fileName.c_str());
        assert(0);
      }

      std::vector<star_particle> sp;
      std::vector<dark_particle> dp;
      dump h;
      read_file(fin, sp, dp, h);

      _pos.clear();
      _vel.clear();
      _mass.clear();

      nstar = h.nstar;
      ndark = h.ndark;

      const int n = h.nstar+h.ndark;
      _pos.reserve(n);
      _vel.reserve(n);
      _mass.reserve(n);

      for (int i = 0; i < h.nstar; i++)
      {
        rotate_one_particle(sp[i].pos, sp[i].vel);
        _pos.push_back(dvec3(sp[i].pos[0], sp[i].pos[1], sp[i].pos[2]));
        _vel.push_back(dvec3(sp[i].vel[0], sp[i].vel[1], sp[i].vel[2]));
        _mass.push_back(sp[i].mass);
      }
      for (int i = 0; i < h.ndark; i++)
      {
        rotate_one_particle(dp[i].pos, dp[i].vel);
        _pos.push_back(dvec3(dp[i].pos[0], dp[i].pos[1], dp[i].pos[2]));
        _vel.push_back(dvec3(dp[i].vel[0], dp[i].vel[1], dp[i].vel[2]));
        _mass.push_back(dp[i].mass);
      }
    }

    int get_nstar() const { return nstar; }
    int get_ndark() const { return ndark; }
    int get_ntot () const {return nstar + ndark;}
    const dvec3&  pos (const int i) const { return _pos[i]; }
    const dvec3&  vel (const int i) const { return _vel[i]; }
    const double& mass(const int i) const { return _mass[i]; }
};



#ifdef GALACTICS
  #include "galactics.h"

  void generateGalacticsModel(const int      procId,
                              const int      nProcs,
                              const int      randomSeed,
                              const int      nMilkyWay,
                              const int      nMWfork,
                              const bool     scaleMass,
                              vector<real4>  &bodyPositions,
                              vector<real4>  &bodyVelocities,
                              vector<ullong> &bodyIDs)
  {
    if (procId == 0) printf("Using MilkyWay model with n= %d per proc, forked %d times \n", nMilkyWay, nMWfork);
    assert(nMilkyWay > 0);
    assert(nMWfork > 0);

    //Verify that all required files are available
    const char* fileList[] = {"cordbh.dat", "dbh.dat", "freqdbh.dat", "mr.dat",
                              "denspsibulge.dat", "denspsihalo.dat", "component_numbers.txt"};
    const int nFiles       = sizeof(fileList) / sizeof(fileList[0]);

    for(int i=0; i < nFiles; i++)
    {
      ifstream ifile(fileList[i]);
      if (!ifile) {
        fprintf(stderr,"Can not find the required input file: %s \n", fileList[i]);
        ::exit(-1);
      }
    }

    //Read in the particle ratios
    int nHalo, nBulge,nDisk;
    ifstream ifile("component_numbers.txt");
    std::string line;
    std::getline(ifile, line);
    sscanf(line.c_str(),"%d %d %d\n", &nHalo, &nBulge, &nDisk);

    fprintf(stderr,"Particle numbers from config file: %d %d %d \n", nHalo, nBulge, nDisk);
    ifile.close();

//    #if 1 /* in this setup all particles will be of equal mass (exact number are galactic-depednant)  */
//      const float fdisk  = 15.1;
//      const float fbulge = 5.1;
//      const float fhalo  = 242.31;
//    #else  /* here, bulge & mw particles have the same mass, but halo particles is 32x heavier */
//      const float fdisk  = 15.1;
//      const float fbulge = 5.1;
//      const float fhalo  = 7.5;
//    #endif
//    const float fsum = fdisk + fhalo + fbulge;

    size_t nMilkyWay2 = nMilkyWay;
    const double fsum = (float)(nHalo + nBulge + nDisk);
    int ndisk  = (int)  (nMilkyWay2 * nDisk /fsum);
    int nbulge = (int)  (nMilkyWay2 * nBulge/fsum);
    int nhalo  = (int)  (nMilkyWay2 * nHalo /fsum);

    assert(ndisk  > 0);
    assert(nbulge > 0);
    assert(nhalo  > 0);

    ndisk = max(1, ndisk);
    nbulge = max(1, nbulge);
    nhalo = max(1, nhalo);



    if (procId == 0)
      fprintf(stderr,"Requested numbers: seed= %d  ndisk= %d  nbulge= %d  nhalo= %d :: ntotal= %d\n",
                      randomSeed, ndisk, nbulge, nhalo, ndisk+nbulge+nhalo);

    const Galactics g(procId, nProcs, randomSeed, ndisk, nbulge, nhalo, nMWfork);
    if (procId == 0)
     printf("Generated numbers:  ndisk= %d  nbulge= %d  nhalo= %d :: ntotal= %d\n",
             g.get_ndisk(), g.get_nbulge(), g.get_nhalo(), g.get_ntot());

    const int ntot = g.get_ntot();
    bodyPositions.resize(ntot);
    bodyVelocities.resize(ntot);
    bodyIDs.resize(ntot);

    //Generate unique 64bit IDs, counter starts at individual boundaries
    //Note that we get 32bit IDs back from the Galactics code
    unsigned long long diskID  = ((unsigned long long) ndisk *procId) + DISKID;
    unsigned long long bulgeID = ((unsigned long long) nbulge*procId) + BULGEID;
    unsigned long long haloID  = ((unsigned long long) nhalo *procId) + DARKMATTERID;

    for (int i= 0; i < ntot; i++)
    {
      assert(!std::isnan(g[i].x));
      assert(!std::isnan(g[i].y));
      assert(!std::isnan(g[i].z));
      assert(g[i].mass > 0.0);

      //Generate unique IDS for each particle in the full model
      if( g[i].id >= 200000000)                             //Dark matter
        bodyIDs[i] = haloID++;
      else if( g[i].id >= 100000000 && g[i].id < 200000000) //Bulge
        bodyIDs[i] = bulgeID++;
      else                                                  //Disk
        bodyIDs[i] = diskID++;

      bodyPositions[i].x = g[i].x;
      bodyPositions[i].y = g[i].y;
      bodyPositions[i].z = g[i].z;
      if(scaleMass)
       bodyPositions[i].w = g[i].mass * 1.0/(double)nProcs;
      else
       bodyPositions[i].w = g[i].mass; // * 1.0/(double)nProcs ,scaled later ..

      assert(!std::isnan(g[i].vx));
      assert(!std::isnan(g[i].vy));
      assert(!std::isnan(g[i].vz));

      bodyVelocities[i].x = g[i].vx;
      bodyVelocities[i].y = g[i].vy;
      bodyVelocities[i].z = g[i].vz;
      bodyVelocities[i].w = 0.0;
    }
  } //generateGalacticsModel
#endif


  /*
   * Generate a Plummer model with mass scaled over the number of processes
   * uses the Plummer class from plummer.h
   */
  void generatePlummerModel(vector<real4>   &bodyPositions,
                            vector<real4>   &bodyVelocities,
                            vector<ullong>  &bodyIDs,
                            const int        procId,
                            const int        nProcs,
                            const int        nPlummer)
  {
    if (procId == 0) printf("Using Plummer model with n= %d per process \n", nPlummer);
    assert(nPlummer > 0);
    const int seed = 19810614 + procId;
    const Plummer m(nPlummer, procId, seed);
    bodyPositions.resize(nPlummer);
    bodyVelocities.resize(nPlummer);
    bodyIDs.resize(nPlummer);
    for (int i= 0; i < nPlummer; i++)
    {
      assert(!std::isnan(m.pos[i].x));
      assert(!std::isnan(m.pos[i].y));
      assert(!std::isnan(m.pos[i].z));
      assert(m.mass[i] > 0.0);
      bodyIDs[i]   = ((unsigned long long) nPlummer)*procId + i;

      bodyPositions[i].x = m.pos[i].x;
      bodyPositions[i].y = m.pos[i].y;
      bodyPositions[i].z = m.pos[i].z;
      bodyPositions[i].w = m.mass[i] * 1.0/nProcs;

      bodyVelocities[i].x = m.vel[i].x;
      bodyVelocities[i].y = m.vel[i].y;
      bodyVelocities[i].z = m.vel[i].z;
      bodyVelocities[i].w = 0;
    }
  }

  /*
   * Generate a spherical model with mass scaled over the number of processes
   */
  void generateSphereModel(vector<real4>    &bodyPositions,
                            vector<real4>   &bodyVelocities,
                            vector<ullong>  &bodyIDs,
                            const int        procId,
                            const int        nProcs,
                            const int        nSphere)
  {
    //Sphere
    if (procId == 0) printf("Using Spherical model with n= %d per process \n", nSphere);
    assert(nSphere >= 0);
    bodyPositions.resize(nSphere);
    bodyVelocities.resize(nSphere);
    bodyIDs.resize(nSphere);

    srand48(procId+19840501);

    /* generate uniform sphere */
    int np = 0;
    while (np < nSphere)
    {
      const double x = 2.0*drand48()-1.0;
      const double y = 2.0*drand48()-1.0;
      const double z = 2.0*drand48()-1.0;
      const double r2 = x*x+y*y+z*z;
      if (r2 < 1)
      {
        bodyIDs[np]   = ((unsigned long long) nSphere)*procId + np;

        bodyPositions[np].x = x;
        bodyPositions[np].y = y;
        bodyPositions[np].z = z;
        bodyPositions[np].w = (1.0/nSphere) * 1.0/nProcs;

        bodyVelocities[np].x = 0;
        bodyVelocities[np].y = 0;
        bodyVelocities[np].z = 0;
        bodyVelocities[np].w = 0;
        np++;
      }//if
    }//while
  }

  /*
   * Generate a unit cube model with mass scaled over the number of processes
   */
  void generateCubeModel(vector<real4>   &bodyPositions,
                         vector<real4>   &bodyVelocities,
                         vector<ullong>  &bodyIDs,
                         const int        procId,
                         const int        nProcs,
                         const int        nCube)
  {
    //Cube
    if (procId == 0) printf("Using Cube model with n= %d per process \n", nCube);
    assert(nCube >= 0);
    bodyPositions.resize(nCube);
    bodyVelocities.resize(nCube);
    bodyIDs.resize(nCube);

    srand48(procId+19840501);

    /* generate uniform cube */
    for (int i= 0; i < nCube; i++)
    {
     const double x = 2*drand48()-1.0;
     const double y = 2*drand48()-1.0;
     const double z = 2*drand48()-1.0;

     bodyIDs[i]   =  ((unsigned long long) nCube)*procId + i;

     bodyPositions[i].x = x;
     bodyPositions[i].y = y;
     bodyPositions[i].z = z;
     bodyPositions[i].w = (1.0/nCube) * 1.0/nCube;

     bodyVelocities[i].x = 0;
     bodyVelocities[i].y = 0;
     bodyVelocities[i].z = 0;
     bodyVelocities[i].w = 0;
    }
  }

  /*
   *
   * Combine a single model nProc times by shuffling the particles a tiny fraction on
   * each process.
   */
  void generateShuffledDiskModel(vector<real4>   &bodyPositions,
                         vector<real4>   &bodyVelocities,
                         vector<ullong>  &bodyIDs,
                         const int        procId,
                         const int        nProcs,
                         const std::string &fileName)
  {
    if (procId == 0) printf("Using disk mode with filename %s\n", fileName.c_str());
      const int seed = procId+19840501;
      srand48(seed);
      const DiskShuffle disk(fileName);
      const int np = disk.get_ntot();
      bodyPositions.resize(np);
      bodyVelocities.resize(np);
      bodyIDs.resize(np);
      for (int i= 0; i < np; i++)
      {
        bodyIDs[i]   =  ((unsigned long long) np)*procId + i;

        bodyPositions[i].x = disk.pos(i).x;
        bodyPositions[i].y = disk.pos (i).y;
        bodyPositions[i].z = disk.pos (i).z;
        bodyPositions[i].w = disk.mass(i) * 1.0/nProcs;

        bodyVelocities[i].x = disk.vel(i).x;
        bodyVelocities[i].y = disk.vel(i).y;
        bodyVelocities[i].z = disk.vel(i).z;
        bodyVelocities[i].w = 0;
      }
  }
