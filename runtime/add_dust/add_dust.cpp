/* Puts two galaxies on a specified orbit
 * 
 * Originally by John Dubinski
 * 
 * modified by Jeroen Bedorf to read in
 * different galaxies and be self-contained 
 * into one file and read in the modified tipsy
 * file format.
 * 
 * */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <math.h>
#include "DustRing.h"

#include "tipsydefs.h"

float float_in(const float a, const float b)
{
  return ((b-a)*((float)rand()/RAND_MAX))+a;
}

struct Rotation
{
  real Axx, Axy, Axz;
  real Ayx, Ayy, Ayz;
  real Azx, Azy, Azz;
  Rotation() {}

  friend vec3 operator << (const Rotation &M, const vec3 &v)
  {
    return vec3(
        M.Axx*v.x + M.Axy*v.y + M.Axz*v.z,
        M.Ayx*v.x + M.Ayy*v.y + M.Ayz*v.z,
        M.Azx*v.x + M.Azy*v.y + M.Azz*v.z);
  }
};

int main(int argc, char **argv)
{
  int i;

  FILE *rv1;

  std::vector<dark_particle> darkMatter;
  std::vector<star_particle> bulge;
  std::vector<star_particle> disk;
  std::vector<star_particle> dust;

  if( argc == 3) {
    if( !(rv1 = fopen(argv[1],"rb")) ) {
      fprintf(stderr,"Can't find first file  %s\n",argv[1]);
      exit(0);
    }
  }
  else {
    fprintf(stderr,"usage: add_dust inFile outfile\n");
    exit(0);
  }

  //Read the particle Data of the main galaxy
  //Read the header
  int NTotal1, NHalo1, NBulge1;

  //Read the header from the binary file
  struct dump h;
  fread(&h, sizeof(h), 1, rv1);

  fprintf(stderr, "First galaxy header %f %d %d %d %d %d \n", h.time, h.nbodies, h.ndim, h.nsph,
      h.ndark, h.nstar);

  NTotal1  = h.nbodies;
  NHalo1   = h.ndark;
  NBulge1  = h.nstar;


  struct dark_particle d;
  double massGalaxy1 = 0;	
  double massGalaxy2 = 0;
  for(i=0; i < NHalo1; i++)
  {
    fread(&d, sizeof(d), 1, rv1);
    const int pID = (int) d.phi;

    darkMatter.push_back(d);	 
  }

  struct star_particle s;

  unsigned int maxDustID = 50000000-1; //-1 since we do +1 later on

  for(i=0; i < NBulge1; i++)
  {
    fread(&s, sizeof(s), 1, rv1);
    int pID = (int) s.phi;

    //           fprintf(stderr,"Pid: %d\t %d \n", i, pID);

    if(pID >= 100000000)
      bulge.push_back(s);
    else if(pID >= 50000000)
    {
      dust.push_back(s);
      maxDustID = std::max((int)maxDustID, pID);
    }
    else
      disk.push_back(s);
  }

  fprintf(stderr,"Read: Total: %d DM: %d Disk: %d Bulge: %d Dust: %d \n",
      NTotal1, (int)darkMatter.size(),
      (int)disk.size(), (int)bulge.size(), (int)dust.size());


  //Dust magic comes here
  //Dust particles get IDs starting at:
  //50.000.000
  //
  const int Ndisk = disk.size();
  Vel1D::Vector VelCurve;
  VelCurve.reserve(Ndisk);
  vec3 L(0.0);
  Real Mtot = 0.0;
  Real Rmin = HUGE;
  Real Rmax = 0.0;

  for (int i = 0; i < Ndisk; i++)
  {
    const star_particle &p = disk[i];
    const vec3 pos(p.pos);
    const vec3 vel(p.vel);
    const Real V = std::sqrt(vel.x*vel.x + vel.y*vel.y);
    if(0.01*V > std::abs(vel.z))
    {
      L    += p.mass * (pos%vel);
      Mtot += p.mass;
      const Real R = std::sqrt(pos.x*pos.x + pos.y*pos.y);
#if 0
      fprintf(stderr," R= %g  V= %g \n", R, V);
#endif
      VelCurve.push_back(Vel1D(R, V));
      Rmin = std::min(Rmin, R);
      Rmax = std::max(Rmax, R);
    }
  }
  L *= 1.0/Mtot;
  fprintf(stderr, " Ncurve= %d :: L= %g %g %g \n", (int)VelCurve.size(), L.x, L.y, L.z);
  fprintf(stderr, "  Rmin= %g  Rmax= %g \n", Rmin, Rmax);
 
#if 0   /* this is to rotate disk into x-y plane, if the disk is not in x-y plane */
  {
    const vec3 e1(L.x, L.y, 0.0);
    const vec3 e2(e1%L);

    const Real costh = L.z/L.abs();
    assert(costh <= 1.0);
    const Real sinth  = std::sqrt(1.0 - costh*costh);
    const Real costhm = 1.0 - costh;

    const vec3 u(e2 * (1.0/e2.abs()));

    Rotation M;
    M.Axx = u.x*u.x*costhm +     costh;
    M.Axy = u.x*u.y*costhm - u.z*sinth;
    M.Axz = u.x*u.z*costhm + u.y*sinth;

    M.Ayx = u.y*u.x*costhm + u.z*sinth;
    M.Ayy = u.y*u.y*costhm +     costh;
    M.Ayz = u.y*u.z*costhm - u.x*sinth;

    M.Azx = u.z*u.x*costhm - u.y*sinth;
    M.Azy = u.z*u.y*costhm + u.x*sinth;
    M.Azz = u.z*u.z*costhm +     costh;

    VelCurve.clear(); 
    VelCurve.reserve(Ndisk);
    vec3 L(0.0);
    real Mtot = 0.0;
    for (int i = 0; i < Ndisk; i++)
    {
      const star_particle &p = disk[i];
      const vec3 pos(M << vec3(p.pos));
      const vec3 vel(M << vec3(p.vel));
      L    += p.mass * (pos%vel);
      Mtot += p.mass;
      VelCurve.push_back(Vel1D(std::sqrt(pos.x*pos.y + pos.y*pos.y), vel.abs()));
    }
    L *= 1.0/Mtot;
    fprintf(stderr, " Ncurve= %d :: L= %g %g %g \n", (int)VelCurve.size(), L.x, L.y, L.z);
  }
#endif

  /* setting radial scale height of the dust ring */
  
  const Real Ro = (Rmax + Rmin)*0.5;
  const Real dR = (Rmax - Rmin)*0.5;
  const Real D  = 0.05*dR;
  const Real nrScale = 3.0;

  /* determining vertical scale-height of the disk */
  Real Zmin = HUGE;
  Real Zmax = 0.0;
  for (int i = 0; i < Ndisk; i++)
  {
    const star_particle &p = disk[i];
    const vec3 pos(p.pos);
    const Real R = std::sqrt(pos.x*pos.x + pos.y*pos.y);
    if(R > Ro - nrScale*D && R < Ro + nrScale*D)
    {
      Zmin = std::min(Zmin, pos.z);
      Zmax = std::max(Zmax, pos.z);
    }
  }
  const real dZ = Zmax - Zmin;
  fprintf(stderr, "Zmin= %g Zmax= %g \n", Zmin, Zmax);

  /* setting vertical scale height of the dust ring */

  const Real H = 0.1*dZ;
  const Real nzScale = 3.0;

  /** Generating dust ring **/

  const int Ndust = 30000;
#if 1
  const DustRing ring(Ndust, Ro, D, H, VelCurve, nrScale, nzScale);
#else
  const DustRing ring(Ndust, Ro, D, H, VelCurve, nrScale, nzScale, DustRing::TORUS);
#endif

  /** Adding dust ring **/
  

  unsigned int dustID = maxDustID+1;
  fprintf(stdout, "%d\n", Ndust);
  for(int i=0; i < Ndust; i++)
  {
    s.mass = 0;
    s.pos[0] = ring.ptcl[i].pos.x;
    s.pos[1] = ring.ptcl[i].pos.y;
    s.pos[2] = ring.ptcl[i].pos.z;
    s.vel[0] = ring.ptcl[i].vel.y;
    s.vel[1] = ring.ptcl[i].vel.y;
    s.vel[2] = ring.ptcl[i].vel.z;
    s.phi = dustID++;

    fprintf(stdout, "%g %g %g \n", s.pos[0], s.pos[1], s.pos[2]);

#if 0
    const real R = std::sqrt(s.pos[0]*s.pos[0] + s.pos[1]*s.pos[1]);
    const real V = std::sqrt(s.vel[0]*s.vel[0] + s.vel[1]*s.vel[1]);
    fprintf(stderr," R= %g  V= %g \n", R, V);
#endif

    dust.push_back(s);
  }

  //End dust magic

  //         dust.clear();
  //         disk.clear();


  //Write the data
  FILE *outfile;
  if( !(outfile = fopen(argv[2],"wb")) ) {
    fprintf(stderr,"Can't open output file %s\n",argv[2]);
    exit(0);
  }

  NTotal1 = (int)darkMatter.size() + (int)disk.size() + (int)bulge.size() + (int)dust.size();
  int NStar = (int)disk.size() + (int)bulge.size() + (int)dust.size();
  //Write tipsy header
  h.nbodies = NTotal1;
  h.ndark   = (int)darkMatter.size();
  h.nstar   = NStar;
  h.nsph    = 0;
  h.ndim    = 3;
  h.time    = 0;

  fwrite(&h, sizeof(h), 1, outfile);
  //First write DM Halo of main galaxy
  for(i=0; i < NHalo1; i++)
  {
    d = darkMatter[i];
    fwrite(&d, sizeof(d), 1, outfile);
  }

  //Now write the star particles
  for(i=0; i < bulge.size(); i++)
  {
    s = bulge[i];          
    fwrite(&s, sizeof(s), 1, outfile);
  }
  //Now write the disk particles
  for(i=0; i < disk.size(); i++)
  {
    s = disk[i];          
    fwrite(&s, sizeof(s), 1, outfile);
  }    
  //Now write the dust particles
  for(i=0; i < dust.size(); i++)
  {
    s = dust[i];          
    fwrite(&s, sizeof(s), 1, outfile);
  }          

  fprintf(stderr,"Wrote: Total: %d DM: %d Disk: %d Bulge: %d Dust: %d \n",
      NTotal1, (int)darkMatter.size(),
      (int)disk.size(), (int)bulge.size(), (int)dust.size());        


  return 0;
}

