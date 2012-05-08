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
#include "anyoption.h"

#include "tipsydefs.h"

float float_in(const float a, const float b)
{
  return ((b-a)*((float)rand()/RAND_MAX))+a;
}


int main(int argc, char **argv)
{

  AnyOption *opt = new AnyOption();
        
  opt->addUsage( "" );
  opt->addUsage( "Usage: " );
  opt->addUsage( "" );
  opt->addUsage( " -h  --help  		        Prints this help " );
  opt->addUsage( " -i  --infile           Input filename ");
  opt->addUsage( " -o  --outfile          Output filename ");
  opt->addUsage( " -N  --Ndust            number of dust particles ");
  opt->addUsage( " -R  --dRshift 0.0      shift from the disk centre in units of Rscale ");
  opt->addUsage( " -D  --Rscale   0.05    Radial   scale height in units of disk radial   extent ");
  opt->addUsage( " -H  --Zscale   0.1     Vertical scale height in units of disk vertical extent ");
  opt->addUsage( " -I  --incl     0.0     inclination of the dust ring wrt to the disk in degrees ");
  opt->addUsage( " -P  --phi      90.0    angle to the rotation axis form X-axis in degrees ");
  opt->addUsage( " -r  --nrScale  3.0     Radial   scale height ");
  opt->addUsage( " -z  --nzScale  3.0     Vertical scale height ");
  opt->addUsage( " -T  --torus            Enable TORUS dust ring instead of CYLINDER");
  opt->addUsage( " -G  --Glow    0       Enable glowing, Nglow = Ndust/Glow, if Glow = 0, no glowing is added");
  opt->addUsage( "" );
        
  opt->setFlag  (  "help", 'h' );     /* a flag (takes no argument), supporting long and short form */ 
  opt->setOption(  "infile",  'i'); /* an option (takes an argument), supporting long and short form */
  opt->setOption(  "outfile", 'o'); 
  opt->setOption(  "Ndust" , 'N');
  opt->setOption(  "dRshit", 'R');
  opt->setOption(  "Rscale", 'D');
  opt->setOption(  "Zscale", 'H');
  opt->setOption(  "nrScale", 'r');
  opt->setOption(  "nzScale", 'z');
  opt->setOption(  "incl", 'I');
  opt->setOption(  "phi", 'P');
  opt->setOption(  "Glow", 'G');
  opt->setFlag  ( "torus", 'T');

  /* for options that will be checked only on the command and line not in option/resource file */
  opt->setCommandFlag(  "zip" , 'z'); /* a flag (takes no argument), supporting long and short form */

  /* for options that will be checked only from the option/resource file */
  opt->setFileOption(  "title" ); /* an option (takes an argument), supporting only long form */

  opt->processCommandArgs( argc, argv );

  if( ! opt->hasOptions()) { /* print usage if no options */
    opt->printUsage();
    delete opt;
    exit(0);
  }
  
  if( opt->getFlag( "help" ) || opt->getFlag( 'h' ) ) 
    opt->printUsage();
  
  char *optarg = NULL;

  FILE *rv1 = NULL;
  FILE *outfile = NULL;
  if((optarg = opt->getValue('i')))
    if( !(rv1 = fopen(optarg,"rb")) ) 
    {
      fprintf(stderr,"Can't find first file  %s\n", optarg);
      exit(0);
    }
          
  if((optarg = opt->getValue('o')))
    if( !(outfile = fopen(optarg,"wb")) ) 
    {
      fprintf(stderr,"Can't open output file %s\n", optarg);
      exit(0);
    }


  int Ndust = 0, Glow = 0;
  real dRshift = 0.0;
  real Rscale   = 0.05;
  real Zscale   = 0.1;
  real nrScale  = 3.0;
  real nzScale  = 3.0;
  real inclination = 0.0;
  real phi = 90.0;
  DustRing::RingType ring_type = DustRing::CYLINDER;

  if ((optarg = opt->getValue('N'))) Ndust = atoi(optarg);
  if ((optarg = opt->getValue('R'))) dRshift  = atof(optarg);
  if ((optarg = opt->getValue('D'))) Rscale   = atof(optarg);
  if ((optarg = opt->getValue('H'))) Zscale   = atof(optarg);
  if ((optarg = opt->getValue('r'))) nrScale  = atof(optarg);
  if ((optarg = opt->getValue('z'))) nzScale  = atof(optarg);
  if ((optarg = opt->getValue('I'))) inclination = atof(optarg);
  if ((optarg = opt->getValue('P'))) phi = atof(optarg);
  if ((optarg = opt->getValue('G'))) Glow = atoi(optarg);
  if (opt->getFlag('T')) ring_type = DustRing::TORUS;

  
  if(Ndust == 0 || rv1 == NULL || outfile == NULL)
  {
    opt->printUsage();
    delete opt;
    exit(0);
  }

  delete opt;

  fprintf(stderr, " Adding %s dust ring: \n", ring_type == DustRing::CYLINDER ? "CYLINDER" : "TORUS");
  fprintf(stderr, "   N=       %d \n", Ndust);
  fprintf(stderr, "   dRshift= %g \n", dRshift);
  fprintf(stderr, "   Rscale=  %g \n", Rscale);
  fprintf(stderr, "   Zscale=  %g \n", Zscale);
  fprintf(stderr, "   incl=    %g  degrees \n", inclination);
  fprintf(stderr, "   phi=     %g  degrees \n", phi);
  fprintf(stderr, "   nrScale= %g \n", nrScale);
  fprintf(stderr, "   nzScale= %g \n", nzScale);
  
  inclination *= M_PI/180.0;
  phi         *= M_PI/180.0;


  std::vector<dark_particle> darkMatter;
  std::vector<star_particle> bulge;
  std::vector<star_particle> disk;
  std::vector<star_particle> dust;

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
#if 0
  double massGalaxy1 = 0;	
  double massGalaxy2 = 0;
#endif
  for(int i=0; i < NHalo1; i++)
  {
    fread(&d, sizeof(d), 1, rv1);
//    const int pID = (int) d.phi;

    darkMatter.push_back(d);	 
  }

  struct star_particle s;

  unsigned int maxDustID = 50000000-1; //-1 since we do +1 later on
  unsigned int maxMasslessGlowID = 70000000-1; //-1 since we do +1 later on
  unsigned int maxMassiveGlowID = 40000000-1; //-1 since we do +1 later on

  for(int i=0; i < NBulge1; i++)
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
  if (Ndisk == 0)
  {
    fprintf(stderr, " FATAL: No disk particles found... Please check your input file\n");
    exit(0);
  }
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

  const Real dR = (Rmax - Rmin)*0.5;
  const Real D  = Rscale*dR;
  const Real Ro = (Rmax + Rmin)*0.5 + dRshift * D;

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

  const Real H = Zscale*dZ;

  /** Generating dust ring **/

  const DustRing ring (Ndust, Ro, D, H, inclination, VelCurve, phi, nrScale, nzScale, ring_type);

  /** Adding dust ring **/


  unsigned int dustID = maxDustID+1;
  //   fprintf(stdout, "%d\n", Ndust);
  for(int i=0; i < Ndust; i++)
  {
    s.mass = 0;
    s.pos[0] = ring.ptcl[i].pos.x;
    s.pos[1] = ring.ptcl[i].pos.y;
    s.pos[2] = ring.ptcl[i].pos.z;
    s.vel[0] = ring.ptcl[i].vel.x;
    s.vel[1] = ring.ptcl[i].vel.y;
    s.vel[2] = ring.ptcl[i].vel.z;
    s.phi = dustID++;

#if 0
    fprintf(stdout, "%g %g %g \n", s.pos[0], s.pos[1], s.pos[2]);
#endif

#if 0
    const real R = std::sqrt(s.pos[0]*s.pos[0] + s.pos[1]*s.pos[1]);
    const real V = std::sqrt(s.vel[0]*s.vel[0] + s.vel[1]*s.vel[1]);
    fprintf(stderr," R= %g  V= %g \n", R, V);
#endif

    dust.push_back(s);
  }
	
  /** Adding ring of glowing particles **/

	if (Glow > 0)
	{
		const int Nglow = Ndisk / Glow;   /* use less particles */
		const Real D1   = D/3;           /* make ring less wider */
		const Real H1   = H/3;           /*          and thinner */
		const DustRing glow(Nglow, Ro, D1, H1, 0, VelCurve, phi, nrScale, nzScale, ring_type);

#if 0  /* uncomment this if you want massless glowing particles */
		unsigned int glowID = maxMasslessGlowID+1;
		for(int i=0; i < Nglow; i++)
		{
			s.mass = 1.0e-12;
			s.pos[0] = glow.ptcl[i].pos.x;
			s.pos[1] = glow.ptcl[i].pos.y;
			s.pos[2] = glow.ptcl[i].pos.z;
			s.vel[0] = glow.ptcl[i].vel.x;
			s.vel[1] = glow.ptcl[i].vel.y;
			s.vel[2] = glow.ptcl[i].vel.z;
			s.phi = glowID++;

			dust.push_back(s);
		}
#else /* otherwise they will have mass equal to the disk star particles */
		unsigned int glowID = maxMassiveGlowID+1;
		for(int i=0; i < Nglow; i++)
		{
			s.mass = disk[0].mass;
			s.pos[0] = glow.ptcl[i].pos.x;
			s.pos[1] = glow.ptcl[i].pos.y;
			s.pos[2] = glow.ptcl[i].pos.z;
			s.vel[0] = glow.ptcl[i].vel.x;
			s.vel[1] = glow.ptcl[i].vel.y;
			s.vel[2] = glow.ptcl[i].vel.z;
			s.phi = glowID++;

			dust.push_back(s);
		}
#endif
	}

	//End dust magic

	//         dust.clear();
	//         disk.clear();


	//Write the data

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
	for(int i=0; i < NHalo1; i++)
	{
		d = darkMatter[i];
		fwrite(&d, sizeof(d), 1, outfile);
	}

	//Now write the star particles
	for(int i=0; i < (int)bulge.size(); i++)
	{
		s = bulge[i];          
		fwrite(&s, sizeof(s), 1, outfile);
	}
	//Now write the disk particles
	for(int i=0; i < (int)disk.size(); i++)
	{
		s = disk[i];          
		fwrite(&s, sizeof(s), 1, outfile);
	}    
	//Now write the dust particles
	for(int i=0; i < (int)dust.size(); i++)
	{
		s = dust[i];          
		fwrite(&s, sizeof(s), 1, outfile);
	}          

	fprintf(stderr,"Wrote: Total: %d DM: %d Disk: %d Bulge: %d Dust: %d \n",
			NTotal1, (int)darkMatter.size(),
			(int)disk.size(), (int)bulge.size(), (int)dust.size());        

	return 0;
}

