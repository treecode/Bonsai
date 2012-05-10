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
#include <string>
#include <sstream>
#include <cmath>
#include <cassert>
#include <vector>
#include "vector3.h"
#include "anyoption.h"

#include "tipsydefs.h"

typedef float real;
typedef vector3<real> vec3;

struct Particle
{
	typedef std::vector<Particle> Vector;
	typedef Vector::      iterator      Iterator;
	typedef Vector::const_iterator constIterator;
	int ID;
	real m;
	vec3 pos;
	vec3 vel;
	real eps;
};

template<class T>
bool readf(T &data, FILE *fin, const int n = 1)
{
	const size_t sz = fread(&data, sizeof(T), n, fin);
	return sz == sizeof(T);
}

template<class T>
bool writef(T &data, FILE *fout, const int n = 1)
{
	const size_t sz = fwrite(&data, sizeof(T), n, fout);
	return  sz == sizeof(T);
}

inline vec3 cvt2xyz(const real R, const real l, const real b)
{
	return vec3(std::sin(b), std::cos(b)*std::cos(l), std::cos(b)*std::sin(l))*R;
}

struct Rotation
{
  real xx, xy, xz;
  real yx, yy, yz;
  real zx, zy, zz;
  Rotation(const real theta = 0.0, const vec3 axis = 0.0)
  {
    const real costh = cos(theta);
    const real sinth = sin(theta);
    const real costhm = 1.0 - costh;
		const vec3 &u = axis;

    xx = u.x*u.x*costhm +     costh;
    xy = u.x*u.y*costhm - u.z*sinth;
    xz = u.x*u.z*costhm + u.y*sinth;

    yx = u.y*u.x*costhm + u.z*sinth;
    yy = u.y*u.y*costhm +     costh;
    yz = u.y*u.z*costhm - u.x*sinth;

    zx = u.z*u.x*costhm - u.y*sinth;
    zy = u.z*u.y*costhm + u.x*sinth;
    zz = u.z*u.z*costhm +     costh;
  }

	friend Rotation operator*(const Rotation &A, const Rotation &B)
	{
		Rotation C;
		C.xx = A.xx*B.xx + A.xy*B.yx + A.xz*B.zx;
		C.xy = A.xx*B.xy + A.xy*B.yy + A.xz*B.zy;
		C.xz = A.xx*B.xz + A.xy*B.yz + A.xz*B.zz;
		
		C.yx = A.yx*B.xx + A.yy*B.yx + A.yz*B.zx;
		C.yy = A.yx*B.xy + A.yy*B.yy + A.yz*B.zy;
		C.yz = A.yx*B.xz + A.yy*B.yz + A.yz*B.zz;
		
		C.zx = A.zx*B.xx + A.zy*B.yx + A.zz*B.zx;
		C.zy = A.zx*B.xy + A.zy*B.yy + A.zz*B.zy;
		C.zz = A.zx*B.xz + A.zy*B.yz + A.zz*B.zz;

		return C;
	}

	friend vec3 operator*(const Rotation &A,  const vec3 &v)
	{
    return vec3(
        A.xx*v.x + A.xy*v.y + A.xz*v.z,
        A.yx*v.x + A.yy*v.y + A.yz*v.z,
        A.zx*v.x + A.zy*v.y + A.zz*v.z);
	}
};

struct Galactic : public Rotation
{
  real l, b;
	Galactic(const real _l, const real _b) : l(_l), b(_b)
	{
		Rotation A(l, vec3(0.0, 0.0, 1.0));
		Rotation B(b, vec3(sin(l), -cos(l), 0.0));

		*this = B*A;
	}
	Galactic(const Rotation &A)
	{
		xx = A.xx;
		xy = A.xy;
		xz = A.xz;
		
		yx = A.yx;
		yy = A.yy;
		yz = A.yz;
		
		zx = A.zx;
		zy = A.zy;
		zz = A.zz;
	}
	
	friend vec3 operator*(const Galactic &A,  const vec3 &v)
	{
    return vec3(
        A.xx*v.x + A.xy*v.y + A.xz*v.z,
        A.yx*v.x + A.yy*v.y + A.yz*v.z,
        A.zx*v.x + A.zy*v.y + A.zz*v.z);
	}
};

inline double to_rad(const double deg)
{
	return deg * M_PI/180.0;
}

inline double to_deg(const double rad)
{
	return rad * 180.0/M_PI;
}

real centerGalaxy(Particle::Vector &ptcl)
{

	real M = 0.0;
	vec3 cm_pos(0.0), cm_vel(0.0);

	for (Particle::constIterator it = ptcl.begin(); it != ptcl.end(); it++)
	{
		M      += it->m;
		cm_pos += it->m * it->pos;
		cm_vel += it->m * it->vel;
	}

	assert (M > 0.0);
	const real iM = 1.0/M;
	cm_pos *= iM;
	cm_vel *= iM;

	for (Particle::Iterator it = ptcl.begin(); it != ptcl.end(); it++)
	{
		it->pos -= cm_pos;
		it->vel -= cm_vel;
	}

	return M;
}

real readGalaxy(dump &h, Particle::Vector &ptcl, FILE *fin, 
		const real massRatio = 1.0,
		const real sizeRatio = 1.0)
{
	readf(h, fin);
	ptcl.resize(h.nbodies);

	for(int i = 0; i < h.ndark; i++)
	{
		dark_particle d;
		readf(d, fin);
		Particle &p = ptcl[i];
		p.ID  = (int) d.phi;
		p.m   = d.mass;
		p.pos = vec3(d.pos[0], d.pos[1], d.pos[2]);
		p.vel = vec3(d.vel[0], d.vel[1], d.vel[2]);
		p.eps = d.eps;
	}
	
	for(int i = 0; i < h.nstar; i++)
	{
		star_particle s;
		readf(s, fin);
		Particle &p = ptcl[h.ndark + i];
		p.ID  = (int)s.phi;
		p.m   = s.mass;
		p.pos = vec3(s.pos[0], s.pos[1], s.pos[2]);
		p.vel = vec3(s.vel[0], s.vel[1], s.vel[2]);
		p.eps = s.eps;
	}

	real M = centerGalaxy(ptcl);

	const real velRatio = std::sqrt(massRatio/sizeRatio);
	for (Particle::Iterator it = ptcl.begin(); it != ptcl.end(); it++)
	{
		it->m   *= massRatio;
		it->pos *= sizeRatio;
		it->vel *=  velRatio;
	}

	return M * massRatio;
}

void writeGalaxy(dump &h, Particle::Vector &ptcl, FILE *fout)
{
	assert((int)ptcl.size() == h.nbodies);

	writef(h, fout);
	for (int i = 0; i < h.ndark; i++)
	{
		assert(i < h.nbodies);
		dark_particle d;
		const Particle &p = ptcl[i];
		d.mass   = p.m;
		d.pos[0] = p.pos.x;
		d.pos[1] = p.pos.y;
		d.pos[2] = p.pos.z;
		d.vel[0] = p.vel.x;
		d.vel[1] = p.vel.y;
		d.vel[2] = p.vel.z;
		d.phi    = p.ID;
		d.eps    = p.eps;
		writef(d, fout);
	}

	for (int i = h.ndark; i < h.nbodies; i++)
	{
		assert(i < h.nbodies);
		star_particle s;
		const Particle &p = ptcl[i];
		s.mass   = p.m;
		s.pos[0] = p.pos.x;
		s.pos[1] = p.pos.y;
		s.pos[2] = p.pos.z;
		s.vel[0] = p.vel.x;
		s.vel[1] = p.vel.y;
		s.vel[2] = p.vel.z;
		s.phi    = p.ID;
		s.eps    = p.eps;
		writef(s, fout);
	}

}

int main(int argc, char **argv)
{

	std::string mw_fname, m31_fname, out_fname;
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help         Prints this help ");
		ADDUSAGE("     --mw  #        Input  filename for the Milky Way galaxy ");
		ADDUSAGE("     --m31 #        Input  filename for the Andromeda galaxy");
		ADDUSAGE("     --out #        Output filename for the merger IC");
		ADDUSAGE(" ");

		opt.setFlag( "help" ,   'h');
		opt.setOption( "mw");
		opt.setOption( "m31");
		opt.setOption( "out");

		opt.processCommandArgs( argc, argv );


		if( ! opt.hasOptions()) { /* print usage if no options */
			opt.printUsage();
			exit(0);
		}

		if( opt.getFlag( "help" ) || opt.getFlag( 'h' ) ) 
		{
			opt.printUsage();
			exit(0);
		}

		char *optarg = NULL;
		if ((optarg = opt.getValue("mw" )))    mw_fname  = std::string(optarg);
		if ((optarg = opt.getValue("m31")))    m31_fname = std::string(optarg);
		if ((optarg = opt.getValue("out")))    out_fname = std::string(optarg);

		if (mw_fname.empty() || m31_fname.empty() || out_fname.empty())
		{
			opt.printUsage();
			exit(0);
		}
	}

	const float Vunit = 100.0;    /* km/s */
	const float Runit = 1.0;      /* kpc  */

	const float Vr = -125.0/Vunit;
	const float Vt =  100.0/Vunit;  /* from Johan's thesis */
	const float Vt_phi = to_rad(45.0);
	const float lVt = to_rad(+180.0);  /* this is orientation of the tangential velocity */
	const float bVt = to_rad(   0.0);  /* p.18 in arXiv/astro-ph/9509010 */

	const float sizeRatio = 1.0; //300.0/200.0;  /* size(M31)/size(MW) */
	const float massRatio = 1.0;//  7.1/5.8;    /* mass(M31)/mass(MW) */

	const float Rsep      = 778.0/Runit;  
//	const float Rp        = 20.0/Runit;     /* page 18 in astro-ph/9509010 */
	const float lMW       = to_rad(0.0);
	const float bMW       = to_rad(-90.0);
	const float lM31      = to_rad(240.0);
	const float bM31      = to_rad(-30.0);
	const float lR        = to_rad(121.0);
	const float bR        = to_rad(-23.0);

	const Galactic rot1(lMW,  bMW);
	const Galactic rot2(lM31, bM31);

	const Galactic rotVr = Galactic(lR,  bR);
//	const Galactic rotVt = Galactic(lVt, bVt);  /* wrong */
	const vec3 Rij  = rotVr * vec3(Rsep, 0.0, 0.0);
	const vec3 vRij = rotVr * vec3(Vr,   0.0, 0.0);
	const vec3 vTij = rotVr * vec3(0.0,  Vt*cos(Vt_phi), Vt*sin(Vt_phi));
	const vec3 Vij  = vRij + vTij;
	fprintf(stderr, " R= %g  V= %g \n", Rij.abs(), Vij.abs());
	fprintf(stderr, " vR.R/R= %g  vT.R/R= %g \n", (vRij*Rij)/Rij.abs(), (vTij*Rij)/Rij.abs());
	fprintf(stderr, " V.R/R= %g  Vr= %g \n", (Vij*Rij)/Rij.abs(), Vr);

	FILE *fin1 = NULL;
	if( !(fin1 = fopen(mw_fname.c_str(),"rb")) ) 
	{
		fprintf(stderr,"Can't find Milky Way galaxy file  %s\n", mw_fname.c_str());
		exit(0);
	}
	FILE *fin2 = NULL;
	if( !(fin2 = fopen(m31_fname.c_str(),"rb")) ) 
	{
		fprintf(stderr,"Can't find Andromeda galaxy file %s\n", m31_fname.c_str());
		exit(0);
	}

	FILE *fout;
	if( !(fout = fopen(out_fname.c_str(),"wb")) ) 
	{
		fprintf(stderr,"Can't open output file %s\n", out_fname.c_str());
		exit(0);
	}

	dump h1, h2;
	Particle::Vector ptcl1, ptcl2;

	const real M1 = readGalaxy(h1, ptcl1, fin1);
	const real M2 = readGalaxy(h2, ptcl2, fin2, massRatio, sizeRatio);

	fclose(fin1);
	fclose(fin2);

	fprintf(stderr, "Galaxy 1: %f %d %d %d %d %d \n",
			h1.time, h1.nbodies, h1.ndim, h1.nsph,	h1.ndark, h1.nstar);
	fprintf(stderr, "Galaxy 2: %f %d %d %d %d %d \n",
			h2.time, h2.nbodies, h2.ndim, h2.nsph,	h2.ndark, h2.nstar);

	/* compute Rperi */
	{
		const real M  = M1 + M2;
		const real U  = M/Rij.abs();
		const real T  = Vij.norm2()*0.5;
		const real E  = T - U;   /* specific total energy in the CoM frame */
		if (E > 0.0)  /* hyperbolic orbit */
			fprintf(stderr, "the orbit is HYPERBOLIC: \n");
		else if (E < 0.0)  /* elliptic orbit */
			fprintf(stderr, "the orbit is ELLIPTIC: \n");
		else
		{
			fprintf(stderr, "PARABOLIC orbits are not supported, Change your parameters slightly..\n");
			exit(0);
		}

		const real L  =  (Rij%Vij).abs();
		const real A  =  L*L;
		const real B  =  M;
		const real C  = -2.0*E;
		const real D2 =  B*B - A*C;
		assert(D2 > 0.0);
		const real D   = std::sqrt(D2); 
		assert(B+D > 0.0);
		const real Rp  = std::sqrt(A/(B+D));
		fprintf(stderr,"  pericentre disance (Rp) : %g \n", Rp);

		const real Vff = std::sqrt(2.0*M/Rsep);
		const real Vd  = std::abs(Rij*Vij)/Rij.abs();
		const real Tff = 2.0/3.0*Rij.abs()/Vff;  /* free-fall time */
		const real Td  = Rij.abs()/Vd;    /* drift time */
		fprintf(stderr, " Tcoll~ %g [ Tff= %g  Td= %g ] \n", 
				  std::min(Tff, Td), Tff, Td);
//				std::sqrt(1.0/(1.0/(Tff*Tff) + 1.0/(Td*Td))), Tff, Td);
	}

	int NHALO = 0, NDISK = 0, NBULGE = 0, NDUST = 0;
	int NDISKGLOW = 0, NDUSTGLOW = 0;

	for (Particle::constIterator it = ptcl1.begin(); it != ptcl1.end(); it++)
	{
		const Particle &p = *it;
		if (p.ID >= 100000000)
			NBULGE++;
		else if (p.ID >= 70000000)
			NDUSTGLOW++;           
		else if (p.ID >= 50000000)
			NDUST++;           
		else if (p.ID >= 40000000)
			NDISKGLOW++;
		else
			NDISK++;
	}

	fprintf(stderr,"nobj in galaxy 1: %d   Mass: %f\n", h1.nbodies, M1);
	fprintf(stderr,"nobj in galaxy 1: halo %d disk %d diskGlow %d bulge %d dust %d dustGlow %d\n",
			NHALO, NDISK, NDISKGLOW, NBULGE , NDUST, NDUSTGLOW);

	NHALO = 0, NDISK = 0, NBULGE = 0, NDUST = 0;
	NDISKGLOW = 0, NDUSTGLOW = 0;

	for (Particle::constIterator it = ptcl2.begin(); it != ptcl2.end(); it++)
	{
		const Particle &p = *it;
		if (p.ID >= 100000000)
			NBULGE++;
		else if (p.ID >= 70000000)
			NDUSTGLOW++;           
		else if (p.ID >= 50000000)
			NDUST++;           
		else if (p.ID >= 40000000)
			NDISKGLOW++;
		else
			NDISK++;
	}

	fprintf(stderr,"nobj in galaxy 2: %d   Mass: %f\n", h2.nbodies, M2);
	fprintf(stderr,"nobj in galaxy 2: halo %d disk %d diskGlow %d bulge %d dust %d dustGlow %d\n",
			NHALO, NDISK, NDISKGLOW, NBULGE , NDUST, NDUSTGLOW);

	Particle::Vector merger;
	merger.reserve(128);
	for (int i = 0; i < h1.ndark; i++)
	{
		Particle p = ptcl1[i];
		p.pos = rot1 * p.pos;
		p.vel = rot1 * p.vel;
		merger.push_back(p);
	}
	for (int i = 0; i < h2.ndark; i++)
	{
		Particle p = ptcl2[i];
		p.pos = rot2 * p.pos + Rij;
		p.vel = rot2 * p.vel + Vij;
		merger.push_back(p);
	}

	for (int i = h1.ndark; i < h1.nbodies; i++)
	{
		Particle p = ptcl1[i];
		p.pos = rot1 * p.pos;
		p.vel = rot1 * p.vel;
		merger.push_back(p);
	}
	for (int i = h2.ndark; i < h2.nbodies; i++)
	{
		Particle p = ptcl2[i];
		p.pos = rot2 * p.pos + Rij;
		p.vel = rot2 * p.vel + Vij;
		merger.push_back(p);
	}

	centerGalaxy(merger);

	dump h;
	h.nbodies = h1.nbodies + h2.nbodies;
	h.ndark   = h1.ndark   + h2.ndark;
	h.nstar   = h1.nstar   + h2.nstar;
	h.nsph    = 0;
	h.ndim    = 3;
	h.time    = 0;

	writeGalaxy(h, merger, fout);

	fclose(fout);
	return 0;
}

