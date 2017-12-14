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
#include "anyoption.h"

#include "tipsydefs.h"

typedef struct 
{
	int ID;
	float m, x, y, z, vx, vy, vz, eps;
} phase;


double rot[3][3];

void rotmat(double i,double w)
{
	rot[0][0] = cos(w);
	rot[0][1] = -cos(i)*sin(w);
	rot[0][2] = -sin(i)*sin(w);
	rot[1][0] = sin(w);
	rot[1][1] = cos(i)*cos(w);
	rot[1][2] = sin(i)*cos(w);
	rot[2][0] = 0.0;
	rot[2][1] = -sin(i);
	rot[2][2] = cos(i);
	fprintf(stderr,"%g %g %g\n",rot[0][0], rot[0][1], rot[0][2]);
	fprintf(stderr,"%g %g %g\n",rot[1][0], rot[1][1], rot[1][2]);
	fprintf(stderr,"%g %g %g\n",rot[2][0], rot[2][1], rot[2][2]);
}

void rotate(double rot[3][3],float *vin)
{
	static double vout[3];

	for(int i=0; i<3; i++) {
		vout[i] = 0;
		for(int j=0; j<3; j++)
			vout[i] += rot[i][j] * vin[j]; 
		/* Remember the rotation matrix is the transpose of rot */
	}
	for(int i=0; i<3; i++)
		vin[i] = (float) vout[i];
}

void euler(int nobj, phase* rv, double inc, double omega)
{
	rotmat(inc,omega);
	for(int i=0; i<nobj; i++)
	{
		float r[3], v[3];
		r[0] = rv[i].x;
		r[1] = rv[i].y;
		r[2] = rv[i].z;
		v[0] = rv[i].vx;
		v[1] = rv[i].vy;
		v[2] = rv[i].vz;

		rotate(rot,r);
		rotate(rot,v);

		rv[i].x = r[0]; 
		rv[i].y = r[1]; 
		rv[i].z = r[2]; 
		rv[i].vx = v[0];
		rv[i].vy = v[1];
		rv[i].vz = v[2];
	}
}



void centerGalaxy(phase* r, int nobj)
{
	int i;
	float xc, yc, zc, vxc, vyc, vzc, mtot;

	mtot = 0;
	xc = yc = zc = vxc = vyc = vzc = 0;
	for(i=0; i<nobj; i++) {
		xc += r[i].m*r[i].x;
		yc += r[i].m*r[i].y;
		zc += r[i].m*r[i].z;
		vxc += r[i].m*r[i].vx;
		vyc += r[i].m*r[i].vy;
		vzc += r[i].m*r[i].vz;
		mtot += r[i].m;
	}
	xc /= mtot;
	yc /= mtot;
	zc /= mtot;
	vxc /= mtot;
	vyc /= mtot;
	vzc /= mtot;
	for(i=0; i<nobj; i++) {
		r[i].x -= xc;
		r[i].y -= yc;
		r[i].z -= zc;
		r[i].vx -= vxc;
		r[i].vy -= vyc;
		r[i].vz -= vzc;
	}  
}




int main(int argc, char **argv)
{

//  bool useRp = false;
//	bool useVrel = false;	
	std::string infilename1, infilename2, outfilename;
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile1 #        Input filename for the first galaxy ");
		ADDUSAGE(" -I  --infile2 #        Input filename for the second galaxy");
		ADDUSAGE(" -o  --outfile #        Output filename for the IC");
//		ADDUSAGE("     --useRp            Use pericenter distance instead of impac parameter [" << (useRp ? std::string("TRUE") : std::string("FALSE")) << "]");
//		ADDUSAGE("     --useVrel          Also use relative velocity parameter [" << (useVrel ? "TRUE" : "FALSE") << "]");
		ADDUSAGE(" ");

		opt.setFlag( "help" ,   'h');
//		opt.setFlag( "useRp");
//		opt.setFlag( "useVrel");
		opt.setOption( "infile1",  'i');
		opt.setOption( "infile2",  'I');
		opt.setOption( "outfile",  'o');

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
		if ((optarg = opt.getValue("infile1")))    infilename1  = std::string(optarg);
		if ((optarg = opt.getValue("infile2")))    infilename2  = std::string(optarg);
		if ((optarg = opt.getValue("outfile")))    outfilename  = std::string(optarg);
//		if (opt.getFlag("useRp"))   useRp   = true;
//		if (opt.getFlag("useVrel")) useVrel = true;

		if (infilename1.empty() || infilename2.empty() || outfilename.empty())
		{
			opt.printUsage();
			exit(0);
		}
	}
	
	int i;
	phase *r, *r2;

	FILE *rv1 = NULL,*rv2 = NULL;

	float ds=1.0, vs, ms=1.0;
	double m1, m2, mu1, mu2, vp = 0;
	float b=1.0, rsep=10.0;
	float x, y, vx, vy, x1, y1, vx1, vy1 ,  x2, y2, vx2, vy2;
	float theta, tcoll;
	double inc1=0, omega1=0;
	double inc2=0, omega2=0;
#if 0
	double Rp = -1, Vrel = -1;
#endif

	if( !(rv1 = fopen(infilename1.c_str(),"rb")) ) 
	{
		fprintf(stderr,"Can't find first file  %s\n", infilename1.c_str());
		exit(0);
	}
	if( !(rv2 = fopen(infilename2.c_str(),"rb")) ) 
	{
		fprintf(stderr,"Can't find second file %s\n", infilename2.c_str());
		exit(0);
	}

	std::cout << "Enter size ratio (for gal2): ";
	std::cin >> ds;
	std::cout << "Enter mass ratio (for gal2): ";
	std::cin >> ms;
#if 0
	if (useRp)
	{
		std::cout << "Enter pericentre distance (Rp): ";
		std::cin >> Rp;
	}
	else
	{
		std::cout << "Enter relative impact parameter (b): ";
		std::cin >> b;
	}
#else
		std::cout << "Enter pericentre distance (Rp): ";
		std::cin >> b;
#endif

	std::cout << "Enter initial separation: ";
	std::cin >> rsep;
#if 0
	if (useVrel)
	{
		std::cout << "Enter relative velocity (Vrel): ";
		std::cin >> Vrel;
	}
#endif
	std::cout << "Enter Euler angles for first galaxy:\n";
	std::cout << "Enter inclination: ";
	std::cin >> inc1;
	std::cout << "Enter omega: ";
	std::cin >> omega1;
	std::cout << "Enter Euler angles for second galaxy:\n";
	std::cout << "Enter inclination: ";
	std::cin >> inc2;
	std::cout << "Enter omega: ";
	std::cin >> omega2;



	double inc1_inp, inc2_inp, om2_inp, om1_inp;

	inc1_inp = inc1;
	inc2_inp = inc2;
	om1_inp = omega1;
	om2_inp = omega2;


	inc1   *= M_PI/180.;
	inc2   *= M_PI/180.;
	omega1 *= M_PI/180.;
	omega2 *= M_PI/180.;
	omega1 += M_PI;

	fprintf(stderr,"Size ratio: %f Mass ratio: %f \n", ds, ms);
	fprintf(stderr,"Relative impact par: %f Initial sep: %f \n", b, rsep);
	fprintf(stderr,"Euler angles first: %f %f Second: %f %f \n",
			inc1, omega1,inc2,omega2);

	vs = std::sqrt(ms/ds); /* adjustment for internal velocities */

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


	r = (phase *) calloc(NTotal1,sizeof(phase));

	struct dark_particle d;
	double massGalaxy1 = 0;	
	double massGalaxy2 = 0;

	int NHALO =0, NDISK =0, NBULGE = 0, NDUST = 0;
	int NDISKGLOW = 0, NDUSTGLOW = 0;


	for(i=0; i < NHalo1; i++)
	{
		fread(&d, sizeof(d), 1, rv1);
		r[i].ID       = (int) d.phi;
		r[i].m        = d.mass;
		r[i].x        = d.pos[0]; r[i].y = d.pos[1]; r[i].z = d.pos[2];
		r[i].vx       = d.vel[0]; r[i].vy = d.vel[1]; r[i].vz = d.vel[2];
		r[i].eps      = d.eps;
		massGalaxy1     += d.mass;
		NHALO++;
	}

	struct star_particle s;

	for(i=0; i < NBulge1; i++)
	{
		fread(&s, sizeof(s), 1, rv1);
		r[NHalo1+i].ID        = (int) s.phi;
		r[NHalo1+i].m         = s.mass;
		r[NHalo1+i].x         = s.pos[0]; r[NHalo1+i].y  = s.pos[1]; r[NHalo1+i].z  = s.pos[2];
		r[NHalo1+i].vx        = s.vel[0]; r[NHalo1+i].vy = s.vel[1]; r[NHalo1+i].vz = s.vel[2];
		r[NHalo1+i].eps       = s.eps;
		massGalaxy1          += s.mass;


		if(r[NHalo1+i].ID >= 100000000)
			NBULGE++;
		else if(r[NHalo1+i].ID >= 70000000)
			NDUSTGLOW++;           
		else if(r[NHalo1+i].ID >= 50000000)
			NDUST++;           
		else if(r[NHalo1+i].ID >= 40000000)
			NDISKGLOW++;
		else
			NDISK++;


	}


	fprintf(stderr,"nobj in galaxy 1: %d   Mass: %f\n",NTotal1, massGalaxy1);
	fprintf(stderr,"nobj in galaxy 1: halo %d disk %d diskGlow %d bulge %d dust %d dustGlow %d\n",
			NHALO, NDISK, NDISKGLOW, NBULGE , NDUST, NDUSTGLOW);

	centerGalaxy(r, NTotal1); /* centre everything of the main galaxy */


	//Read the particle Data of the second galaxy
	//Read the header
	fread(&h, sizeof(h), 1, rv2);

	fprintf(stderr, "Second Galaxy header %f %d %d %d %d %d \n",
			h.time, h.nbodies, h.ndim, h.nsph, h.ndark, h.nstar);

	int NTotal2   = h.nbodies;
	int NHalo2    = h.ndark;
	int NBulge2   = h.nstar;


	r2 = (phase *) calloc(NTotal2,sizeof(phase));

	NHALO =0;
	NDISK =0;
	NBULGE = 0;
	NDUST = 0;

	for(i=0; i < NHalo2; i++)
	{
		fread(&d, sizeof(d), 1, rv2);
		r2[i].ID       = (int) d.phi;
		r2[i].m        = d.mass;
		r2[i].x        = d.pos[0]; r2[i].y = d.pos[1]; r2[i].z = d.pos[2];
		r2[i].vx       = d.vel[0]; r2[i].vy = d.vel[1]; r2[i].vz = d.vel[2];
		r2[i].eps      = d.eps;
		massGalaxy2   += d.mass;
		NHALO++;
	}


	for(i=0; i < NBulge2; i++)
	{
		fread(&s, sizeof(s), 1, rv2);
		r2[NHalo2+i].ID       = (int) s.phi;
		r2[NHalo2+i].m        = s.mass;
		r2[NHalo2+i].x        = s.pos[0]; r2[NHalo2+i].y  = s.pos[1]; r2[NHalo2+i].z  = s.pos[2];
		r2[NHalo2+i].vx       = s.vel[0]; r2[NHalo2+i].vy = s.vel[1]; r2[NHalo2+i].vz = s.vel[2];
		r2[NHalo2+i].eps      = s.eps;
		massGalaxy2          += s.mass;

		const int ID = (int)s.phi;
		if(ID >= 100000000)
			NBULGE++;
		else if(ID >= 70000000)
			NDUSTGLOW++;           
		else if(ID >= 50000000)
			NDUST++;           
		else if(ID >= 40000000)
			NDISKGLOW++;
		else
			NDISK++;
	}


	fprintf(stderr,"nobj in galaxy 2: %d   massTest: %f\n",NTotal2, massGalaxy2);
	fprintf(stderr,"nobj in galaxy 2: halo %d disk %d diskGlow %d bulge %d dust %d dustGlow %d\n",
			NHALO, NDISK, NDISKGLOW, NBULGE , NDUST, NDUSTGLOW);

	centerGalaxy(r2, NTotal2); /* centre everything of the added galaxy */

	//Sum mass of galaxy 1
	m1 = massGalaxy1;
	fprintf(stderr,"m1 %g\t%f\n",m1, m1);

	//Sum mass of galaxy 2
	m2 = massGalaxy2;
	m2 = ms*m2;             //Adjust total mass
	fprintf(stderr,"m2 %g\t %g\n",m2,r2[1].m);

	mu1 = m2/(m1 + m2);
	mu2 = -m1/(m1 + m2);


	/* Relative Parabolic orbit - anti-clockwise */
	if( b > 0 ) 
	{
		vp = std::sqrt(2.0*(m1 + m2)/b);

#if 0
		if (useVrel) vp = Vrel;
		if (useRp  ) 	b = std::sqrt(Rp * (1.0 + 2*(m1+m2)/(Rp*vp*vp)));
#endif

		x = 2*b - rsep;  y = -2*std::sqrt(b*(rsep-b));
		vx = std::sqrt(b*(rsep-b))*vp/rsep; vy = b*vp/rsep;
	}
	else 
	{
		b = 0;
		x = - rsep; y = 0.0;
		vx = std::sqrt(2.0*(m1 + m2)/rsep); vy = 0.0;
	}

	/* Calculate collison time */
	if( b > 0 ) 
	{
		theta = atan2(y,x);
		tcoll = (0.5*tan(0.5*theta) + pow(tan(0.5*theta),3.0)/6.)*4*b/vp;
		fprintf(stderr,"Collision time is t=%g\n",tcoll);
	}
	else {
		tcoll = -pow(rsep,1.5)/(1.5*std::sqrt(2.0*(m1+m2)));
		fprintf(stderr,"Collision time is t=%g\n",tcoll);
	}

	/* These are the orbital adjustments for a parabolic encounter */
	/* Change to centre of mass frame */
	x1  =  mu1*x;  x2   =  mu2*x;     
	y1  =  mu1*y;  y2   =  mu2*y;
	vx1 =  mu1*vx; vx2  =  mu2*vx;
	vy1 =  mu1*vy; vy2  =  mu2*vy;


	/* Rotate the galaxies */
	euler(NTotal1, r, inc1,omega1);
	euler(NTotal2,r2, inc2,omega2);

	for(i=0; i< NTotal1; i++) {
		r[i].x  = r[i].x  + x1;
		r[i].y  = r[i].y  + y1;
		r[i].vx = r[i].vx + vx1;
		r[i].vy = r[i].vy + vy1;
	}
	/* Rescale and reset the second galaxy */
	for(i=0; i< NTotal2; i++) {
		r2[i].m *= ms;
		r2[i].x = ds*r2[i].x + x2;
		r2[i].y = ds*r2[i].y + y2;
		r2[i].z = ds*r2[i].z;
		r2[i].vx = vs*r2[i].vx + vx2;
		r2[i].vy = vs*r2[i].vy + vy2;
		r2[i].vz = vs*r2[i].vz;
	}



	FILE *outfile;
	if( !(outfile = fopen(outfilename.c_str(),"wb")) ) 
	{
		fprintf(stderr,"Can't open output file %s\n", outfilename.c_str());
		exit(0);
	}


	//Write tipsy header
	h.nbodies = NTotal2 + NTotal1;
	h.ndark   = NHalo1  + NHalo2;
	h.nstar   = NBulge1 + NBulge2;
	h.nsph    = 0;
	h.ndim    = 3;
	h.time    = 0;

	fwrite(&h, sizeof(h), 1, outfile);
	//First write DM Halo of main galaxy
	int maxDMID = -1;
	for(i=0; i < NHalo1; i++)
	{
		d.mass      = r[i].m;
		d.pos[0]    = r[i].x;
		d.pos[1]    = r[i].y;
		d.pos[2]    = r[i].z;
		d.vel[0]    = r[i].vx;
		d.vel[1]    = r[i].vy;
		d.vel[2]    = r[i].vz;
		d.phi       = r[i].ID;    
		d.eps       = r[i].eps;
		maxDMID     = std::max(maxDMID, d.phi);
		fwrite(&d, sizeof(d), 1, outfile);
	}

	maxDMID++;
	//Now for each child galaxy that has to be added
	for(i=0; i < NHalo2; i++)
	{
		d.mass      = r2[i].m;
		d.pos[0]    = r2[i].x;
		d.pos[1]    = r2[i].y;
		d.pos[2]    = r2[i].z;
		d.vel[0]    = r2[i].vx;
		d.vel[1]    = r2[i].vy;
		d.vel[2]    = r2[i].vz;
		d.eps       = r2[i].eps;;
		d.phi       = maxDMID++; 
		fwrite(&d, sizeof(d), 1, outfile);
	}      

	//Now write the star particles of main galaxy
	int maxBulgeID = -1;
	int maxDiskID  = -1;
	int maxDustID  = -1;


	//         maxDMID = -1;
	for(i=NHalo1; i < NHalo1 + NBulge1; i++)
	{
		s.mass      = r[i].m;
		s.pos[0]    = r[i].x;
		s.pos[1]    = r[i].y;
		s.pos[2]    = r[i].z;
		s.vel[0]    = r[i].vx;
		s.vel[1]    = r[i].vy;
		s.vel[2]    = r[i].vz;
		s.phi       = r[i].ID;    
		s.eps       = r[i].eps;;    

#if 0
		if(s.phi >= 100000000)
			maxBulgeID  = std::max(maxBulgeID, s.phi);
		else if(s.phi >= 50000000)
			maxDustID  = std::max(maxDustID, s.phi);            
		else
			maxDiskID  = std::max(maxDiskID, s.phi);
#endif

		fwrite(&s, sizeof(s), 1, outfile);
	}

	fprintf(stderr, "%d %d %d \n",
			maxBulgeID,maxDustID, maxDiskID);

	maxBulgeID++;
	maxDustID++;
	maxDiskID++;
	//Now for each child galaxy that has to be added
	for(i=NHalo2; i < NHalo2 + NBulge2; i++)
	{
		s.mass      = r2[i].m;
		s.pos[0]    = r2[i].x;
		s.pos[1]    = r2[i].y;
		s.pos[2]    = r2[i].z;
		s.vel[0]    = r2[i].vx;
		s.vel[1]    = r2[i].vy;
		s.vel[2]    = r2[i].vz;
		s.eps       = r2[i].eps;; 
		s.phi       = r2[i].ID; 

#if 0
		if(s.phi >= 100000000)
			s.phi       = maxBulgeID++;   
		else if(s.phi >= 50000000)
			s.phi       = maxDustID++;            
		else
			s.phi       = maxDiskID++;   
#endif


		fwrite(&s, sizeof(s), 1, outfile);
	}      

	fclose(outfile);

	fprintf(stderr,"Max ids: Halo %d Bulge: %d Disk: %d Dust: %d \n",
			maxDMID, maxBulgeID, maxDiskID, maxDustID);


	//Write to the settings file

	char settingsFile[256];
	sprintf(settingsFile, "%s.settings", outfilename.c_str());

	fprintf(stderr, "Settings written to: %s \n", settingsFile);

	if( !(outfile = fopen(settingsFile,"w")) ) {
		fprintf(stderr,"Can't open settings file %s\n",settingsFile);
		exit(0);
	} 

	fprintf(outfile, "Input file 1: %s \n", argv[1]);
	fprintf(outfile, "Input file 2: %s \n", argv[2]);

	fprintf(outfile, "Size ratio  : %f \n", ds);
	fprintf(outfile, "Mass ratio  : %f \n", ms);

	fprintf(outfile, "Seperation  : %f \n", rsep);
	fprintf(outfile, "Pericenter  : %f \n", b);

	fprintf(outfile, "inclination1: %f \n", inc1_inp);
	fprintf(outfile, "omage1      : %f \n", om1_inp);  

	fprintf(outfile, "inclination2: %f \n", inc2_inp);
	fprintf(outfile, "omage2      : %f \n", om2_inp);  

	fclose(outfile);  

	return 0;
}

