/*

Bonsai V2: A parallel GPU N-body gravitational Tree-code

(c) 2010-2012:
Jeroen Bedorf
Evghenii Gaburov
Simon Portegies Zwart

Leiden Observatory, Leiden University

http://castle.strw.leidenuniv.nl
http://github.com/treecode/Bonsai

*/

#ifdef WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <Windows.h>
#define M_PI        3.14159265358979323846264338328
#endif

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <fstream>
#include <sstream>
#include "log.h"
#include "anyoption.h"
#include "renderloop.h"

#if ENABLE_LOG
bool ENABLE_RUNTIME_LOG;
#endif

using namespace std;

#include "../profiling/bonsai_timing.h"

int devID;
int renderDevID;

extern void initTimers()
{
#ifndef CUXTIMER_DISABLE
  // Set up the profiling timing info
  build_tree_init();
  compute_propertiesD_init();
  dev_approximate_gravity_init();
  parallel_init();
  sortKernels_init();
  timestep_init();
#endif
}

extern void displayTimers()
{
#ifndef CUXTIMER_DISABLE
  // Display all timing info on the way out
  build_tree_display();
  compute_propertiesD_display();
  //dev_approximate_gravity_display();
  //parallel_display();
  //sortKernels_display();
  //timestep_display();
#endif
}

#include "octree.h"

#ifdef USE_OPENGL
#include "renderloop.h"
#include <cuda_gl_interop.h>
#endif

void read_dumbp_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,  vector<int> &bodiesIDs,  float eps2,
                     string fileName, int rank, int procs, int &NTotal2, int &NFirst, int &NSecond, int &NThird, octree *tree, int reduce_bodies_factor)  
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  
  //Now we have different types of files, try to determine which one is used
  /*****
  If individual softening is on there is only one option:
  Header is formatted as follows: 
  N     #       #       #
  so read the first number and compute how particles should be distributed
  
  If individual softening is NOT enabled, i can be anything, but for ease I assume standard dumbp files:
  no Header
  ID mass x y z vx vy vz
  now the next step is risky, we assume mass adds up to 1, so number of particles will be : 1 / mass
  use this as initial particle distribution
  
  */
  
  
  char fullFileName[256];
  sprintf(fullFileName, "%s", fileName.c_str());

  cout << "Trying to read file: " << fullFileName << endl;

  ifstream inputFile(fullFileName, ios::in);

  if(!inputFile.is_open())
  {
    cout << "Can't open input file \n";
    exit(0);
  }
  
  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;

  #ifndef INDSOFT
     inputFile >> idummy >> positions.w;
     inputFile.seekg(0, ios::beg); //Reset file pointer
     NTotal = (int)(1 / positions.w);
  #else
     //Read the Ntotal from the file header
     inputFile >> NTotal >> NFirst >> NSecond >> NThird;
  #endif
  
  
  
  //Rough divide
  uint perProc = NTotal / procs;
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;

  int globalParticleCount = 0;

  while(!inputFile.eof()) {
    
    inputFile >> idummy
              >> positions.w >> positions.x >> positions.y >> positions.z
              >> velocity.x >> velocity.y >> velocity.z;    

	globalParticleCount++;

	if( globalParticleCount % reduce_bodies_factor == 0 ) 
		positions.w *= reduce_bodies_factor;

	if( globalParticleCount % reduce_bodies_factor != 0 )
		continue;

    #ifndef INDSOFT
      velocity.w = sqrt(eps2);
    #else
      inputFile >> velocity.w; //Read the softening from the input file
    #endif
    
    bodyPositions.push_back(positions);
    bodyVelocities.push_back(velocity);
    
    #ifndef INDSOFT    
      idummy = particleCount;
    #endif
    
    bodiesIDs.push_back(idummy);  
    
    particleCount++;
  
  
    if(bodyPositions.size() > perProc && procCntr != procs)
    {       
      tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
      procCntr++;
      
      bodyPositions.clear();
      bodyVelocities.clear();
      bodiesIDs.clear();
    }
  }//end while
  
  inputFile.close();
  
  //Clear the last one since its double
  bodyPositions.resize(bodyPositions.size()-1);  
  NTotal2 = particleCount-1;
  
  cerr << "NTotal: " << NTotal << "\tper proc: " << perProc << "\tFor ourself:" << bodiesIDs.size() << endl;
}


void read_tipsy_file_parallel(vector<real4> &bodyPositions, vector<real4> &bodyVelocities,
                              vector<int> &bodiesIDs,  float eps2, string fileName, 
                              int rank, int procs, int &NTotal2, int &NFirst, 
                              int &NSecond, int &NThird, octree *tree,
                              vector<real4> &dustPositions, vector<real4> &dustVelocities,
                              vector<int> &dustIDs, int reduce_bodies_factor,
                              int reduce_dust_factor)  
{
  //Process 0 does the file reading and sends the data
  //to the other processes
  /* 

     Read in our custom version of the tipsy file format.
     Most important change is that we store particle id on the 
     location where previously the potential was stored.
  */
  
  
  char fullFileName[256];
  sprintf(fullFileName, "%s", fileName.c_str());

  cout << "Trying to read file: " << fullFileName << endl;
  
  
  
  ifstream inputFile(fullFileName, ios::in | ios::binary);
  if(!inputFile.is_open())
  {
    cout << "Can't open input file \n";
    exit(0);
  }
  
  dump  h;
  inputFile.read((char*)&h, sizeof(h));  

  int NTotal;
  int idummy;
  real4 positions;
  real4 velocity;

     
  //Read tipsy header  
  NTotal        = h.nbodies;
  NFirst        = h.ndark;
  NSecond       = h.nstar;
  NThird        = h.nsph;

  tree->set_t_current((float) h.time);
  
  //Rough divide
  uint perProc = NTotal / procs;
  bodyPositions.reserve(perProc+10);
  bodyVelocities.reserve(perProc+10);
  bodiesIDs.reserve(perProc+10);
  perProc -= 1;

  //Start reading
  int particleCount = 0;
  int procCntr = 1;
  
  dark_particle d;
  star_particle s;

  int globalParticleCount = 0;
  int bodyCount = 0;
  int dustCount = 0;
  
  for(int i=0; i < NTotal; i++)
  {
    if(i < NFirst)
    {
      inputFile.read((char*)&d, sizeof(d));
      velocity.w        = d.eps;
      positions.w       = d.mass;
      positions.x       = d.pos[0];
      positions.y       = d.pos[1];
      positions.z       = d.pos[2];
      velocity.x        = d.vel[0];
      velocity.y        = d.vel[1];
      velocity.z        = d.vel[2];
      idummy            = d.phi;
    }
    else
    {
      inputFile.read((char*)&s, sizeof(s));
      velocity.w        = s.eps;
      positions.w       = s.mass;
      positions.x       = s.pos[0];
      positions.y       = s.pos[1];
      positions.z       = s.pos[2];
      velocity.x        = s.vel[0];
      velocity.y        = s.vel[1];
      velocity.z        = s.vel[2];
      idummy            = s.phi;
    }

	globalParticleCount++;
   
    #ifdef USE_DUST
      if(idummy >= 50000000 && idummy < 100000000)
      {
        dustCount++;
        if( dustCount % reduce_dust_factor == 0 ) 
          positions.w *= reduce_dust_factor;

        if( dustCount % reduce_dust_factor != 0 )
          continue;
        dustPositions.push_back(positions);
        dustVelocities.push_back(velocity);
        dustIDs.push_back(idummy);      
      }
      else
      {
        bodyCount++;
        if( bodyCount % reduce_bodies_factor == 0 ) 
		      positions.w *= reduce_bodies_factor;

	      if( bodyCount % reduce_bodies_factor != 0 )
		      continue;
        bodyPositions.push_back(positions);
        bodyVelocities.push_back(velocity);
        bodiesIDs.push_back(idummy);  
      }

    
    #else
      if( globalParticleCount % reduce_bodies_factor == 0 ) 
        positions.w *= reduce_bodies_factor;

      if( globalParticleCount % reduce_bodies_factor != 0 )
        continue;
      bodyPositions.push_back(positions);
      bodyVelocities.push_back(velocity);
      bodiesIDs.push_back(idummy);  
    #endif
    
    particleCount++;
  
  
    if(bodyPositions.size() > perProc && procCntr != procs)
    { 
      tree->ICSend(procCntr,  &bodyPositions[0], &bodyVelocities[0],  &bodiesIDs[0], (int)bodyPositions.size());
      procCntr++;
      
      bodyPositions.clear();
      bodyVelocities.clear();
      bodiesIDs.clear();
    }
  }//end while
  
  inputFile.close();
  
  //Clear the last one since its double
//   bodyPositions.resize(bodyPositions.size()-1);  
//   NTotal2 = particleCount-1;
  NTotal2 = particleCount;
  cerr << "NTotal: " << NTotal << "\tper proc: " << perProc << "\tFor ourself:" << bodiesIDs.size() << "\tNDust: " << dustPositions.size() << endl;
  cerr << "NTotal: " << NTotal << "\tper proc: " << perProc << "\tFor ourself:" << bodiesIDs.size() << endl;  
}


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

void euler(vector<real4> &bodyPositions,
           vector<real4> &bodyVelocities,
           double inc, double omega)
{
  rotmat(inc,omega);
  size_t nobj = bodyPositions.size();
  for(uint i=0; i < nobj; i++)
  {
      float r[3], v[3];
      r[0] = bodyPositions[i].x;
      r[1] = bodyPositions[i].y;
      r[2] = bodyPositions[i].z;
      v[0] = bodyVelocities[i].x;
      v[1] = bodyVelocities[i].y;
      v[2] = bodyVelocities[i].z;

      rotate(rot,r);
      rotate(rot,v);

      bodyPositions[i].x = r[0]; 
      bodyPositions[i].y = r[1]; 
      bodyPositions[i].z = r[2]; 
      bodyVelocities[i].x = v[0];
      bodyVelocities[i].y = v[1];
      bodyVelocities[i].z = v[2];
  }
}



double centerGalaxy(vector<real4> &bodyPositions,
                    vector<real4> &bodyVelocities)
{
    size_t nobj = bodyPositions.size();
    float xc, yc, zc, vxc, vyc, vzc, mtot;
  

    mtot = 0;
    xc = yc = zc = vxc = vyc = vzc = 0;
    for(uint i=0; i< nobj; i++) {
            xc   += bodyPositions[i].w*bodyPositions[i].x;
            yc   += bodyPositions[i].w*bodyPositions[i].y;
            zc   += bodyPositions[i].w*bodyPositions[i].z;
            vxc  += bodyPositions[i].w*bodyVelocities[i].x;
            vyc  += bodyPositions[i].w*bodyVelocities[i].y;
            vzc  += bodyPositions[i].w*bodyVelocities[i].z;
            mtot += bodyPositions[i].w;
    }
    xc /= mtot;
    yc /= mtot;
    zc /= mtot;
    vxc /= mtot;
    vyc /= mtot;
    vzc /= mtot;
    for(uint i=0; i< nobj; i++)
    {
      bodyPositions[i].x  -= xc;
      bodyPositions[i].y  -= yc;
      bodyPositions[i].z  -= zc;
      bodyVelocities[i].x -= vxc;
      bodyVelocities[i].y -= vyc;
      bodyVelocities[i].z -= vzc;
    }
    
    return mtot;
}




int setupMergerModel(vector<real4> &bodyPositions1,
                     vector<real4> &bodyVelocities1,
                     vector<int>   &bodyIDs1,
                     vector<real4> &bodyPositions2,
                     vector<real4> &bodyVelocities2,
                     vector<int>   &bodyIDs2){
        uint i;
        double ds=1.0, vs, ms=1.0;
        double mu1, mu2, vp;
        double b=1.0, rsep=10.0;
        double x, y, vx, vy, x1, y1, vx1, vy1 ,  x2, y2, vx2, vy2;
        double theta, tcoll;
        double inc1=0, omega1=0;
        double inc2=0, omega2=0;
        
        
        ds = 1.52;
        ms = 1.0;
        b = 10;
        rsep = 168;
        inc1 = 0;
        omega1 = 0;
        inc2 = 180;
        omega2 = 0;


        if(ds < 0)
        {
          cout << "Enter size ratio (for gal2): ";
          cin >> ds;
          cout << "Enter mass ratio (for gal2): ";
          cin >> ms;
          cout << "Enter relative impact parameter: ";
          cin >> b;
          
          cout << "Enter initial separation: ";
          cin >> rsep;
          cout << "Enter Euler angles for first galaxy:\n";
          cout << "Enter inclination: ";
          cin >> inc1;
          cout << "Enter omega: ";
          cin >> omega1;
          cout << "Enter Euler angles for second galaxy:\n";
          cout << "Enter inclination: ";
          cin >> inc2;
          cout << "Enter omega: ";
          cin >> omega2;
        }


        double inc1_inp, inc2_inp, om2_inp, om1_inp;
        
        inc1_inp = inc1;
        inc2_inp = inc2;
        om1_inp = omega1;
        om2_inp = omega1;


        inc1   *= M_PI/180.;
        inc2   *= M_PI/180.;
        omega1 *= M_PI/180.;
        omega2 *= M_PI/180.;
        omega1 += M_PI;

        fprintf(stderr,"Size ratio: %f Mass ratio: %f \n", ds, ms);
        fprintf(stderr,"Relative impact par: %f Initial sep: %f \n", b, rsep);
        fprintf(stderr,"Euler angles first: %f %f Second: %f %f \n",
                        inc1, omega1,inc2,omega2);

        vs = sqrt(ms/ds); /* adjustment for internal velocities */


        //Center everything in galaxy 1 and galaxy 2
        double galaxyMass1 = centerGalaxy(bodyPositions1, bodyVelocities1);
        double galaxyMass2 = centerGalaxy(bodyPositions2, bodyVelocities2);


        galaxyMass2 = ms*galaxyMass2;             //Adjust total mass

        mu1 =  galaxyMass2/(galaxyMass1 + galaxyMass2);
        mu2 = -galaxyMass1/(galaxyMass1 + galaxyMass2);
        
        double m1 = galaxyMass1;
        double m2 = galaxyMass2;

        
        /* Relative Parabolic orbit - anti-clockwise */
        if( b > 0 ) {
                vp = sqrt(2.0*(m1 + m2)/b);
                x = 2*b - rsep;  y = -2*sqrt(b*(rsep-b));
                vx = sqrt(b*(rsep-b))*vp/rsep; vy = b*vp/rsep;
        }
        else {
                b = 0;
                x = - rsep; y = 0.0;
                vx = sqrt(2.0*(m1 + m2)/rsep); vy = 0.0;
        }

        /* Calculate collison time */
        if( b > 0 ) {
                theta = atan2(y,x);
                tcoll = (0.5*tan(0.5*theta) + pow(tan(0.5*theta),3.0)/6.)*4*b/vp;
                fprintf(stderr,"Collision time is t=%g\n",tcoll);
        }
        else {
                tcoll = -pow(rsep,1.5)/(1.5*sqrt(2.0*(m1+m2)));
                fprintf(stderr,"Collision time is t=%g\n",tcoll);
        }

        /* These are the orbital adjustments for a parabolic encounter */
        /* Change to centre of mass frame */
        x1  =  mu1*x;  x2   =  mu2*x;     
        y1  =  mu1*y;  y2   =  mu2*y;
        vx1 =  mu1*vx; vx2  =  mu2*vx;
        vy1 =  mu1*vy; vy2  =  mu2*vy;


        /* Rotate the galaxies */
        euler(bodyPositions1, bodyVelocities1, inc1,omega1);
        euler(bodyPositions2, bodyVelocities2, inc2,omega2);

        for(i=0; i< bodyPositions1.size(); i++) {
                bodyPositions1[i].x  = (float) (bodyPositions1[i].x  + x1);
                bodyPositions1[i].y  = (float) (bodyPositions1[i].y  + y1);
                bodyVelocities1[i].x = (float) (bodyVelocities1[i].x + vx1);
                bodyVelocities1[i].y = (float) (bodyVelocities1[i].y + vy1);
        }
        /* Rescale and reset the second galaxy */
        for(i=0; i< bodyPositions2.size(); i++) {
                bodyPositions2[i].w = (float) ms*bodyPositions2[i].w;
                bodyPositions2[i].x = (float) (ds*bodyPositions2[i].x + x2);
                bodyPositions2[i].y = (float) (ds*bodyPositions2[i].y + y2);
                bodyPositions2[i].z = (float) ds*bodyPositions2[i].z;
                bodyVelocities2[i].x = (float) (vs*bodyVelocities2[i].x + vx2);
                bodyVelocities2[i].y = (float) (vs*bodyVelocities2[i].y + vy2);
                bodyVelocities2[i].z = (float) vs*bodyVelocities2[i].z;
        }


        //Put them into one 
        bodyPositions1.insert(bodyPositions1.end(),  bodyPositions2.begin(), bodyPositions2.end());
        bodyVelocities1.insert(bodyVelocities1.end(), bodyVelocities2.begin(), bodyVelocities2.end());
        bodyIDs1.insert(bodyIDs1.end(), bodyIDs2.begin(), bodyIDs2.end());
  

        return 0;
}


long long my_dev::base_mem::currentMemUsage;
long long my_dev::base_mem::maxMemUsage;

int main(int argc, char** argv)
{
  vector<real4> bodyPositions;
  vector<real4> bodyVelocities;
  vector<int>   bodyIDs;

  vector<real4> dustPositions;
  vector<real4> dustVelocities;
  vector<int>   dustIDs;  
  
  float eps      = 0.05f;
  float theta    = 0.75f;
  float timeStep = 1.0f / 16.0f;
  float  tEnd      = 1;
  devID      = 0;
  renderDevID = 0;

  string fileName       =  "";
  string logFileName    = "gpuLog.log";
  string snapshotFile   = "snapshot_";
  int snapshotIter      = -1;
  float  killDistance   = -1.0;
  float  remoDistance   = -1.0;
  int    snapShotAdd    =  0;
  int rebuild_tree_rate = 2;
  int reduce_bodies_factor = 1;
  int reduce_dust_factor = 1;
  string fullScreenMode = "";
  bool direct = false;
  bool fullscreen = false;
  bool displayFPS = false;

#if ENABLE_LOG
  ENABLE_RUNTIME_LOG = false;
#endif

#ifdef USE_OPENGL
	TstartGlow = 0.0;
	dTstartGlow = 1.0;
#endif

	/************** beg - command line arguments ********/
#if 1
	{
		AnyOption opt;

#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename ");
		ADDUSAGE("     --logfile #        Log filename [" << logFileName << "]");
		ADDUSAGE("     --dev #            Device ID [" << devID << "]");
		ADDUSAGE("     --renderdev #      Rendering Device ID [" << renderDevID << "]");
		ADDUSAGE(" -t  --dt #             time step [" << timeStep << "]");
		ADDUSAGE(" -T  --tend #           N-body end time [" << tEnd << "]");
		ADDUSAGE(" -e  --eps #            softening (will be squared) [" << eps << "]");
		ADDUSAGE(" -o  --theta #          opening angle (theta) [" <<theta << "]");
		ADDUSAGE("     --snapname #       snapshot base name (N-body time is appended in 000000 format) [" << snapshotFile << "]");
		ADDUSAGE("     --snapiter #       snapshot iteration (N-body time) [" << snapshotIter << "]");
		ADDUSAGE("     --killdist #       kill distance (-1 to disable) [" << killDistance << "]");
		ADDUSAGE("     --rmdist #         Particle removal distance (-1 to disable) [" << remoDistance << "]");
		ADDUSAGE("     --valueadd #       value to add to the snapshot [" << snapShotAdd << "]");
		ADDUSAGE(" -r  --rebuild #        rebuild tree every # steps [" << rebuild_tree_rate << "]");
		ADDUSAGE("     --reducebodies #   cut down bodies dataset by # factor ");
#ifdef USE_DUST
    ADDUSAGE("     --reducedust #     cut down dust dataset by # factor ");
#endif
#if ENABLE_LOG
    ADDUSAGE("     --log              enable logging ");
#endif
        ADDUSAGE("     --direct           enable N^2 direct gravitation [" << (direct ? "on" : "off") << "]");
#ifdef USE_OPENGL
		ADDUSAGE("     --fullscreen #     set fullscreen mode string");
    ADDUSAGE("     --displayfps       enable on-screen FPS display");
		ADDUSAGE("     --Tglow  #         enable glow @ # Myr [" << TstartGlow << "]");
		ADDUSAGE("     --dTglow  #        reach full brightness in @ # Myr [" << dTstartGlow << "]");
#endif


		ADDUSAGE(" ");


		opt.setFlag( "help" ,   'h');
		opt.setOption( "infile",  'i');
		opt.setOption( "dt",      't' );
		opt.setOption( "tend",    'T' );
		opt.setOption( "eps",     'e' );
		opt.setOption( "theta",   'o' );
		opt.setOption( "rebuild", 'r' );
		opt.setOption( "dev" );
		opt.setOption( "renderdev" );
		opt.setOption( "logfile" );
		opt.setOption( "snapname");
		opt.setOption( "snapiter");
		opt.setOption( "killdist");
		opt.setOption( "rmdist");
		opt.setOption( "valueadd");
		opt.setOption( "reducebodies");
#ifdef USE_DUST
    opt.setOption( "reducedust");
#endif /* USE_DUST */
#if ENABLE_LOG
		opt.setFlag("log");
#endif
    opt.setFlag("direct");
#ifdef USE_OPENGL
		opt.setOption( "fullscreen");
		opt.setOption( "Tglow");
		opt.setOption( "dTglow");
    opt.setFlag("displayfps");
#endif
  
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

    if (opt.getFlag("direct")) direct = true;
    if (opt.getFlag("displayfps")) displayFPS = true;

#if ENABLE_LOG
    if (opt.getFlag("log")) ENABLE_RUNTIME_LOG = true;
#endif    
		char *optarg = NULL;
		if ((optarg = opt.getValue("infile")))       fileName           = string(optarg);
		if ((optarg = opt.getValue("logfile")))      logFileName        = string(optarg);
		if ((optarg = opt.getValue("dev")))          devID              = atoi  (optarg);
        renderDevID = devID;
        if ((optarg = opt.getValue("renderdev")))    renderDevID        = atoi  (optarg);
		if ((optarg = opt.getValue("dt")))           timeStep           = (float) atof  (optarg);
		if ((optarg = opt.getValue("tend")))         tEnd               = (float) atof  (optarg);
		if ((optarg = opt.getValue("eps")))          eps                = (float) atof  (optarg);
		if ((optarg = opt.getValue("theta")))        theta              = (float) atof  (optarg);
		if ((optarg = opt.getValue("snapname")))     snapshotFile       = string(optarg);
		if ((optarg = opt.getValue("snapiter")))     snapshotIter       = atoi  (optarg);
		if ((optarg = opt.getValue("killdist")))     killDistance       = (float) atof  (optarg);
		if ((optarg = opt.getValue("rmdist")))       remoDistance       = (float) atof  (optarg);
		if ((optarg = opt.getValue("valueadd")))     snapShotAdd        = atoi  (optarg);
		if ((optarg = opt.getValue("rebuild")))      rebuild_tree_rate  = atoi  (optarg);
		if ((optarg = opt.getValue("reducebodies"))) reduce_bodies_factor = atoi  (optarg);
        if ((optarg = opt.getValue("reducedust")))	 reduce_dust_factor = atoi  (optarg);
#if USE_OPENGL
        if ((optarg = opt.getValue("fullscreen")))	 fullScreenMode     = string(optarg);
        if ((optarg = opt.getValue("Tglow")))	 TstartGlow  = (float)atof(optarg);
        if ((optarg = opt.getValue("dTglow")))	 dTstartGlow  = (float)atof(optarg);
        dTstartGlow = std::max(dTstartGlow, 1.0f);
#endif
		if (fileName.empty()) 
		{
			opt.printUsage();
			exit(0);
		}

#undef ADDUSAGE
	}
#endif
	/************** end - command line arguments ********/

	
  cerr << "Used settings: \n";
	cerr << "Input filename " << fileName << endl;
	cerr << "Log filename " << logFileName << endl;
	cerr << "Theta: \t\t"             << theta        << "\t\teps: \t\t"          << eps << endl;
	cerr << "Timestep: \t"          << timeStep     << "\t\ttEnd: \t\t"         << tEnd << endl;
	cerr << "snapshotFile: \t"      << snapshotFile << "\tsnapshotIter: \t" << snapshotIter << endl;
	cerr << "Input file: \t"        << fileName     << "\t\tdevID: \t\t"        << devID << endl;
	cerr << "Kill distance: \t"      << killDistance     << "\t\tRemove dist: \t"   << remoDistance << endl;
	cerr << "Snapshot Addition: \t"  << snapShotAdd << endl;
	cerr << "Rebuild tree every " << rebuild_tree_rate << " timestep\n";
        
        if( reduce_bodies_factor > 1 )
          cout << "Reduce number of non-dust bodies by " << reduce_bodies_factor << " \n";
        if( reduce_dust_factor > 1 )
          cout << "Reduce number of dust bodies by " << reduce_dust_factor << " \n";
        
        #if ENABLE_LOG
          if (ENABLE_RUNTIME_LOG)
            cerr << " Runtime logging is ENABLED \n";
          else
            cerr << " Runtime logging is DISABLED \n";
        #endif
        cerr << " Direct gravitation is " << (direct ? "ENABLED" : "DISABLED") << endl;
        #if USE_OPENGL
                cerr << "Tglow = " << TstartGlow << endl;
                cerr << "dTglow = " << dTstartGlow << endl;
        #endif
        #ifdef USE_MPI                
          cerr << " Code is built WITH MPI Support \n";
        #else
          cerr << " Code is built WITHOUT MPI Support \n";          
        #endif

  int NTotal, NFirst, NSecond, NThird;
  NTotal = NFirst = NSecond = NThird = 0;

#ifdef USE_OPENGL
  // create OpenGL context first, and register for interop
  initGL(argc, argv, fullScreenMode.c_str());
  cudaGLSetGLDevice(devID);
#endif

  initTimers();

  //Creat the octree class and set the properties
  octree *tree = new octree(argv, devID, theta, eps, snapshotFile, snapshotIter,  timeStep, (int)tEnd, killDistance, (int)remoDistance, snapShotAdd, rebuild_tree_rate, direct);
                            
                            
  //Get parallel processing information  
  int procId = tree->mpiGetRank();
  int nProcs = tree->mpiGetNProcs();
  
  //Used for profiler
  char *gpu_prof_log;
  gpu_prof_log=getenv("CUDA_PROFILE_LOG");
  if(gpu_prof_log){
    char tmp[50];
    sprintf(tmp,"process%d_%s",procId,gpu_prof_log);
#ifdef WIN32
    SetEnvironmentVariable("CUDA_PROFILE_LOG", tmp);
#else
    setenv("CUDA_PROFILE_LOG",tmp,1);
#endif
  }
      
  if(nProcs > 1)
  {
    logFileName.append("-");
    
    char buff[16];
    sprintf(buff,"%d-%d", nProcs, procId);
    logFileName.append(buff);
  }
  
  ofstream logFile(logFileName.c_str());
    
  tree->set_context(logFile, false); //Do logging to file and enable timing (false = enabled)
  
  if(procId == 0)
  {
   #ifdef TIPSYOUTPUT
      read_tipsy_file_parallel(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, 
                               procId, nProcs, NTotal, NFirst, NSecond, NThird, tree,
                               dustPositions, dustVelocities, dustIDs, reduce_bodies_factor, reduce_dust_factor);    
   #else
      read_dumbp_file_parallel(bodyPositions, bodyVelocities, bodyIDs, eps, fileName, procId, nProcs, NTotal, NFirst, NSecond, NThird, tree, reduce_bodies_factor);
   #endif

  }
  else
  {
    tree->ICRecv(0, bodyPositions, bodyVelocities,  bodyIDs);
  }

#ifdef TIPSYOUTPUT
	fprintf(stderr, " t_current = %g\n", tree->get_t_current());
#endif
  
  
  //#define SETUP_MERGER
  #ifdef SETUP_MERGER
    vector<real4> bodyPositions2;
    vector<real4> bodyVelocities2;
    vector<int>   bodyIDs2;  
    
    bodyPositions2.insert(bodyPositions2.begin(),   bodyPositions.begin(),  bodyPositions.end());
    bodyVelocities2.insert(bodyVelocities2.begin(), bodyVelocities.begin(), bodyVelocities.end());
    bodyIDs2.insert(bodyIDs2.begin(), bodyIDs.begin(), bodyIDs.end());
    
    
    setupMergerModel(bodyPositions,  bodyVelocities,  bodyIDs,
                     bodyPositions2, bodyVelocities2, bodyIDs2);
    
    NTotal *= 2;
    NFirst *= 2;
    NSecond *= 2;
    NThird *= 2;
  #endif
  
  
  //Set the properties of the data set, it only is really used by process 0, which does the 
  //actual file I/O  
  tree->setDataSetProperties(NTotal, NFirst, NSecond, NThird);
  
  if(procId == 0)  
    cout << "Dataset particle information:\t" << NTotal << " " << NFirst << " " << NSecond << " " << NThird << std::endl;
  
  //Sanity check for standard plummer spheres
  double mass = 0, totalMass;
  for(unsigned int i=0; i < bodyPositions.size(); i++)
  {
    mass += bodyPositions[i].w;
  }
  
  tree->load_kernels();

#ifdef USE_MPI
  MPI_Reduce(&mass,&totalMass,1, MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);
#else
  totalMass = mass;
#endif
  
  if(procId == 0)   cerr << "Combined Mass: "  << totalMass << "\tNTotal: " << NTotal << std::endl;

  //Domain setup
  tree->createORB();
  
  //First distribute the initial particle distribution
  //over all available processes
  if(tree->nProcs > 1)
  {
    tree->createDistribution(&bodyPositions[0], (int)bodyPositions.size());  
  }

  //Print the domain division
  if(tree->nProcs > 1)
  {
    if(tree->procId == 0)
      for(int i = 0;i< tree->nProcs;i++)     
      {      
        cerr << "Domain: " << i << " " << tree->domainRLow[i].x << " " << tree->domainRLow[i].y << " " << tree->domainRLow[i].z << " " 
                                       << tree->domainRHigh[i].x << " " << tree->domainRHigh[i].y << " " << tree->domainRHigh[i].z <<endl;
      }
  }

  tree->mpiSync();    
  

  printf("Starting! \n");
  

  double t0 = tree->get_time();

  tree->localTree.setN((int)bodyPositions.size());
  tree->allocateParticleMemory(tree->localTree);
  
  //Load data onto the device
  for(uint i=0; i < bodyPositions.size(); i++)
  {
    tree->localTree.bodies_pos[i] = bodyPositions[i];
    tree->localTree.bodies_vel[i] = bodyVelocities[i];
    tree->localTree.bodies_ids[i] = bodyIDs[i];

    tree->localTree.bodies_Ppos[i] = bodyPositions[i];
    tree->localTree.bodies_Pvel[i] = bodyVelocities[i];
    tree->localTree.bodies_time[i] = make_float2(tree->get_t_current(), tree->get_t_current());
  }

  tree->localTree.bodies_time.h2d();
  tree->localTree.bodies_pos.h2d();
  tree->localTree.bodies_vel.h2d();
  tree->localTree.bodies_Ppos.h2d();
  tree->localTree.bodies_Pvel.h2d();
  tree->localTree.bodies_ids.h2d();
   
  //Distribute the particles so each process has particles
  //assigned to his domain
  if(nProcs > 1)
  {    
    double ttemp = tree->get_time();
    cout << "Before exchange tree has : " << tree->localTree.n << " particles \n" ;
    while(tree->exchange_particles_with_overflow_check(tree->localTree));
    cout << "After exchange tree has : " << tree->localTree.n << " particles \n" ;    
    //Send the new and old particles to the device
    tree->localTree.bodies_pos.h2d();
    tree->localTree.bodies_vel.h2d();
    tree->localTree.bodies_ids.h2d();
    tree->localTree.bodies_acc0.h2d();
    tree->localTree.bodies_acc1.h2d();
    tree->localTree.bodies_time.h2d();
    
    //This is only required the first time since we have no predict call before we build the tree
    //Every next call this is not required since the predict call fills the predicted positions
    tree->localTree.bodies_Ppos.copy(tree->localTree.bodies_pos,
                                     tree->localTree.bodies_pos.get_size());
    tree->localTree.bodies_Pvel.copy(tree->localTree.bodies_vel,
                                     tree->localTree.bodies_vel.get_size());
    
    printf("Initial exchange Took in total: %lg sec\n", tree->get_time()-ttemp);
  }
  
  //If required set the dust particles
  #ifdef USE_DUST
    if( (int)dustPositions.size() > 0)
    {
      fprintf(stderr, "Allocating dust properties for %d dust particles \n",
          (int)dustPositions.size());   
      tree->localTree.setNDust((int)dustPositions.size());
      tree->allocateDustMemory(tree->localTree);
      
      //Load dust data onto the device
      for(uint i=0; i < dustPositions.size(); i++)
      {
        tree->localTree.dust_pos[i] = dustPositions[i];
        tree->localTree.dust_vel[i] = dustVelocities[i];
        tree->localTree.dust_ids[i] = dustIDs[i];
      }

      tree->localTree.dust_pos.h2d();
      tree->localTree.dust_vel.h2d();
      tree->localTree.dust_ids.h2d();    
   }
  #endif //ifdef USE_DUST
	
  
  //Start the integration
#ifdef USE_OPENGL
  octree::IterationData idata;
  initAppRenderer(argc, argv, tree, idata, displayFPS);
  printf("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);  
#else
  tree->iterate(); 

  printf("Finished!!! Took in total: %lg sec\n", tree->get_time()-t0);
  
  logFile.close();

#ifdef USE_MPI
  MPI_Finalize();
#endif

  delete tree;
  tree = NULL;
#endif

  displayTimers();
  return 0;
}
