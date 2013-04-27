#include "main.h"

extern void mysrand(const long seed);

void gen_halo(int nptcl, int myseed, float *buffer, int verbose)
{
	int i, j, k, nobj=10000;
	int seed= -123;
	float q=0.9, rtrunc=5.0;
	float rhoran, massapp;
	float x, y, z, vx, vy, v, v2, R, vmax, vmax2;
	float phi, cph, sph, cth, sth, vR, vp, vz;
	float E, Lz, rad, rad2;
	float f0, frand, fmax, psi;
	float dfhalo_(), halodens_(), pot_();
	float ran1();
	float dr, rhomax1;
	float t, mass;
	float u1, v1, u1max, v1max;
	float xcm, ycm, zcm, vxcm, vycm, vzcm;
	float zip = 0.0;
	float rhomax=0.15, rhomin, rhotst;
	float rhocur, rcyl;
	float stream=0.5, massfrac=1.0;
	float psic_halo_n, psi_n;
	float fcut, fac, ebh, psic_bh;
	float dfmax, emax, psibh;
	float b, c, d, int_const, sig2, aeff;
	float v02, vb2, coef1, coef2;
	int icofm=1;
	int jj;
	char harmfile[80];

	strcpy(harmfile,"dbh.dat");
  stream = 0.5;
  nobj = nptcl;
  seed = myseed;
  mysrand(myseed);
  icofm =1;

  if (nobj <= 0)
  {
    fquery("Enter streaming fraction",&stream);
    iquery("Enter the number of particles",&nobj);
    iquery("Enter negative integer seed",&seed);
    iquery("Center particles (0=no, 1=yes)",&icofm);
  }
  /*	cquery("Enter harmonics file",harmfile); */

  readmassrad(); /* reads in mass and truncation radius of components */
  readharmfile_(harmfile,&gparam,&cparam,&bparam,&flags);
  readdenspsihalo_();

  c = gparam.c; v0 = gparam.v0; a = gparam.a; 
  cbulge = gparam.cbulge; v0bulge=gparam.v0bulge;
  psi0 = gparam.psi0; haloconst = gparam.haloconst;

  psic_halo = cparam.psic_halo;
  bhmass = bparam.bhmass;
  rtrunc = haloedge;
  mass = halomass/nobj;
  sig2 = -psi0;
  u1max = rtrunc;
  v1max = 0.5*M_PI;

  r = (phase *) calloc(nobj,sizeof(phase));

  /* `Find maximum of rho*r^2 */
  dr = rtrunc/100;
  rhomax1 = 0.0;
  psic_halo_n = -psic_halo;
  if(bhmass!=0) {
    psic_bh = bhmass/rtrunc;
    effectivescalelength_(&aeff);
    d = bhmass/aeff;
    int_const = d/(1-psic_halo_n/sig2)-psic_bh; 
  }else {
    d = 0.;
    int_const = 0.;
    psic_bh = 0.;
    aeff = 0.;
  }
  for(i=0; i<100; i++) {
    float z, rcyl, rhocur; 
    rcyl = i*dr; 
    z = 0; 
    rhocur = halodens_(&rcyl,&z); 
    /*
       fprintf(stdout,"%g %g %g\n",rcyl,rhocur,rhocur*rcyl*rcyl);
       */
    rhocur *= (rcyl*rcyl); 
    if( rhocur > rhomax1  )
      rhomax1 = rhocur; 
  } 
  if (verbose)
    fprintf(stderr,"rhomax on R-axis is %g\n",rhomax1); 
  j = 0; 
  for(i=0; i<100; i++) { 
    rcyl = 0; 
    z = i*dr; 
    rhocur = halodens_(&rcyl,&z);
    rhocur *= (z*z); 
    if( rhocur > rhomax1  )
      rhomax1 = rhocur; 
  } 
  if (verbose)
    fprintf(stderr,"rhomax1 = %g\n",rhomax1); 
  rhomax = 1.5*rhomax1; 
  rhomin = 1.0e-10*rhomax; 
  fcut = dfhalo_(&psic_halo_n); 
  dfmaximumhalo_(&emax,&dfmax); 
  if (verbose)
    fprintf(stderr,"Calculating halo positions and velocities\n");
  for(i=0; i<nobj;) {
    u1 = u1max*ran1(&seed);
    v1 = 2.0*v1max*(ran1(&seed) - 0.5);
    R = u1;
    z = R*tan(v1);

    rhotst = halodens_(&R,&z);
    rhotst = rhotst*(R*R + z*z);
    j++;
    if( rhotst < rhomin )
      continue;

    rhoran = (rhomax - rhomin)*ran1(&seed);
    if( rhoran > rhotst )
      continue;
    phi = 2.0*M_PI*ran1(&seed);
    x = R*cos(phi);
    y = R*sin(phi);
    psi = pot_(&R,&z);
    psi_n = -psi;
    if (psi > psic_halo)
      continue;
    psibh = bhmass/sqrt(R*R+z*z);
    vmax2 = 2*(psi_n+psibh-psic_halo_n-psic_bh);
    vmax = sqrt(vmax2);
    if( psi_n > emax ) {
      fmax = dfmax;	
    }
    else {
      dfcorrectionhalo_(&psi_n,&fac);
      fmax = (dfhalo_(&psi_n)-fcut)/fac;
    } 
    f0 = 0.0; frand = 1.0; /* dummy starters */
    j = 0;
    while( frand > f0 ) {
      v2 = 1.1*vmax2;
      while( v2 > vmax2 ) {
        vR = 2*vmax*(ran1(&seed) - 0.5);
        vp = 2*vmax*(ran1(&seed) - 0.5);
        vz = 2*vmax*(ran1(&seed) - 0.5);
        v2 = vR*vR + vp*vp + vz*vz;
      }
      ebh = psi_n + psibh - 0.5*v2;
      b = sig2 + ebh + int_const;
      c = sig2 - ebh - int_const; 
      E = 0.5*(b-sqrt(c*c+4.*d));
      dfcorrectionhalo_(&E,&fac);
      f0 = (dfhalo_(&E)-fcut)/fac;
      frand = fmax*ran1(&seed);
      j++;
    }

    if( ran1(&seed) < stream )
      vp = fabs(vp);
    else
      vp = -fabs(vp);

    cph = x/R; sph = y/R;
    vx = vR*cph - vp*sph;
    vy = vR*sph + vp*cph;

    r[i].x = (float) x;
    r[i].y = (float) y;
    r[i].z = (float) z;
    r[i].vx = (float)vx;
    r[i].vy = (float)vy;
    r[i].vz = (float)vz;
    i++;
    if (verbose)
      if( i % 1000 == 0 ) fprintf(stderr,".");
  }
  if (verbose)
    fprintf(stderr,"\n");

  if( icofm ) {
    xcm = ycm =zcm = vxcm =vycm =vzcm = 0;
    for(i=0; i<nobj; i++) {
      xcm += r[i].x;
      ycm += r[i].y;
      zcm += r[i].z;
      vxcm += r[i].vx;
      vycm += r[i].vy;
      vzcm += r[i].vz;
    }
    xcm /= nobj; ycm /=nobj; zcm /= nobj;
    vxcm /= nobj; vycm /=nobj; vzcm /= nobj;
    if (verbose)
      fprintf(stderr,"%f %f %f %f",xcm,zcm,vycm,vzcm);
    for(i=0; i<nobj; i++) {
      r[i].x -= xcm;
      r[i].y -= ycm;
      r[i].z -= zcm;
      r[i].vx -= vxcm;
      r[i].vy -= vycm;
      r[i].vz -= vzcm;
    }
  }
  if (buffer == NULL)
  {
    t = 0.0;
#ifdef ASCII
    fprintf(stdout,"%d\n",nobj);
    for(i=0; i<nobj; i++) {
      fprintf(stdout,"% 15.7e % 15.7e % 15.7e % 15.7e % 15.7e % 15.7e % 15.7e\n", mass, r[i].x, r[i].y, r[i].z, r[i].vx, r[i].vy, r[i].vz);
    }
#else
    for(i=0; i<nobj; i++) {
      fwrite(&mass,sizeof(float),1,stdout);
      fwrite(r+i,sizeof(phase),1,stdout);
    } 
#endif
  }
  else
  {
    const int min_halo = 200000000;
    const int max_halo = 400000000;
    int NEL = 8;
    int i,pc;
    for (i = 0, pc= 0; i < nobj; i++, pc += NEL)
    {
      *((int*)&buffer[pc]) = min_halo + (i%(max_halo-min_halo));
      buffer[pc+1] = mass;
      buffer[pc+2] = r[i].x;
      buffer[pc+3] = r[i].y;
      buffer[pc+4] = r[i].z;
      buffer[pc+5] = r[i].vx;
      buffer[pc+6] = r[i].vy;
      buffer[pc+7] = r[i].vz;
    }
  }
}

