#ifndef TIPSY_H
#define TIPSY_H

#define DIM 3
typedef float Real;

struct head {

    head(double time = 0.0, int nbodies = 0, int ndim = 0, int nsph = 0, int ndark = 0, int nstar = 0)
     : time(time), nbodies(nbodies), ndim(ndim), nsph(nsph), ndark(ndark), nstar(nstar)
    {}

    double time;
    int nbodies;
    int ndim;
    int nsph;
    int ndark;
    int nstar;
};

struct dark_particle
{
    dark_particle(float mass = 0.0, float posx = 0.0, float posy = 0.0, float posz = 0.0,
        float velx = 0.0, float vely = 0.0, float velz = 0.0, float eps = 0.0, int phi = 0)
      : mass(mass), eps(eps), phi(phi)
    {
        pos[0] = posx;
        pos[1] = posy;
        pos[2] = posz;
        vel[0] = velx;
        vel[1] = vely;
        vel[2] = velz;
    }

    float mass;
    float pos[DIM];
    float vel[DIM];
    float eps;
    int phi;
};

struct star_particle {

	star_particle(float mass = 0.0, float posx = 0.0, float posy = 0.0, float posz = 0.0,
		float velx = 0.0, float vely = 0.0, float velz = 0.0, float metals = 0.0,
	    float tform = 0.0, float eps = 0.0, int phi = 0)
	 : mass(mass), pos(), vel(), metals(metals), tform(tform), eps(eps), phi(phi)
    {
        pos[0] = posx;
        pos[1] = posy;
        pos[2] = posz;
        vel[0] = velx;
        vel[1] = vely;
        vel[2] = velz;
    }

    float mass;
    float pos[DIM];
    float vel[DIM];
    float metals;
    float tform;
    float eps;
    int phi;
};

#endif // TIPSY_H
