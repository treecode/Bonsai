#ifndef TIPSYDEFS_H
#define TIPSYDEFS_H

#define MAXDIM 3
#define forever for(;;)

typedef float Real;

struct gas_particle {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real rho;
    Real temp;
    Real hsmooth;
    Real metals ;
    Real phi ;
} ;

//struct gas_particle *gas_particles;

struct dark_particle {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real eps;
    int phi ;
public:
  int  getID() const {return phi;}
  void setID(int ID) {  phi = ID; }
} ;

struct star_particle {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real metals ;
    Real tform ;
    Real eps;
    int phi ;
public:
  int  getID() const {return phi;}
  void setID(int ID) {  phi = ID; }
} ;


//V2 structures use 64 bit integers for particle storage
//otherwise they take up the same space for compatibility

struct dark_particleV2 {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
  private:
    int _ID[2]; //replaces phi and eps
  public:
    unsigned long long getID() const 
    {
      union
      {
        int ID[2];
        unsigned long long IDl;
      } t;
      t.ID[0] = _ID[0];
      t.ID[1] = _ID[1];
      return t.IDl;
    }
    void setID(unsigned  long long ID) 
    { 
      union
      {
        int ID[2];
        unsigned long long IDl;
      } t;
      t.IDl = ID;
      _ID[0] = t.ID[0];
      _ID[1] = t.ID[1];
    }
    int getID_V1() const {return _ID[1];}
//    Real eps;
} ;

//JB: Use orignal structure otherwise reading of 
//particle ID fails
struct star_particleV2 {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real metals ;
    Real tform ;
  private:
    int _ID[2]; //replaces phi and eps
  public:
    unsigned long long getID() const 
    {
      union
      {
        int ID[2];
        unsigned long long IDl;
      } t;
      t.ID[0] = _ID[0];
      t.ID[1] = _ID[1];
      return t.IDl;
    }
    void setID(unsigned  long long ID) 
    { 
      union
      {
        int ID[2];
        unsigned long long IDl;
      } t;
      t.IDl = ID;
      _ID[0] = t.ID[0];
      _ID[1] = t.ID[1];
    }
    int getID_V1() const {return _ID[1];}
//    Real eps;
} ;

#if 0
struct star_particleV2 : public dark_particleV2 {
    Real metals ;
    Real tform ;
private:
public:
} ;
#endif

struct dump {
    double time ;
    int nbodies ;
    int ndim ;
    int nsph ;
    int ndark ;
    int nstar ;
} ;

struct dumpV2 {
    double time ;
    int nbodies ;
    int ndim ;
    int nsph ;
    int ndark ;
    int nstar ;
    int version;
} ;


typedef struct dump header ;

#endif






