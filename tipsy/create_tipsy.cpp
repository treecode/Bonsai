#include <fstream>

#define MAXDIM 3
#define Real float

int main()
{
    double time = 0.0;
    int nbodies = 2;
    int ndim = 3;
    int nsph = 0;
    int ndark = 1;
    int nstar = 1;
    
    std::ofstream os("test.tipsy", std::ios::binary);
    os.write((char*) &time, sizeof(double));
    os.write((char*) &nbodies, sizeof(int));
    os.write((char*) &ndim, sizeof(int));
    os.write((char*) &nsph, sizeof(int));
    os.write((char*) &ndark, sizeof(int));
    os.write((char*) &nstar, sizeof(int));

    // dark particle:
    {
        Real mass = 0.00443291711;
        Real pos[MAXDIM] = {-0.0120858643, -0.0410266742, -0.00437123608};
        Real vel[MAXDIM] = {-1.60500276, -0.643298388, -0.367065489};
        Real eps = 0.206867486;
        int phi = 200000000;
    
        os.write((char*) &mass, sizeof(Real));
        os.write((char*) pos, sizeof(Real));
        os.write((char*) pos + 1, sizeof(Real));
        os.write((char*) pos + 2, sizeof(Real));
        os.write((char*) vel, sizeof(Real));
        os.write((char*) vel + 1, sizeof(Real));
        os.write((char*) vel + 2, sizeof(Real));
        os.write((char*) &eps, sizeof(Real));
        os.write((char*) &phi, sizeof(int));
    }
    
    // star particle:
    {
        Real mass = 0.00430987403;
        Real pos[MAXDIM] = {-0.0269202571, 0.0215147045, 0.00322919339};
        Real vel[MAXDIM] = {-0.0424188301, 0.809628189, -0.663158596};
        Real metals = 0.0;
        Real tform = 0.0;
        Real eps = 0.204935506;
        int phi = 100000000;

        os.write((char*) &mass, sizeof(Real));
        os.write((char*) pos, sizeof(Real));
        os.write((char*) pos + 1, sizeof(Real));
        os.write((char*) pos + 2, sizeof(Real));
        os.write((char*) vel, sizeof(Real));
        os.write((char*) vel + 1, sizeof(Real));
        os.write((char*) vel + 2, sizeof(Real));
        os.write((char*) &metals, sizeof(Real));
        os.write((char*) &tform, sizeof(Real));
        os.write((char*) &eps, sizeof(Real));
        os.write((char*) &phi, sizeof(int));
    }
}

