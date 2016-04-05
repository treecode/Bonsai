#include <fstream>
#include <iostream>

#define MAXDIM 3

typedef float Real;

struct dark_particle {
    Real mass;
    Real pos[MAXDIM];
    Real vel[MAXDIM];
    Real eps;
    int phi;
};

int main(int argc, char** argv)
{
    double time;
    int nbodies;
    int ndim;
    int nsph;
    int ndark;
    int nstar;

    std::cout << "file = " << argv[1] << std::endl;    

    std::ifstream is(argv[1], std::ios::binary);
    is.read((char*) &time, sizeof(double));
    is.read((char*) &nbodies, sizeof(int));
    is.read((char*) &ndim, sizeof(int));
    is.read((char*) &nsph, sizeof(int));
    is.read((char*) &ndark, sizeof(int));
    is.read((char*) &nstar, sizeof(int));

    std::cout << "time = " << time << std::endl;    
    std::cout << "nbodies = " << nbodies << std::endl;    
    std::cout << "ndim = " << ndim << std::endl;    
    std::cout << "nsph = " << nsph << std::endl;    
    std::cout << "ndark = " << ndark << std::endl;    
    std::cout << "nstar= " << nstar << std::endl;    

    // dark particles
    {
        dark_particle d;
    
        is.read((char*) &d, sizeof(d));

        std::cout << "mass = " << d.mass << std::endl;    
    }
}

