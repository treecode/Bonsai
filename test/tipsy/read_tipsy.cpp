#include "tipsy.h"
#include <fstream>
#include <iostream>
#include <set>

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
    std::set<int> dark_phis;

    for (int i = 0; i != ndark; ++i)
    {
        dark_particle d;
        is.read((char*) &d, sizeof(d));
        dark_phis.insert(d.phi);
    }

    std::cout << dark_phis.size() << std::endl;
    for (auto p : dark_phis) std::cout << p << std::endl;

    // star particles
    std::set<int> star_phis;

    for (int i = 0; i != nstar; ++i)
    {
        star_particle s;
        is.read((char*) &s, sizeof(s));
        star_phis.insert(s.phi);
    }

    std::cout << star_phis.size() << std::endl;
    for (auto s : star_phis) std::cout << s << std::endl;
}

