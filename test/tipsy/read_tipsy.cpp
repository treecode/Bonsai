#include "tipsy.h"
#include <fstream>
#include <iostream>
#include <set>

int main(int argc, char** argv)
{
    std::ifstream is(argv[1], std::ios::binary);

    head h;
    is.read((char*) &h, sizeof(h));

    std::cout << "time = " << h.time << std::endl;    
    std::cout << "nbodies = " << h.nbodies << std::endl;    
    std::cout << "ndim = " << h.ndim << std::endl;    
    std::cout << "nsph = " << h.nsph << std::endl;    
    std::cout << "ndark = " << h.ndark << std::endl;    
    std::cout << "nstar= " << h.nstar << std::endl;    

    // dark particles
    std::set<int> dark_phis;

    for (int i = 0; i != h.ndark; ++i)
    {
        dark_particle d;
        is.read((char*) &d, sizeof(d));
        std::cout << d.mass << " "
                  << d.pos[0] << " "
                  << d.pos[1] << " "
                  << d.pos[2] << " "
                  << d.vel[0] << " "
                  << d.vel[1] << " "
                  << d.vel[2] << " "
                  << d.eps << " "
                  << d.phi << std::endl;
        dark_phis.insert(d.phi);
    }

    std::cout << "dark_phis.size() = " << dark_phis.size() << std::endl;
    //for (auto p : dark_phis) std::cout << p << std::endl;

    // star particles
    std::set<int> star_phis;

    for (int i = 0; i != h.nstar; ++i)
    {
        star_particle s;
        is.read((char*) &s, sizeof(s));
        std::cout << s.mass << " "
                  << s.pos[0] << " "
                  << s.pos[1] << " "
                  << s.pos[2] << " "
                  << s.vel[0] << " "
                  << s.vel[1] << " "
                  << s.vel[2] << " "
                  << s.metals << " "
                  << s.tform << " "
                  << s.eps << " "
                  << s.phi << std::endl;
        star_phis.insert(s.phi);
    }

    std::cout << "star_phis.size() = " << star_phis.size() << std::endl;
    //for (auto s : star_phis) std::cout << s << std::endl;
}

