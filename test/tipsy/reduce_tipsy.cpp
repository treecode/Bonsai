#include <fstream>
#include <iostream>
#include <vector>
#include "tipsy.h"

int main(int argc, char** argv)
{
    if (argc != 4) {
    	std::cout << "USAGE: " << argv[0] << " <file-in> <file-out> <reduce-factor>" << std::endl;
        std::cerr << "Wrong number of arguments." << std::endl;
        return 1;
    }

    std::ifstream is(argv[1], std::ios::binary);
    std::ofstream os(argv[2], std::ios::binary);
    int reduce_factor = std::stoi(argv[3]);

    head h_in;
    is.read((char*) &h_in, sizeof(head));

    head h_out(h_in);
    h_out.nbodies = static_cast<int>(h_out.nbodies / reduce_factor) + (h_out.nbodies % reduce_factor == 0 ? 0 : 1);
    h_out.ndark = static_cast<int>(h_out.ndark / reduce_factor) + (h_out.ndark % reduce_factor == 0 ? 0 : 1);
    h_out.nstar = static_cast<int>(h_out.nstar / reduce_factor) + (h_out.nstar % reduce_factor == 0 ? 0 : 1);
    os.write((char*) &h_out, sizeof(head));

    dark_particle d;
    for (int i = 0; i != h_in.ndark; ++i)
    {
        is.read((char*) &d, sizeof(dark_particle));
        if (i % reduce_factor == 0) {
            d.mass *= reduce_factor;
            os.write((char*) &d, sizeof(dark_particle));
        }
    }

    star_particle s;
    for (int i = 0; i != h_in.nstar; ++i)
    {
        is.read((char*) &s, sizeof(star_particle));
        if (i % reduce_factor == 0) {
        	s.mass *= reduce_factor;
            os.write((char*) &s, sizeof(star_particle));
        }
    }
}
