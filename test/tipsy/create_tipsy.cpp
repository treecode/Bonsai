#include <iostream>
#include <fstream>
#include "tipsy.h"

int main()
{
    head h(0.0, 1, DIM, 0, 1, 0);

    std::ofstream os("test.tipsy", std::ios::binary);
    os.write((char*) &h, sizeof(h));

    dark_particle d(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 200000000);
    os.write((char*) &d, sizeof(d));

    //dark_particle d(0.00443292, -0.0120859, -0.0410267, -0.00437124, -1.605, -0.643298, -0.367065, 0.206867, 200000000);
    //os.write((char*) &d, sizeof(d));

    //star_particle s(0.00430987, -0.0269203, 0.0215147, 0.00322919, -0.0424188, 0.809628, -0.663159, 0, 0, 0.204936, 100000000);
    //os.write((char*) &s, sizeof(s));
}
