#include <fstream>
#include "tipsy.h"

int main()
{
	head h(0.0, 1, DIM, 0, 0, 1);
    
    std::ofstream os("test.tipsy", std::ios::binary);
    os.write((char*) &h, sizeof(h));

    star_particle s(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 100000000);
	os.write((char*) &s, sizeof(s));
}
