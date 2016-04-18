#include <fstream>
#include <iostream>
#include <Eigen/Eigenvalues>

#define DIM 3

struct head {

	head() : time(0.0), nbodies(0), ndim(DIM), nsph(0), ndark(0), nstar(0) {}

    double time;
    int nbodies;
    int ndim;
    int nsph;
    int ndark;
    int nstar;
};

struct dark_particle {
    float mass;
    float pos[DIM];
    float vel[DIM];
    float eps;
    int phi;
};

struct star_particle {
    float mass;
    float pos[DIM];
    float vel[DIM];
    float metals;
    float tform;
    float eps;
    int phi;
};

int main(int argc, char** argv)
{
    if (argc != 2) exit(1);
    std::cout << "file = " << argv[1] << std::endl;    

    std::ifstream is(argv[1], std::ios::binary);

    head h;
    is.read((char*) &h, sizeof(h));
    int after_header = is.tellg();

    std::cout << "time = " << h.time << std::endl;
    std::cout << "nbodies = " << h.nbodies << std::endl;
    std::cout << "ndim = " << h.ndim << std::endl;
    std::cout << "nsph = " << h.nsph << std::endl;
    std::cout << "ndark = " << h.ndark << std::endl;
    std::cout << "nstar = " << h.nstar << std::endl;

    double x,y,z;
    double S[DIM] = {0.0};
//    Eigen::Matrix3d I;
//    Eigen::Vector3d S;
//    I.setZero();
//    S.setZero();

    dark_particle d;
    for (int i = 0; i < h.ndark; ++i)
    {
        is.read((char*) &d, sizeof(d));

        x = d.pos[0];
        y = d.pos[1];
        z = d.pos[2];

//        I(0,0) += y*y + z*z;
//        I(0,1) += -x*y;
//        I(0,2) += -x*z;
//        I(1,0) += -y*x;
//        I(1,1) += x*x + z*z;
//        I(1,2) += -y*z;
//        I(2,0) += -z*x;
//        I(2,1) += -z*y;
//        I(2,2) += x*x + y*y;

        S[0] += x;
        S[1] += y;
        S[2] += z;
    }

    star_particle s;
    for (int i = 0; i < h.nstar; ++i)
    {
        is.read((char*) &s, sizeof(s));

        x = s.pos[0];
        y = s.pos[1];
        z = s.pos[2];

//        I(0,0) += y*y + z*z;
//        I(0,1) += -x*y;
//        I(0,2) += -x*z;
//        I(1,0) += -y*x;
//        I(1,1) += x*x + z*z;
//        I(1,2) += -y*z;
//        I(2,0) += -z*x;
//        I(2,1) += -z*y;
//        I(2,2) += x*x + y*y;

        S[0] += x;
        S[1] += y;
        S[2] += z;
    }

    is.seekg(after_header);

//    Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(I);
//    std::cout << eigen_solver.eigenvalues() << std::endl;
//    std::cout << eigen_solver.eigenvectors().col(0) << std::endl;
//    std::cout << eigen_solver.eigenvectors().col(1) << std::endl;
//    std::cout << eigen_solver.eigenvectors().col(2) << std::endl;

    std::cout << "center = " << S[0] << std::endl;
    S[0] /= h.nbodies;
    S[1] /= h.nbodies;
    S[2] /= h.nbodies;
    std::cout << "center = " << S[0] << std::endl;

    head h1, h2;

    for (int i = 0; i < h.ndark; ++i)
    {
        is.read((char*) &d, sizeof(d));
        if (d.pos[0] > S[0]) ++h1.ndark;
        else ++h2.ndark;
    }

    for (int i = 0; i < h.nstar; ++i)
    {
        is.read((char*) &s, sizeof(s));
        if (s.pos[0] > S[0]) ++h1.nstar;
        else ++h2.nstar;
    }

    h1.nbodies = h1.ndark + h1.nstar;
    h2.nbodies = h2.ndark + h2.nstar;

    std::cout << "h1.nbodies = " << h1.nbodies << std::endl;
    std::cout << "h2.nbodies = " << h2.nbodies << std::endl;

    std::ofstream os1("1.tipsy", std::ios::binary);
    std::ofstream os2("2.tipsy", std::ios::binary);

    os1.write((char*) &h1, sizeof(h1));
    os2.write((char*) &h2, sizeof(h2));

    is.seekg(after_header);

    for (int i = 0; i < h.ndark; ++i)
    {
        is.read((char*) &d, sizeof(d));
        if (d.pos[0] > S[0]) os1.write((char*) &d, sizeof(d));
        else os2.write((char*) &d, sizeof(d));
    }

    for (int i = 0; i < h.nstar; ++i)
    {
        is.read((char*) &s, sizeof(s));
        if (s.pos[0] > S[0]) os1.write((char*) &s, sizeof(s));
        else os2.write((char*) &s, sizeof(s));
    }

}

