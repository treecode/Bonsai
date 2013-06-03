#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <fstream>
#include "vector3.h"

struct Plummer{
	std::vector<double> mass;
	std::vector<dvec3> pos, vel;
	Plummer(
			unsigned long n, 
			const int procId,
			unsigned int  seed = 19810614, 
			const char   *filename = "plummer.dat") 
		: mass(n), pos(n), vel(n)
		{
#if 0
			{
				std::ifstream ifs(filename);
				if(!ifs.fail()){
					unsigned long ntmp, stmp;
					ifs.read((char *)&ntmp, sizeof(unsigned long));
					ifs.read((char *)&stmp, sizeof(unsigned long));
					if(n == ntmp  && seed == stmp){
						ifs.read((char *)&mass[0], n*sizeof(double));
						ifs.read((char *)& pos[0], n*sizeof(dvec3));
						ifs.read((char *)& vel[0], n*sizeof(dvec3));
						if(!ifs.fail()){
							fprintf(stdout, "plummer : read from %s\n", filename);
						}
						return;
					}
				}
			}
#endif
			my_srand(seed);
			unsigned long i = 0;
			while(i < n){
				const double X1 = my_rand();
				const double X2 = my_rand();
				const double X3 = my_rand();
				const double c = (pow(X1,-2.0/3.0) - 1.0);
        if (c < 0)
        {
          fprintf(stderr, "%d : THIS MAKES [c=%e] NaN, continue\n", (int)i, c);
          continue;
        }
				const double R = 1.0/sqrt( c );
        #ifdef WIN32
        if(R != R) //nan compared with nan is false
        {
          fprintf(stderr, "%d : Nan detected R [c=%e] NaN, continue\n", (int)i, c);
          continue;
        }
        #else
        if(std::isnan(R))
        {
          fprintf(stderr, "%d : Nan detected R [c=%e] NaN, continue\n", (int)i, c);
          continue;
        }
        #endif

				if(R < 100.0) {
					double Z = (1.0 - 2.0*X2)*R;
					if((R*R - Z*Z) < 0.0){
						fprintf(stderr, "%d : THIS MAKES NaN, continue\n", (int)i);
						continue;
					}
					double X = sqrt(R*R - Z*Z) * cos(2.0*M_PI*X3);
					double Y = sqrt(R*R - Z*Z) * sin(2.0*M_PI*X3);
					
					if(!(X == X)){
						fprintf(stderr, "NaN detectd X[%d], continue\n", (int)i);
						continue;
					}
					if(!(Y == Y)){
						fprintf(stderr, "NaN detectd Y[%d], continue\n", (int)i);
						continue;
					}

					const double Ve = sqrt(2.0)*pow( (1.0 + R*R), -0.25 );

					double X4 = 0.0; 
					double X5 = 0.0;

					while( 0.1*X5 >= X4*X4*pow( (1.0-X4*X4), 3.5) ) {
						X4 = my_rand(); X5 = my_rand(); 
					} 

					const double V = Ve*X4;

					const double X6 = my_rand();
					const double X7 = my_rand();

					double Vz = (1.0 - 2.0*X6)*V;
					double Vx = sqrt(V*V - Vz*Vz) * cos(2.0*M_PI*X7);
					double Vy = sqrt(V*V - Vz*Vz) * sin(2.0*M_PI*X7);

					const double conv = 3.0*M_PI/16.0;
					X *= conv; Y *= conv; Z *= conv;    
					Vx /= sqrt(conv); Vy /= sqrt(conv); Vz /= sqrt(conv);

					double M = 1.0;
					mass[i] = M / (double)n;

					pos[i][0] = X;
					pos[i][1] = Y;
					pos[i][2] = Z;

					vel[i][0] = Vx;
					vel[i][1] = Vy;
					vel[i][2] = Vz;

					/*
					   tmp_i = ldiv(i, 256);
					   if(tmp_i.rem == 0) printf("i = %d \n", i);
					   */

					ldiv_t tmp_i = ldiv(i, n/64);

					if(tmp_i.rem == 0 && procId == 0) {
						printf(".");
						fflush(stdout); 
					}
					i++; 
				}		
			} // while (i<n)
			double mcm = 0.0;

			double xcm[3], vcm[3];
			for(int k=0;k<3;k++) {
				xcm[k] = 0.0; vcm[k] = 0.0;
			} /* k */

			for(i=0; i<n; i++) {
				mcm += mass[i];
				for(int k=0;k<3;k++) {
					xcm[k] += mass[i] * pos[i][k]; 
					vcm[k] += mass[i] * vel[i][k]; 
				} /* k */ 
			}  /* i */
			for(int k=0;k<3;k++) {
				xcm[k] /= mcm; vcm[k] /= mcm;
			} /* k */

			for(i=0; i<n; i++) {
				for(int k=0;k<3;k++) {
					pos[i][k] -= xcm[k]; 
					vel[i][k] -= vcm[k]; 
				} /* k */ 
			} /* i */ 
			if (procId == 0)
				printf("\n");
#if 0
			{
				std::ofstream ofs(filename);
				if(!ofs.fail()){
					unsigned long ntmp = n;
					unsigned long stmp = seed;
					ofs.write((char *)&ntmp, sizeof(unsigned long));
					ofs.write((char *)&stmp, sizeof(unsigned long));
					ofs.write((char *)&mass[0], n*sizeof(double));
					ofs.write((char *)& pos[0], n*sizeof(dvec3));
					ofs.write((char *)& vel[0], n*sizeof(dvec3));
					if(!ofs.fail()){
						fprintf(stdout, "plummer : wrote to %s\n", filename);
					}
				}
			}
#endif
		}
#if 0
	static void my_srand(const unsigned int seed){
		srand(seed);
	}
	static double my_rand(void) {
		return rand()/(1. + RAND_MAX);
	}
#else
	static void my_srand(const int seed){
		srand48(seed);
	}
	static double my_rand(void) {
		return drand48();
	}
#endif
};
