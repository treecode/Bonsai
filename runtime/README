CUDA Runtime API version of the Bonsai


To quickly test the single GPU code:
```
cmake -DUSE_CUB=0 -DUSE_DUST=0 -DUSE_MPI=0 -DUSE_MPIMT=0
# Optionally run ccmake . to modify the CMake settings
make
ulimit -s unlimited
./bonsai2_slowdust -i model3_child_compact.tipsy -T 10
```

It should give something like:
iter=160 : time= 10  Etot= -417.1994462  Ekin= 430.589   Epot= -847.788 : de= -0.00416569 ( 0.00416569 ) d(de)= -0 ( 0.00031664 ) t_sim=  0.545423 sec


To generate MilkyWay galaxy you need galactics.parallel fork of John Dubinsky galactics code. Once you have tarball do the following in this folder:
1) tar xzf galactics.parallel.tar.gz
2) cd galactic.parallel/src
3) make -f Makefile.[ifort/gnu]  use ifort if you have Intel Fortran compiler (3x faster than gfortran), otherwise use Makefile.gnu
4) cp libgengalaxy.a ../../
5) cd ../../
6)
   cmake -DUSE_GALACTICS=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=mpicxx
  or
   cmake -DUSE_GALACTICS_IFORT=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_COMPILER=mpicxx

7) (not needed, but kept here for references) ccmake .
     change USE_GALACTICS        from OFF to ON
     change USE_GALACTICS_IFORT  from OFF to ON if you use Intel Fortran compiler
   type  c, c, g
8) make -j

To run galactic, you need the DF data, these are located in galactics_df_mw. Copy them to the exection folder and unzip
1) cd my_exec_folder
2) cp path_to_bonsai/runtime/galactics_mw_df/*gz .
3) gzip -vd *.gz
4) mpirun -np 4 path_to_bonsai/runtime/bonsai2_slowdust --milkyway 500000   -o 0.4 -T 10 -r 1 --eps 0.1  -t 0.001 2>&1 | tee log_MWa

Units:
  basic: distance  = 1 kpc, speed= 100 km/s
  derived: mass= distance*speed^2/G= 2.324876e9 Msun, time= sqrt(distance^3/G/mass) = 9.778145 Myr

Enjoy the results!


