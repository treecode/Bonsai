War of galaxies
===============



Installation
------------

```
cmake \
  -DCMAKE_BUILD_TYPE=release \
  -DUSE_OPENGL=ON \
  -DUSE_MPI=OFF \
  -DUSE_CUB=OFF \
  -DUSE_THRUST=ON \
  -DWAR_OF_GALAXIES=ON \
  <source-path>
```

Usage
-----

Starting Bonsai with

```
./bonsai2_slowdust \
  -i <source-path>/war-of-galaxies/galaxy_types/dummy.tipsy \
  --war-of-galaxies <source-path>/tools/war-of-galaxies/galaxy_types/available
```

will show a empty simulation with a single dummy particle. 
