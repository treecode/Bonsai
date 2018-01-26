War of galaxies
===============

Installation
------------

Before the compilation the git submodule must be updated with

```
git submodule init
git submodule update
```

Compile with

```
cmake \
  -DCMAKE_BUILD_TYPE=release \
  -DUSE_OPENGL=ON \
  -DUSE_MPI=OFF \
  -DUSE_CUB=OFF \
  -DUSE_THRUST=ON \
  -DWAR_OF_GALAXIES=ON \
  <source-path>/runtime/
make -j <n>
```

Usage
-----

Starting Bonsai with

```
./bonsai2_slowdust \
  -i <source-path>/tools/war-of-galaxies/galaxy_types/dummy.tipsy \
  --war-of-galaxies <source-path>/tools/war-of-galaxies/galaxy_types/available
```

will show an empty simulation. In trues there is a single dummy particle at
position (0,0,10000) with zero mass, because Bonsai can not run without any
particles.

Actual galaxies are located or symlinked in
`<source-path>/tools/war-of-galaxies/galaxy_types/available`, with the naming
scheme identifying the galaxy numbering.

The release (insertion) or removal of galaxies will be controlled by json
commands. Therefore, you can start the python script:

```
<source-path>/tools/war-of-galaxies/client_json.py
```

Please find examples of json commands at

```
<source-path>/tools/war-of-galaxies/json_examples.txt
```
