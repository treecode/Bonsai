#!/bin/sh
ulimit -s unlimited
export CUDA_VISIBLE_DEVICES=1
mpirun hostname -s > /tmp/hostfile
nhost=`cat /tmp/hostfile|wc -l`
nprog=3
np=$(($nhost*$nprog))
echo "Nhost= $nhost  Np= $np"
mpirun  ./bonsai_clrshm $np
sleep 1
mpirun -hostfile /tmp/hostfile -np $np -loadbalance bash -c '
ulimit -s unlimited &&
vglrun -d :0.0 ./bonsai_driver  << EOF
  ./bonsai2_slowdust -f ./dataIn/snap__00510.0000.bonsai  -t 0.015625 -T 1000 --snapiter 1 --usempiio  --snapname data/snap_ --quickdump 0.125 --quickratio 0.2 --usempiio 
  ./bonsai_io
  ./bonsai_io -q
EOF
'
sleep 1
mpirun  ./bonsai_clrshm $np
