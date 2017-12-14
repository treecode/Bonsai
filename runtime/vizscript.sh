#!/bin/sh
ulimit -s unlimited
export CUDA_VISIBLE_DEVICES=1
mpirun hostname -s > /tmp/hostfile
nhost=`cat /tmp/hostfile|wc -l`
np=$(($nhost*2))
echo "Nhost= $nhost  Np= $np"
mpirun -np $nhost ./bonsai_clrshm $np
mpirun -hostfile /tmp/hostfile -np $np -loadbalance bash -c '
ulimit -s unlimited &&
vglrun -d :0.0 ./bonsai_driver  << EOF
  ./bonsai2_slowdust -f ./dataIn/snap__00510.0000.bonsai --reducebodies 1 -t 0.015625 -T 1000 --quickdump 0.125 --quickratio 0.2 --usempiio --noquicksync
  ./renderer -I --reduceDM 0 -d --noquicksync
EOF
'
sleep 1
mpirun -np $nhost ./bonsai_clrshm $np
