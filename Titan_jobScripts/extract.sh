#!/bin/sh
awk '{if (NR==1) {dir=$1; app= $2}; if (NR==3) {tot= $2; grav= $4; gpu= $6}; if (NR==4) {print $4, $6, $8, tot, grav, gpu, dir,app} }'
