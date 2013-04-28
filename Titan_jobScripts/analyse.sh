#!/bin/sh
filename=$1
ninter=`cat $filename|grep direct|awk '{if ($8 >= 31) {dir+= $10; apprx += $12; print dir/1e12," ", apprx/1e12}}'|tail -n 1`
nflops=`cat $filename|grep direct|awk '{if ($8 >= 31) {dir+= $10*23; apprx += $12*65; print dir/1e12," ", apprx/1e12}}'|tail -n 1`
time=`cat $filename|grep TOTAL|tail -n 1|awk '{print "TOTAL=", $4, " GRAV=", $6, " GPU=", $8}'`
pflop=`echo $nflops $time | awk '{print "Performance [TFlop/s]: GPU= ", ($1+$2)/$8, " GPU+LET=", ($1+$2)/$6, " Effective=", ($1+$2)/$4}'`

echo $ninter
echo $nflops
echo $time
echo $pflop

