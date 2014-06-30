#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import subprocess 


fileNameBase    = str(sys.argv[1]) 

dirName  =  os.path.dirname (fileNameBase)
fileName =  os.path.basename(fileNameBase)

#Get the file list
snapShotFiles =   [x for x in os.listdir(dirName) if x.startswith(fileName)]

#print sorted(snapShotFiles)

snapShotFiles = [x for x in snapShotFiles if (not "eps" in x)]
snapShotFiles = [x for x in snapShotFiles if (not "jpeg" in x)]
snapShotFiles = [x for x in snapShotFiles if (not "bmp" in x)]
snapShotFiles = [x for x in snapShotFiles if (not "avi" in x)]
snapShotFiles = [x for x in snapShotFiles if ("density" in x)]

snapShotFiles.sort(key=lambda x: float(x.split('-')[-1]))

print snapShotFiles


#For each file write the config, and launch the plotter
counter = 0
for x in snapShotFiles:
  
  #if(not "density.txt" in x):
  if(not "density-" in x):
    continue

  countOut = str(counter).rjust(6, '0')
  
  #First convert binary to ascii
  
  com = './convDensBinToAscii ' + dirName + '/' + x + ' /tmp/tempC2A.txt'
  print com
  p = subprocess.Popen(com, shell=True)
  p.wait()

  #Read the time
  
  fileIn = open("/tmp/tempC2A.txt")
  line = fileIn.readline()
  time = line.split()[-1]
  fileIn.close()
  
  print 'Snapshot time= ', time
    
  fileOut = open("temp.txt", "w")
  
  #outputName = dirName + "/" + x
  #outputName = outputName[:-4] + ".eps"
  outputName = dirName + "/" + str(countOut) + ".eps"
  
  fileOut.write("set terminal postscript enhanced color\n") 
  fileOut.write("set output \""+outputName+"\"\n")
  
  fileOut.write("set title 'T=" + time + "'\n")
  fileOut.write("set size square\n")
  fileOut.write("set xrange[0:1024]\n")
  fileOut.write("set yrange[0:1024]\n")
  #fileOut.write("set cbrange[-2:5]\n")
  fileName = '/tmp/tempC2A.txt'
  fileOut.write("p '" + fileName + "' u 1:2:($3 > 0 ? log10($3) : 0 ) with image notitle\n\n")
  
 
  #fileOut.write("p '" + fileName + "' binary skip=64 format='%float%float%float%float' array=(1024,1024) u ($1 > 0 ? log10($1) : 0 )  w image\n\n")  
  

  outputName = dirName + "/" + str(countOut) + ".jpeg"
  fileOut.write("set terminal jpeg\n") 
  fileOut.write("set output \""+outputName+"\"\n")  
  fileOut.write("set title 'T=" + time + "'\n")
  fileOut.write("set size square\n")
  fileOut.write("set xrange[0:1024]\n")
  fileOut.write("set yrange[0:1024]\n")
  #fileOut.write("set cbrange[-1:4]\n")
  fileOut.write("set palette color\n")
  fileOut.write("set pm3d map\n")
  fileOut.write("set palette defined\n")
  fileOut.write("set palette model RGB\n")

  fileName = '/tmp/tempC2A.txt'
  fileOut.write("p '" + fileName + "' u 1:2:($3 > 0 ? log10($3) : 0 ) with image notitle\n\n")
  

  #fileOut.write("p '" + fileName + "' binary skip=64 format='%float%float%float%float%float' array=(1024,1024) u ($1 > 0 ? log10($1) : 0 )  w image\n\n")
  fileOut.close()
  counter = counter + 1

  #Launch gnuplot
  com = 'gnuplot temp.txt'
  print com
  p = subprocess.Popen(com, shell=True)
  p.wait()

com = "rm -f /tmp/tempC2A.txt"
p = subprocess.Popen(com, shell=True)
p.wait()

print "To make a movie use: "
print "mencoder \"mf://*jpeg\" -mf fps=10 -o out_dens.avi -ovc lavc -lavcopts vcodec=mpeg4 "


#Done
