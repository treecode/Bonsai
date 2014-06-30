#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import subprocess 

import Image, ImageDraw, ImageFont


#Get the imaging library from: 
#http://www.pythonware.com/products/pil/
# python setup.py install

print "Usage: program folder-with-density-files"


fileNameBase    = str(sys.argv[1]) 

dirName  =  os.path.dirname (fileNameBase)
fileName =  os.path.basename(fileNameBase)

#Get the file list
snapShotFiles =   [x for x in os.listdir(dirName) if x.startswith(fileName)]

#print sorted(snapShotFiles)

snapShotFiles = [x for x in snapShotFiles if (not "eps" in x)]
snapShotFiles = [x for x in snapShotFiles if (not "jpeg" in x)]
snapShotFiles = [x for x in snapShotFiles if (not "bmp" in x)]
snapShotFiles = [x for x in snapShotFiles if (not ".avi" in x)]
snapShotFiles = [x for x in snapShotFiles if ("density" in x)]

snapShotFiles.sort(key=lambda x: float(x.split('-')[-1]))

print snapShotFiles


#For each file write the config, and launch the plotter
counter = 0
for x in snapShotFiles:
  #if(not "density.txt" in x):
  if(not "density-" in x):
    continue
  if("eps"  in x or "jpeg" in x or "bmp" in x):
      continue

  temp = x.split('-')
  time = float(temp[-1])
  timeGyr = time*9.7676470588235293
  #test =  '%.3f' % float(temp[-1])
  #test = test.rjust(9, '0')
  
  prePend = str(counter).rjust(6, '0')
 
  outputName = dirName + "/" + x
  outputName2 = dirName + "/" + prePend + "_" + x
    
  counter = counter + 1

  #Launch the density program
  com = './gen_image_voxel ' + outputName + " " + outputName2 + " color_map.bmp"
  p = subprocess.Popen(com, shell=True)
  p.wait()
  
  #Add text to the image
  
  # use a truetype font
  font = ImageFont.truetype("/usr/local/share/fonts/c/CenturyGothic.ttf", 30)
  
  bmpName = outputName2 + "-top.bmp"

  image = Image.open(bmpName)  
  draw  = ImageDraw.Draw(image)
  textOut = "T= " + str(round(timeGyr, 2)) + " Myr"
  draw.text((50, 950), textOut, font=font)
  del draw 
  image.save(bmpName,"BMP",quality=100)  
  
  #Rotate the front view by 90 degrees and add text
  bmpName = outputName2 + "-front.bmp"
  image = Image.open(bmpName)
  image = image.rotate(90)
  draw  = ImageDraw.Draw(image)
  textOut = "T= " + str(round(timeGyr, 2)) + " Myr"
  draw.text((50, 575), textOut, font=font)
  del draw 
  image.save(bmpName,"BMP",quality=100)  

#Done
print "To convert the bmps into a movie use: "
print "mencoder \"mf://*-top.bmp\" -mf fps=10 -o out_top.avi -ovc lavc -lavcopts vcodec=mpeg4"
print "mencoder \"mf://*-front.bmp\" -mf fps=10 -o out_front.avi -ovc lavc -lavcopts vcodec=mpeg4"
