import sys
import os
import numpy as np
import matplotlib.pylab as plt
import math

def plot2D(x, y, z, zmin= None, zmax= None, xlim=None, ylim=None, nx = 200, ny = 200):

  if zmin == None:
    zmin = min(z)

  if zmax == None:
    zmax = max(z)

  for i in range(len(z)):
    z[i] = min(z[i], zmax)
    z[i] = max(z[i], zmin)

  xi = np.linspace(min(x), max(x), nx)
  yi = np.linspace(min(y), max(y), ny)
  zi = plt.mlab.griddata(x, y, z, xi, yi)

  plt.contourf(xi, yi, zi, 32, cmap=plt.cm.jet) #, norm=plt.Normalize(zmin, zmax))
  plt.contourf(xi, yi, zi, 32, norm=plt.Normalize(zmin, zmax))
  if 1 == 1:

    if (xlim == None):
      plt.xlim(-6.0, +6.0)
    else:
      plt.xlim(xlim[0], xlim[1]);

    if (ylim == None):
      plt.ylim(-6.0, +6.0)
    else:
      plt.ylim(ylim[0], ylim[1]);

  else:
    plt.xlim(-0.5, +0.5)
    plt.ylim(-0.5, +0.5)


  plt.colorbar()
  plt.show()

x = []
y = []
w = []

data = sys.stdin.readlines();
zcrd_min = -1
zcrd_max = +1
for line in data:

  wrd = line.split();

  xcrd = float(wrd[1]);
  ycrd = float(wrd[2]);
  zcrd = float(wrd[3]);
  wcrd = float(wrd[4]);

  if zcrd > zcrd_min and zcrd < zcrd_max:
    x.append(xcrd)
    y.append(ycrd)
    w.append(math.log10(wcrd))

print len(w)
plot2D(x,y,w, xlim=[-300, 300], ylim=[-300,300])





