#!/bin/python

import sys
import commands

def dPrint(line):
  print(line);
  return 0;

def execCommand(command):
  result = commands.getstatusoutput(command)
  if result[0] != 0:
    print("Error occured while executing system command %s " % command);
    print("status= %d:" % result[0])
    print("error = %s:" % result[1])
    sys.exit(-1)

  return result


def countNumProcs(prefix):
  firstFile = prefix+"00000.0000-*"
  command = "ls %s" % (firstFile)
  dPrint(command)

  result = execCommand(command)
  status = result[0]
  output = result[1];

  if status != 0:
    print("Please check prefix, currently using");
    print("status= %d: error= %s" % (status, output))
    sys.exit(-1)
  
  output = result[1].split('\n')

  return len(output)

def countNumSnapshots(prefix):
  firstFile = prefix+"*-0"
  command = "ls %s" % (firstFile)
  dPrint(command)

  result = execCommand(command)
  status = result[0]
  output = result[1];
  
  output = result[1].replace("-0","").replace(prefix,"").split('\n')
  dPrint(output)

  return output


def usage(appName, exitCode):
  print "Usage: "
  print "%s path_to_snapshots prefix_of_a_snapshot" % appName
  print " --- Example: "
  print "%s MilkyWayPD MW"
  sys.exit(exitCode)

    
  

if __name__ == "__main__":

  if len(sys.argv) < 3:
    usage(sys.argv[0], -1)

  path=sys.argv[1]
  prefix0=sys.argv[2];

  prefix = path + "/" + prefix0 + "_";

  dPrint(prefix0);
  dPrint(prefix);

  nProc = countNumProcs(prefix)
  print "Found snapshots for %d procs " % nProc

  snapshots = countNumSnapshots(prefix)
  print "Found %d snapshots per proc " % len(snapshots)

  mkFolder = "mkdir %s" % path +  "/" + prefix0
  result = execCommand(mkFolder)


  for snap in snapshots:
    destFolder = path + "/" + prefix0 + "/" + snap
    mkFolder = "mkdir %s" % destFolder
    print mkFolder
    execCommand(mkFolder)
    for proc in range(0,nProc):
      srcFile = prefix+snap+("-%d" % proc)
      dstFile = destFolder + "/" + ("%06d" % proc)
      mvFile = "mv %s %s" % (srcFile, dstFile)
      print mvFile
      execCommand(mvFile)





#for arg in sys.argv:
#  print arg;
