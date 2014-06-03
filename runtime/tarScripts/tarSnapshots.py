#!/bin/python

newVolumeScript=[
"#! /bin/bash",
"# For this script it's advisable to use a shell, such as Bash,",
"# that supports a TAR_FD value greater than 9.",
"echo Preparing volume $TAR_VOLUME of $TAR_ARCHIVE.",
"name=`expr $TAR_ARCHIVE : '\(.*\)-.*'`",
"case $TAR_SUBCOMMAND in",
"-c)       ;;",
"-d|-x|-t) test -r ${name:-$TAR_ARCHIVE}-$TAR_VOLUME || exit 1",
"          ;;",
"*)        exit 1",
"esac",
"echo ${name:-$TAR_ARCHIVE}-$TAR_VOLUME >&$TAR_FD"]

import sys
import commands
import os

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


def countNumSnapshots(prefix):
  firstFile = prefix
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
  print "%s path_to_snapshots split_size in [GB]  " % appName
  print " -- "
  print "path_to_snapshot should be folder structure produced by sortSnapshots.py"
  print " --- Example:"
  print "%s MilkyWayPD/MW 200"
  sys.exit(exitCode)

  

if __name__ == "__main__":
  script= ""
  for line in newVolumeScript:
    script += line+"\n"

  if len(sys.argv) < 3:
    usage(sys.argv[0], -1);
   
  splitSize = int(sys.argv[2]);
  if (splitSize <= 0):
    print("Error: Unkown split_size= %d " % splitSize)
    print("Argument: %s " % sys.argv[2]);
    usage(sys.argv[0], -1);

  pwd = os.getcwd()
  prefix  = sys.argv[1];
  prefixT = prefix+".tarball";
  dPrint(prefix);
  dPrint(prefixT);

  snapshots = countNumSnapshots(prefix)
  print "Found %d snapshots per proc " % len(snapshots)

  mkFolder = "mkdir %s" % prefixT
  result = execCommand(mkFolder)
  
  f = open("%s/newVolumeScript.sh" % prefixT, "w")
  f.write(script)
  f.close();

  chPerm = "chmod a+x %s/newVolumeScript.sh" % prefixT;
  execCommand(chPerm)
  
  print "to untar:"
  print "cd %s" % prefixT
  print "tar -x -F ./newVolumeScript.sh -f xxx.xxx/archive.tar"

  for snap in snapshots:
    srcFolder = prefix  + "/"  + snap
    dstFolder = prefixT + "/"  + snap
    print srcFolder
    print dstFolder
    mkFolder = "mkdir %s" % dstFolder
    execCommand(mkFolder)
    dstFile  = dstFolder + "/archive.tar"
    tarFolder = "cd %s && tar -cL%d -F %s/%s/newVolumeScript.sh -f %s/%s %s " %  (prefix, splitSize*1024*1024, pwd, prefixT, pwd, dstFile, snap)
    print tarFolder
    execCommand(tarFolder)

  print "to untar:"
  print "cd %s" % prefixT
  print "tar -x -F ./newVolumeScript.sh -f xxx.xxx/archive.tar"

