#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <vector>
#include <string>

#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */

int getFolderContent(std::string folder, std::vector<std::string> &files)
{
  DIR *dp;
  struct dirent *dirp;
  if((dp  = opendir(folder.c_str())) == NULL)
  {
    fprintf(stderr,"Error( %d ) opening %s \n", errno, folder.c_str());
    return errno;
  }

  while ((dirp = readdir(dp)) != NULL)
  {
    files.push_back(std::string(dirp->d_name));
  }
  closedir(dp);
  return 0;
}


int main(int argc, char * argv[])
{
//  using ShmQHeader = SharedMemoryServer<BonsaiSharedQuickHeader>;
//  using ShmQData   = SharedMemoryServer<BonsaiSharedQuickData>;
//  using ShmSHeader = SharedMemoryServer<BonsaiSharedSnapHeader>;
//  using ShmSData   = SharedMemoryServer<BonsaiSharedSnapData>;
//  const int n = argc > 1 ? atoi(argv[1]) : 1;
//  fprintf(stderr, "cleaning for n= %d\n", n);
//  for (int i = 0; i < n; i++)
//  {
//#if 0
//    fprintf(stderr, "clear: %s \n", ShmQHeader::type::sharedFile(i));
//    fprintf(stderr, "clear: %s \n", ShmQData  ::type::sharedFile(i));
//    fprintf(stderr, "clear: %s \n", ShmSHeader::type::sharedFile(i));
//    fprintf(stderr, "clear: %s \n", ShmSData  ::type::sharedFile(i));
//#endif
//    //ShmQHeader shmQHeader(ShmQHeader::type::sharedFile(i), 1);
//    //ShmQData   shmQData  (ShmQData  ::type::sharedFile(i), 1);
//  //  ShmSHeader shmSHeader(ShmSHeader::type::sharedFile(i), 1);
//   // ShmSData   shmSData  (ShmSData  ::type::sharedFile(i), 1);
//  }


  fprintf(stderr,"Cleaning up Bonsai's shared memory buffers\n");
  //Get all the shared memory files
  std::string folder("/dev/shm"); //Or on Debian /run/shm also works
  std::vector<std::string> files;
  getFolderContent(folder,files);

  //Check if there are any files created by Bonsai
  for(auto &p : files)
  {
    if(p.find("Bonsai") !=std::string::npos)
    {
      fprintf(stderr,"Removing %s \n", p.c_str());
      shm_unlink(p.c_str());
    }
  }

  fprintf(stderr,"Finished cleaning\n");

  return 0;
}

