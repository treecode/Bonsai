#include <cuda_runtime_api.h>
#include "SharedMemory.h"
#include "BonsaiSharedData.h"

int main(int argc, char * argv[])
{
  using ShmQHeader = SharedMemoryServer<BonsaiSharedQuickHeader>;
  using ShmQData   = SharedMemoryServer<BonsaiSharedQuickData>;
  using ShmSHeader = SharedMemoryServer<BonsaiSharedSnapHeader>;
  using ShmSData   = SharedMemoryServer<BonsaiSharedSnapData>;

  ShmQHeader shmQHeader(ShmQHeader::type::sharedFile(), 1);
  ShmQData   shmQData  (ShmQData  ::type::sharedFile(), 1);

  ShmSHeader shmSHeader(ShmSHeader::type::sharedFile(), 1);
  ShmSData   shmSData  (ShmSData  ::type::sharedFile(), 1);

  return 0;
}

