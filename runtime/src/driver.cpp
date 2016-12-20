#undef NDEBUG
#include <iostream>
#include <mpi.h>
#include <dlfcn.h>
#include <vector>

#include <string>
#include <sstream>
#include <cassert>
#include <functional>
#include <unistd.h>



class DynamicLoader
{
  private:
    void* const m_handle;

  public:
    DynamicLoader(const std::string &filename) :
      m_handle(dlopen(filename.c_str(), RTLD_LAZY)) 
  {
    if (!m_handle)
      throw std::logic_error("can't load library named \"" + filename + "\"");
  }

    template<class T> std::function<T> load(const std::string &functionName) const
    {
      dlerror();
      const void* result = dlsym(m_handle, functionName.c_str());
      if (!result)
      {
        const char* error = dlerror();
        if (error)
          throw std::logic_error("can't find symbol named \"" + functionName + "\": " + error);
      }

      return reinterpret_cast<T*>(result);
    }
};


static std::vector<std::string> lSplitString(const std::string &s, const char delim)
{
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, delim)) 
    if (!item.empty())
      elems.push_back(item);
  return elems;
}

static std::vector<std::vector<std::string>> lParseInput(const int rank)
{
  std::string input;
  std::string tmp;
  if(rank == 0)
     while(std::getline(std::cin, tmp))
         input += tmp + "\n";

  int size[2] = {static_cast<int>(input.size()), static_cast<int>(input.capacity())};
  MPI_Bcast(&size, 2, MPI_INT, 0, MPI_COMM_WORLD);
  input.resize (size[0]);
  input.reserve(size[1]);
  MPI_Bcast(&input[0], size[0], MPI_BYTE, 0, MPI_COMM_WORLD);

  const std::vector<std::string> lines = lSplitString(input,'\n');
  std::vector<std::vector<std::string>> programs;
  for (const auto &line : lines)
    programs.push_back(lSplitString(line,' '));
  return programs;
}

int main(int argc, char *argv[]) 
{
  std::cerr << "Loaded the driver program "<< std::endl;
#ifdef _MPIMT
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
  assert(MPI_THREAD_MULTIPLE == provided);
#else
  MPI_Init(&argc, &argv);
#endif

  int rank, nrank, pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nrank);
  

  const auto &programs = lParseInput(rank);

  if (rank == 0)
  {
     std::cerr << "Executing the following binaries: \n";
     for (const auto &p  : programs)
     {
       for (const auto &arg : p)
         std::cerr << arg << " ";
       std::cerr << std::endl;
     }
     //Retrieve the process ID of the current process
     pid =  getpid();
  }

  const int nprograms = programs.size();
  if (nrank%nprograms != 0)
  {
    if (rank == 0)
      std::cerr << "Fatal: nranks= " << nrank << " should be divisible by the nprogram= " << nprograms << ".\n";
    MPI_Finalize();
    ::exit(0);
  }
  
  //For the SharedMemory it is important that a unique name is used.
  //This works within a single launch instance by using the ranks. However, when we use
  //multiple instances on a single node then this will result in naming clashes.
  //To prevent that use the PID of process 0 as an extra unique identifier and pass this to
  //all the processes within this launch instance
  MPI_Bcast(&pid, 1, MPI_INT, 0, MPI_COMM_WORLD);

//   fprintf(stderr,"received pid: %d\t%d \n", rank,pid);



  const int color = rank%nprograms;
  const int key   = rank/nprograms;

  MPI_Comm comm;
  MPI_Comm_split(MPI_COMM_WORLD,color,key,&comm);

  const auto &arguments = programs[color];
  const std::string &libName = arguments[0];

  const DynamicLoader dll(libName);
  const auto program = dll.load<void(int,char**,MPI_Comm,int)>("main");

  std::vector<char*> argVec;
  for (const auto &arg : arguments)
    argVec.push_back((char*)arg.c_str());
  argVec.push_back(NULL);
  std::cerr << "Ready 2 launch \n" << std::endl; 
  program(static_cast<int>(argVec.size()-1), &argVec[0], comm, pid);

  if (rank == 0)
    fprintf(stderr, " %s finalizing .. \n", argv[0]);
  MPI_Finalize();
  exit(0);
}
