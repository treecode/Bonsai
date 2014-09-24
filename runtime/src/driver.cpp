#include <mpi.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cassert>
#include <functional>
#include <unistd.h>


class DynamicLoader
{
  private:
    void* const m_handle;

  public:
    DynamicLoader(std::string const& filename) :
      m_handle(dlopen(filename.c_str(), RTLD_LAZY)) 
    {
      if (!m_handle)
      {
        throw std::logic_error("can't load library named \"" + filename + "\"");
      }
    }

    template<class T> std::function<T> load(std::string const& functionName) const
    {
      dlerror();
      void* const result = dlsym(m_handle, functionName.c_str());
      if (!result)
      {
        char* const error = dlerror();
        if (error)
        {
          throw std::logic_error("can't find symbol named \"" + functionName + "\": " + error);
        }
      }

      return reinterpret_cast<T*>(result);
    }

};

int main(int argc, char *argv[]) 
{
  MPI_Init(&argc, &argv);

  auto splitComm = [&](const int nway)
  {
    MPI_Comm comm;
    int rank, nrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);
    const int color = rank%nway;
    const int key   = rank/nway;
    MPI_Comm_split(MPI_COMM_WORLD,color,key,&comm);
    return std::make_tuple(comm,color,key);
  };

  const auto commData = splitComm(2);  
  const auto comm  = std::get<0>(commData);
  const auto color = std::get<1>(commData);

  auto getCwd = []()  
  {
    char cwd[FILENAME_MAX];
    assert(getcwd(cwd, sizeof(cwd)) > 0);
    return std::string(cwd);
  };
  const auto cwd = getCwd();

  std::vector<std::string> arguments;
  std::string libName(cwd);
  libName += "/";
  if (color == 0)
  {
    arguments.push_back("app1");
    arguments.push_back("arg11");
    libName += "libapp1.so";
  }
  else
  {
    arguments.push_back("app2");
    arguments.push_back("arg21");
    arguments.push_back("arg22");
    libName += "libapp2.so";
  }

  const DynamicLoader dll(libName);
  const auto program = dll.load<void(int,char**,MPI_Comm)>("run");

  std::vector<char*> argVec;
  for (const auto &arg : arguments)
    argVec.push_back((char*)arg.c_str());
  argVec.push_back(NULL);
  program(static_cast<int>(argVec.size()-1), &argVec[0], comm);

  MPI_Finalize();
  exit(0);
}
