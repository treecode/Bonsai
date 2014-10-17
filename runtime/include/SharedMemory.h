#pragma once

#include <cmath>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <string>
#include <exception>
#include <sys/shm.h>
#include <unistd.h>
    
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>

template<typename T>
class SharedMemoryBase
{
  protected:
    const std::string descriptor;
    int shmfd;
    volatile void *shm;

    struct Header
    {
      size_t size;
      size_t capacity;
      int    locked;
    };

    volatile Header *header;
    volatile T      *data; 

  public:
    using type = T;
    SharedMemoryBase(const std::string &_descriptor) : descriptor(_descriptor), shmfd(-1), shm(NULL) {}
    virtual ~SharedMemoryBase() {}

    struct Exception : public std::exception 
    {
      std::string s;
      Exception(std::string ss) : s(ss) {}
      ~Exception() throw () {} // Updated
      const char* what() const throw() { return s.c_str(); }
    };

  protected:
  
    void detach()
    {
      assert(shm);
      munmap((void*)shm, header->capacity);
      shm = NULL;
    }
    void attach(size_t capacity = 0)
    {
      assert(shmfd >= 0);
      if (!capacity)
      {
        void *map = mmap(NULL,sizeof(Header),PROT_READ|PROT_WRITE, MAP_SHARED,shmfd,0);
        if (map == NULL)
          throw Exception("SharedMemory::attach - mmap failed to map Header.");
        capacity = reinterpret_cast<Header*>(map)->capacity;
        munmap(map,sizeof(Header));
      }
      shm = mmap(NULL,sizeof(Header) + capacity*sizeof(T),PROT_READ|PROT_WRITE, MAP_SHARED,shmfd,0);
      if (shm == NULL)
        throw Exception("SharedMemory::attach - shmat failed.");
      header = (Header*)shm;
      data   = (T*     )((char*)shm + sizeof(Header));
    }
    
  public:

    const volatile T& operator[](const size_t i) const { return data[i]; }
    volatile T& operator[](const size_t i) { return data[i]; }
    
    ////////////////
    
    size_t size()     const { return header->size;     }
    size_t capacity() const { return header->capacity; }
    
    ////////////////
    
    bool resize(const size_t size)
    {
      assert(shm);
      if (size > header->capacity)
        return false;
      header->size = size;
      return true;
    }

    ////////////////
  
    void acquireLock()
    {
      while (!__sync_bool_compare_and_swap(&header->locked, 0, 1));
    } 
    void acquireLock(const float sleep)
    {
      while (!__sync_bool_compare_and_swap(&header->locked, 0, 1))
        usleep(static_cast<int>(sleep*1000));
    } 
    void releaseLock()
    {
      header->locked = 0;
    }
};

template<typename T>
class SharedMemoryServer : public SharedMemoryBase<T>
{
  private:
    using p = SharedMemoryBase<T>;
    using Exception = typename p::Exception;
    using Header    = typename p::Header;

  public:
    SharedMemoryServer(const std::string &descriptor, const size_t capacity) : p(descriptor)
    {
      p::shmfd = shm_open(p::descriptor.c_str(), O_CREAT|O_RDWR, S_IRUSR|S_IWUSR);
      if (p::shmfd < 0)
        throw Exception("SharedMemoryServer::Ctor - shm_open failed with error " + std::to_string(p::shmfd) + "\n");
    
      if (ftruncate(p::shmfd,sizeof(Header)+capacity*sizeof(T)))
        throw Exception("SharedMemoryServer::Ctor = ftruncate failed.");

      p::attach(capacity);
      p::header->capacity    = capacity;
      p::header->size        = 0;
      p::header->locked      = 0;
    }
    ~SharedMemoryServer() 
    {
      p::detach();
      close(p::shmfd);
      shm_unlink(p::descriptor.c_str());
      p::shmfd = 0;
    }
};

template<typename T>
class SharedMemoryClient: public SharedMemoryBase<T>
{
  private:
    using p = SharedMemoryBase<T>;
    using Exception = typename p::Exception;
    using Header    = typename p::Header;

  public:
    SharedMemoryClient(const std::string &descriptor, const double timeOut = HUGE) : p(descriptor) 
    {
      double dt = 0.0;
      while (p::shmfd < 0 && dt < timeOut)
      {
        p::shmfd = shm_open(p::descriptor.c_str(), O_RDWR,0);
        usleep(100000);
        dt += 0.1;
      }
      if (!(dt < timeOut))
        throw "SharedMemoryClient::Ctor - timed out while waiting for server";

      sleep(1);

      p::attach();
    }

    ~SharedMemoryClient()
    {
      p::detach();
      close(p::shmfd);
      p::shmfd = 0;
    }
};
