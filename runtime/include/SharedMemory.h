#pragma once

#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <exception>
#include <sys/shm.h>
#include <unistd.h>
    

template<typename T>
class SharedMemoryBase
{
  private:
    enum {FTOKID = 1234};
  protected:
    const key_t shmKey;
    int shmid;
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
    SharedMemoryBase(const std::string &descriptor) : shmKey(ftok(descriptor.c_str(),FTOKID)), shmid(0), shm(NULL) {}
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
      shm = NULL;
    }
    void attach()
    {
      assert(!shm);
      shm = shmat(shmid, NULL, 0);
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
  
    bool aquireLock()
    {
      return __sync_bool_compare_and_swap(&header->locked, 0, 1);
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
      p::shmid = shmget(p::shmKey, sizeof(Header) + capacity*sizeof(T), IPC_CREAT | 0666);
      if (p::shmid < 0)
        throw Exception("SharedMemoryServer::Ctor - shmget failed with error " + std::to_string(p::shmid) + "\n");
      p::attach();
      p::header->capacity    = capacity;
      p::header->size        = 0;
      p::header->locked      = 0;
    }
    ~SharedMemoryServer() 
    {
      p::detach();
      shmctl(p::shmid, IPC_RMID,0);
      p::shmid = 0;
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
      int shmid = 0;
      while (shmid <= 0 && dt < timeOut)
      {
        shmid = shmget(p::shmKey, sizeof(Header), 0666);
        usleep(100000);
        dt += 0.1;
      }
      if (!(dt < timeOut))
        throw "SharedMemoryClient::Ctor - timed out while waiting for server";

      sleep(1);

      void *shm = shmat(shmid, NULL, 0);
      if (shm == NULL)
        throw Exception("SharedMemoryClient::Ctor - shmat failed");

      const auto header = *(Header*)shm;
      shmdt(shm);

      p::shmid = shmget(p::shmKey, sizeof(Header) + header.capacity*sizeof(T), 0666);
      if (p::shmid < 0)
        throw Exception("SharedMemoryClient::Ctor - shmget failed with error " + std::to_string(p::shmid) + "\n");
      p::attach();
    }

    ~SharedMemoryClient()
    {
      p::detach();
    }
};
