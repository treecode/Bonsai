#pragma once

class IDType
{
  private:
    uint64_t _IDTypePacked;
  public:
    IDType() : _IDTypePacked(0) {}
#if 0  /* eg: unsafe methods. Marked for removal in future commits */
    IDType(const uint64_t ID) : _IDTypePacked(ID) {}
    uint64_t getPacked() const { return _IDTypePacked; }
#endif
    void operator=(const IDType &id)  volatile
    {
      _IDTypePacked = id._IDTypePacked;
    }
    void operator=(const volatile IDType &id)  volatile
    {
      _IDTypePacked = id._IDTypePacked;
    }
    uint64_t get() const  volatile
    {
      return _IDTypePacked;
    }
    uint64_t getID() const volatile
    {
      return _IDTypePacked & ~0xFFFF000000000000ULL;
    }
    uint32_t getType() const volatile
    {
      return static_cast<uint32_t>(_IDTypePacked >> 48);
    }
    void setID(const int64_t ID) volatile
    {
      const uint32_t type = getType();
      _IDTypePacked = (ID & ~0xFFFF000000000000ULL) | (static_cast<uint64_t>(type) << 48);
    }
    void setType(const int type) volatile
    {
      const uint64_t ID = getID();
      _IDTypePacked  = ID | (static_cast<uint64_t>(type) << 48);
    }
};
