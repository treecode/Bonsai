#pragma once

class IDType
{
  private:
    uint64_t _IDTypePacked;
  public:
    IDType() : _IDTypePacked(0) {}
    uint64_t getID() const
    {
      return _IDTypePacked & ~0xFFFF000000000000ULL;
    }
    uint32_t getType() const
    {
      return static_cast<uint32_t>(_IDTypePacked >> 48);
    }
    void setID(const int64_t ID)
    {
      const uint32_t type = getType();
      _IDTypePacked = (ID & ~0xFFFF000000000000ULL) | (static_cast<uint64_t>(type) << 48);
    }
    void setType(const int type)
    {
      const uint64_t ID = getID();
      _IDTypePacked  = ID | (static_cast<uint64_t>(type) << 48);
    }
};
