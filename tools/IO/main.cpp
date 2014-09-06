#include "BonsaiIO.h"

struct real2 { float x,y;};
struct real4 { float x,y,z,w;};
typedef unsigned long long uulong;

void writeSnapshot(
    real4 *bodyPositions,
    real4 *bodyVelocities,
    uulong* bodyIds,
    const int n,
    const int nDomains,
    const std::string &fileName,
    const float time,
    const MPI_Comm &comm,
    const int nRank, const int myRank);

void readSnapshot(
    std::vector<real4>  &bodyPositions,
    std::vector<real4>  &bodyVelocities,
    std::vector<uulong> &bodyID,
    std::vector<real2>  &rhohList,
    const std::string   &fileName,
    const MPI_Comm       &comm,
    const int nRank, 
    const int myRank,
    int &NTotal2,
    int &NFirst, int &NSecond, int &NThird,
    std::vector<real4> &dustPositions, std::vector<real4> &dustVelocities,
    std::vector<uulong> &dustIDs, 
    const int reduce_bodies_factor,
    const int reduce_dust_factor,
    const bool restart);

int main(int argc, char * argv[])
{
  return 0;
};
