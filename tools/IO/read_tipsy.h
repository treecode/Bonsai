#ifndef __READ_TIPSY_H__
#define __READ_TIPSY_H__

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include "tipsydefs.h"

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000
  

class ReadTipsy
{
  private:
    const std::string baseFileName;
    const int rank, nranks;
    const int domains2read;
    const int reduceFactorFirst;
    const int reduceFactorSecond;
    const int reduceFactorThird;

  public:
    typedef unsigned long long long_t;
    typedef float real_t;
    struct real4 {real_t x, y, z, w;};
    
    /***********************/

    double time;
    std::vector<long_t> firstID;
    std::vector<real4 > firstPos;
    std::vector<real4 > firstVel;
    
    std::vector<long_t> secondID;
    std::vector<real4 > secondPos;
    std::vector<real4 > secondVel;
    
    std::vector<long_t> thirdID;
    std::vector<real4 > thirdPos;
    std::vector<real4 > thirdVel;

    /***********************/

    ReadTipsy(
        const std::string &_baseFileName, 
        const int _rank, 
        const int _nranks,
        const int _domains2read,
        const int _reduceFactorFirst  = 1,
        const int _reduceFactorSecond = 1,
        const int _reduceFactorThird  = 1) :
      baseFileName(_baseFileName), 
      rank(_rank), nranks(_nranks),
      domains2read(_domains2read), 
      reduceFactorFirst(_reduceFactorFirst),
      reduceFactorSecond(_reduceFactorSecond),
      reduceFactorThird(_reduceFactorThird)
  {
    if (rank == 0)
      fprintf(stderr, " Reading %d domains using %d ranks \n",
          domains2read, nranks);

    int ranks2read = nranks;
    if (domains2read < nranks)
    {
      if (rank == 0)
        fprintf(stderr, " Using only first %d ranks \n", domains2read);
      ranks2read = domains2read;
    }

    const bool readerRank = rank < ranks2read;

    if (readerRank)
    {
      std::vector<int> domainsPerRank(ranks2read, 0);
      for (int i = 0; i < domains2read; i++)
        domainsPerRank[i%ranks2read]++;
      std::vector< std::pair<int,int> > domainsBegEnd(ranks2read);
      domainsBegEnd[0].first  = 0;
      domainsBegEnd[0].second = domainsPerRank[0];
      for (int i = 1; i < ranks2read; i++)
      {
        domainsBegEnd[i].first  = domainsBegEnd[i-1].first + domainsPerRank[i-1];
        domainsBegEnd[i].second = domainsBegEnd[i  ].first + domainsPerRank[i  ];
        assert(domainsBegEnd[i].first == domainsBegEnd[i-1].second);
      }
      assert(domainsBegEnd[ranks2read-1].second == domains2read);

      for (int domain = domainsBegEnd[rank].first ; domain < domainsBegEnd[rank].second; domain++)
      {
        if (rank == 0)
          fprintf(stderr, " reading domain %d [ %d %d ] \n", domain, domainsBegEnd[rank].first, domainsBegEnd[rank].second);

	std::string fullFileName = baseFileName;
	if(_domains2read > 1)
		fullFileName.assign( baseFileName + (domains2read>1 ? std::to_string(domain) : std::string()));

        std::ifstream inputFile(fullFileName, std::ios::in | std::ios::binary);
        if(!inputFile.is_open())
        {
            fprintf(stderr, " rank= %d  domain= %d :: %s input file is missing\n" ,
          	    rank, domain, fullFileName.c_str());
	    //File renaming does not work when using a single rank. If that is the case in this situation
	    //than adding the 0 will work. If there is another reason the first open attempt fail then 
	    //the program will fail after all.
	    if(_domains2read == 1) 
	    {
		fullFileName.append(std::string("0"));
		inputFile.open(fullFileName,  std::ios::in | std::ios::binary);
        	if(!inputFile.is_open())
		{
          		fprintf(stderr, " rank= %d  domain= %d :: %s input file is missing\n" ,
            			  rank, domain, fullFileName.c_str());
          		::exit(-1);
		}
	    }
	    else
	    {
		::exit(-1);
	    }
        }

        readInputFile(inputFile, domain);

        inputFile.close();
      }
    }

  }

  protected:

    void readInputFile(std::istream &inputFile, const int domain)
    {
      dumpV2  h;
      inputFile.read((char*)&h, sizeof(h));  
  
      long_t idummy;
      real4 positions;
      real4 velocity;
  
      //Read Tipsy header
      const int NTotal        = h.nbodies;
      const int NFirst        = h.ndark;
      const int NSecond       = h.nstar;
      const int NThird        = h.nsph;
      //fprintf(stderr, "Info about stars: %d %d %d \n", NTotal, NFirst, NSecond);

      assert(NThird == 0);
      assert(NTotal == NFirst + NSecond + NThird);
	

      if (domain == 0)
      {
        fprintf(stderr, "Tipsy file version: %d \n", (int)h.version);
        fprintf(stderr, "Simulation time: %g \n", (float)h.time);
      }
      time = h.time;

      int fileFormatVersion = 0;
      if(h.version == 2) fileFormatVersion = 2;

      //Start reading
      int globalParticleCount = 0;
      int firstCount  = 0;
      int secondCount = 0;

      dark_particleV2 d;
      star_particleV2 s;

      for (int i=0; i < NTotal; i++)
      {
        int whichOne = -1;
        if(i < NFirst)
        {
          inputFile.read((char*)&d, sizeof(d));
          velocity.w        = 0;
          positions.w       = d.mass;
          positions.x       = d.pos[0];
          positions.y       = d.pos[1];
          positions.z       = d.pos[2];
          velocity.x        = d.vel[0];
          velocity.y        = d.vel[1];
          velocity.z        = d.vel[2];
          idummy            = d.getID();
          if(fileFormatVersion == 0)
            idummy = d.getID_V1() + DARKMATTERID;
          
          firstCount++;
          whichOne = 1;
        }
        else
        {
          inputFile.read((char*)&s, sizeof(s));
          velocity.w        = 0;
          positions.w       = s.mass;
          positions.x       = s.pos[0];
          positions.y       = s.pos[1];
          positions.z       = s.pos[2];
          velocity.x        = s.vel[0];
          velocity.y        = s.vel[1];
          velocity.z        = s.vel[2];
          idummy            = s.getID();
          if(fileFormatVersion == 0)
          {
            if(s.getID_V1() >= 100000000)
              idummy    = s.getID_V1() + BULGEID; //Bulge particles
            else
              idummy    = s.getID_V1();
          }

          secondCount++;
          whichOne = 2;
        }

        if(positions.z < -10e10)
        {
          fprintf(stderr," domain= %d: Removing particle %d because of Z is: %f \n", 
              domain, globalParticleCount, positions.z);
          continue;
        }

        switch (whichOne)
        {
          case 1:
	    if (reduceFactorFirst > 0){
            if (firstCount % reduceFactorFirst == 0)
            {
              positions.w *= reduceFactorFirst;
              firstID .push_back(idummy); 
              firstPos.push_back(positions);
              firstVel.push_back(velocity);
            }
	    }
            break;
          case 2:
	    if(reduceFactorSecond > 0){
            if (secondCount % reduceFactorSecond == 0)
            {
              positions.w *= reduceFactorSecond;
              secondID .push_back(idummy); 
              secondPos.push_back(positions);
              secondVel.push_back(velocity);
            }
	    }
            break;
          default: 
            assert(whichOne == 1 || whichOne == 2);
        }

      }

    }

};

#endif /* __READ_TIPSY_H__ */
