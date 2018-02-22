/*
 *
 *  Process the input of the following SPLASH command:
 *  ./ssplash to ascii <binary-phantom-file>
 *
 *  Boundaries can be read from the ASCII file if a modified SPLASH version is used
 *  alternatively an optional text file can be given from which the boundaries are 
 *
 */


#include "BonsaiIO.h"
#include "IDType.h"
#include <array>
#include <fstream>
#include <sstream>
#include <iostream>
#include <map>
#include <algorithm> 
#include <cctype>
#include <locale>
#include <vector>
#include <regex>
#include "anyoption.h"

std::vector<std::string> Split(const std::string& subject) {
    static const std::regex re{"\\s+"};
    std::vector<std::string> container{
        std::sregex_token_iterator(subject.begin(), subject.end(), re, -1), 
        std::sregex_token_iterator()
    };
    return container;
}


// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
        return !std::isspace(ch);
    }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
        return !std::isspace(ch);
    }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}


std::string trim2(std::string& str)
{
  size_t first = str.find_first_not_of(' ');
  size_t last = str.find_last_not_of(' ');
  std::cerr << "bounds: " << first << "\t" << last << std::endl;
  return str.substr(first, (last-first+1));
}

typedef struct real4
{
  float x,y,z,w;
} real4;

typedef struct real2
{
  float x,y,z,w;
} real2;

typedef float float4[4];
typedef float float3[3];
typedef float float2[2];  

//The hydro properties: x = pressure, y = soundspeed, z = Energy , w = Balsala Switch

#define DARKMATTERID  3000000000000000000
#define DISKID        0
#define BULGEID       2000000000000000000
#define SPHBOUND          100000000000000 

static IDType lGetIDType (const long long id)
{
  IDType ID;
  ID.setID(id);
  ID.setType(3);     /* Everything is Dust until told otherwise */
  if(id >= DISKID  && id < BULGEID)       
  {
    ID.setType(2);  /* Disk */
    ID.setID(id - DISKID);
  }
  else if(id >= BULGEID && id < DARKMATTERID)  
  {
    ID.setType(1);  /* Bulge */
    ID.setID(id - BULGEID);
  }
  else if (id >= DARKMATTERID)
  {
    ID.setType(0);  /* DM */
    ID.setID(id - DARKMATTERID);
  }
  return ID;
};


void readPhantomSPHFile(std::vector<real4>    &bodyPositions, 
                        std::vector<real4>    &bodyVelocities, 
                        std::vector<IDType>   &bodiesIDs, 
                        std::vector<real2>    &bodyDensRho, 
                        std::vector<real4>    &bodyDrvt, 
                        std::vector<real4>    &bodyHydro, 
                        double                &time,
						float3				  &min_range,
						float3				  &max_range,
                        std::string fileName,
                        const int reduceFactor) {
  
    bodyPositions.clear();
  

    std::cerr << "Trying to read file: " << fileName << std::endl;

    std::ifstream inputFile(fileName, std::ios::in);
    
    if(!inputFile.is_open())
    {
      std::cerr << "Can't open input file \n";
      exit(0);
    }
    
    // Read  through the header until we find (optional) boundaries,
    // the column descriptions

    std::string textLine;

    std::string sdummy;

    std::map<std::string, int> colmap;
    colmap["x"] = -1;
    colmap["y"] = -1;
    colmap["z"] = -1;
    colmap["particle mass"] = -1;
    colmap["h"] = -1;
    colmap["density"] = -1;
    colmap["v_x"] = -1;
    colmap["v_y"] = -1;
    colmap["v_z"] = -1;
    colmap["u"] = -1;
    colmap["itype"] = -1;
    //We need: x,y,z,mass,h,density,vx,vy,vz,
    //optionally: y,itype
    //
    std::map<int,int> ptype_count; 
    for(int i=0; i < 7; i++) ptype_count[i] = 0;

    do
    {
        std::getline(inputFile, textLine);

		if(textLine.find("time") != std::string::npos)
		{
			std::getline(inputFile, textLine);  //Jump to the line that encodes the time
			std::stringstream ss(textLine);
			ss >> sdummy >> time; 
		}
		if(textLine.find("boundaries") != std::string::npos)
		{
			std::cerr << "This is a boundaries line: " << textLine << std::endl;
			std::stringstream ss(textLine);
			ss >> sdummy >> sdummy >> min_range[0] >> max_range[0] >> min_range[1] >> 
				max_range[1] >> min_range[2] >> max_range[2];
		}
		if(textLine.find("units") != std::string::npos)
		{
			//Find the columns
			std::getline(inputFile, textLine); //Unit values
			std::getline(inputFile, textLine); //Unit names
			std::getline(inputFile, textLine); //Empty
			std::getline(inputFile, textLine); //Column names
			assert(textLine[2] == 'x'); // Assure that we are in the right line
			std::stringstream ss(textLine);
			//
			std::cerr << "This is header line: " << textLine << std::endl;
			int idx = 0;
			char buff[32];
			ss.read(buff,2); //Remove the # symbol
			while(ss)
			{
				ss.read(buff, 16);
				std::string temp(buff, 15); //1 less to remove end of text char
				trim(temp); //Remove spaces
				//std::cerr << "TEST: " << idx << "\t" << buff[0] << std::endl;
				if(colmap.find(temp) != colmap.end())
				{
					colmap[temp] = idx;
					//std::cerr << "TEST2: " << idx << "\t" << colmap[temp] << "\t" << temp << std::endl;
				}
				idx++;
			}
			break;
		} //if units, column indexing
    } while(true);
            
	fprintf(stderr,"Time: %f \nx-bounds: %f\t%f\ny-bounds: %f\t%f\nz-bounds: %f\t%f\n",
                    time, min_range[0], max_range[0], min_range[1], max_range[1], min_range[2], max_range[2]);

//	unsigned int prev = 0;
	int counter = 0;
	int			pid = 0;
    real4       pos;
    real4       vel;
    real4       drvt = {0,0,0,0};
    real4       hydro = {0,0,0,0};
    real2       rhoh;
    int         ptype;
    while(std::getline(inputFile, textLine))
	{
		//Split the line in the columns
		trim(textLine);
		std::vector<std::string> columns = Split(textLine.c_str());

		//Fill our data
		pos.x = atof(columns[colmap["x"]].c_str());
		pos.y = atof(columns[colmap["y"]].c_str());
		pos.z = atof(columns[colmap["z"]].c_str());
		pos.w = atof(columns[colmap["particle mass"]].c_str());
		
		vel.x = atof(columns[colmap["v_x"]].c_str());
		vel.y = atof(columns[colmap["v_y"]].c_str());
		vel.z = atof(columns[colmap["v_z"]].c_str());
	
		rhoh.x = atof(columns[colmap["density"]].c_str());
		rhoh.y = atof(columns[colmap["h"]].c_str());
    
		hydro.z = atof(columns[colmap["u"]].c_str());
		
		ptype = atof(columns[colmap["itype"]].c_str());

		if(counter % 50000 == 0) fprintf(stderr,"Reading line: %d \n", counter);
      
		if(reduceFactor > 0)
      	{
        	if(counter % reduceFactor == 0)
	        {
			  pos.w *= reduceFactor;
			  bodyPositions.push_back(pos);
			  bodyVelocities.push_back(vel);
			  bodyDensRho.push_back(rhoh);
			  bodyDrvt.push_back(drvt);
			  bodyHydro.push_back(hydro);

              ptype_count[ptype]++;
			  
			  //ptype == 1 is default gas particle 
			  //ptype == 4 is boundary particle
			  size_t        offset=0;
			  if(ptype == 4)
			  {
				  offset = SPHBOUND; 
			  }
		  
			  //Convert the ID to a star (disk for now) particle
			  bodiesIDs.push_back(lGetIDType(pid++ + offset));
			}
      }//reduceFactor 
	  counter++;

#if 0	
		if(columns.size() != prev){
			std::cerr<< "Columns: " << columns.size() << std::endl; 
			std::cerr<< "Columns: " << textLine << std::endl; 
			prev = columns.size();
			for(size_t i=0; i < columns.size(); i++)
				std::cerr << i << "\t" << columns[i] << std::endl;
			
		}
#endif

    } //while


  inputFile.close();
  
  fprintf(stderr, "read %d bodies from dump file \n", counter);
  fprintf(stderr, "Particle count per type: \n");
  for(int i=0; i < 7; i++)
  {
      fprintf(stderr,"%d\t\t%d\n", i, ptype_count[i]);
  }
};


void readDomainBoundaries(std::string &fileName, float3 &min_range, float3 &max_range)
{
    std::ifstream inputFile(fileName, std::ios::in);

    if(!inputFile.is_open())
    {
      std::cerr << "Can't open input file: " << fileName << " \n";
      exit(0);
    }

    std::string textLine;
    std::getline(inputFile, textLine);
    std::stringstream ss(textLine);
    ss >> min_range[0] >> min_range[1] >> min_range[2] >> 
          max_range[0] >> max_range[1] >> max_range[2];
}



#if 1
template<typename IO, size_t N>
static double writeStars(std::vector<real4>     &bodyPositions, 
                         std::vector<real4>     &bodyVelocities, 
                         std::vector<IDType>    &bodiesIDs, 
                         std::vector<real2>    &bodyDensRho, 
                         std::vector<real4>    &bodyDrvt, 
                         std::vector<real4>    &bodyHydro, 
                         IO &out,
                         std::array<size_t,N> &count)
{
    double dtWrite = 0;
    
    const int nS  = bodyPositions.size();
  
  
    BonsaiIO::DataType<IDType> S_id ("Stars:IDType",           nS);
    BonsaiIO::DataType<real4>  S_pos("Stars:POS:real4",        nS);
    BonsaiIO::DataType<float3> S_vel("Stars:VEL:float[3]",     nS);
    BonsaiIO::DataType<float2> S_rhoh("Stars:RHOH:float[2]",   nS);
    BonsaiIO::DataType<float4> S_hydro("Stars:HYDRO:float[4]", nS);
    BonsaiIO::DataType<float4> S_drv  ("Stars:DRVT:float[4]" , nS);  
	
    

    for (int i = 0; i< nS; i++)
    {
      S_id[i] = bodiesIDs[i];   
      S_pos[i] = bodyPositions[i];
      S_vel[i][0] = bodyVelocities[i].x;
      S_vel[i][1] = bodyVelocities[i].y;
      S_vel[i][2] = bodyVelocities[i].z; 
      
      S_rhoh[i][0] = bodyDensRho[i].x;
      S_rhoh[i][1] = bodyDensRho[i].y;      
      
      S_hydro[i][0] = bodyHydro[i].x;
      S_hydro[i][1] = bodyHydro[i].y;
      S_hydro[i][2] = bodyHydro[i].z; 
      S_hydro[i][3] = bodyHydro[i].w; 
      
      S_drv[i][0] = bodyDrvt[i].x;
      S_drv[i][1] = bodyDrvt[i].y;
      S_drv[i][2] = bodyDrvt[i].z; 
      S_drv[i][3] = bodyDrvt[i].w;       
    }    

  
    out.write(S_id);
    out.write(S_pos);
    out.write(S_vel);
    out.write(S_rhoh);
    out.write(S_hydro);
    out.write(S_drv);    
  
  

  return dtWrite;
}
#endif

int main(int argc, char * argv[])
{
  MPI_Comm comm = MPI_COMM_WORLD;

  MPI_Init(&argc, &argv);
    
  int nranks, rank;
  MPI_Comm_size(comm, &nranks);
  MPI_Comm_rank(comm, &rank);

  AnyOption opt;

  #define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}

  ADDUSAGE("Converter command line usage:");
  ADDUSAGE("                  ");
  ADDUSAGE(" -h  --help             Prints this help ");
  ADDUSAGE(" -i  --infile #         Input snapshot filename in ASCII format, output of 'ssplash to ascii'");
  ADDUSAGE(" -o  --outfile #        Output file, input file converted to Bonsai format");
  ADDUSAGE(" -r  --reduce #         Reduction factor of the input file [1]");
  ADDUSAGE(" -p  --periodicity #    The periodic axis: 1-x, 2-y, 4-z . Bitwise, sum to combine (xyz =7) [0]");
  ADDUSAGE(" -d  --dfile #          Domain information file with min/high domains, on single line: xmin ymin zmin xhigh yhigh zhigh [0's]");
  opt.setFlag("help", 'h');
  opt.setOption("infile", 'i');
  opt.setOption("outfile", 'o');
  opt.setOption("dfile", 'd');
  opt.setOption("reduce", 'r');
  opt.setOption("periodicity", 'p');

  opt.processCommandArgs( argc, argv );
  if( ! opt.hasOptions() ||  opt.getFlag( "help" ) || opt.getFlag( 'h' ) )
  {
  	/* print usage if no options or requested help */
    if (rank == 0)  opt.printUsage();
    ::exit(0);
  }
  
  std::string baseName;
  std::string outputName;
  std::string dName;
  int reduceFactor  = 1;
  int periodicity   = 0;

  char *optarg = NULL;
  if ((optarg = opt.getValue("infile")))      baseName   = std::string(optarg); 

  if ((optarg = opt.getValue("outfile")))     outputName = std::string(optarg);
  if ((optarg = opt.getValue("dfile")))       dName      = std::string(optarg);
  if ((optarg = opt.getValue("reduce")))      reduceFactor = atoi(optarg);
  if ((optarg = opt.getValue("periodicity"))) periodicity  = atoi(optarg);
  
  if(baseName.empty() || outputName.empty())
  {
    if (rank == 0)  opt.printUsage();
    ::exit(0);
  }
 

  if( rank == 0)
  {
      fprintf(stderr,"Basename: %s  outputname: %s \n", baseName.c_str(), outputName.c_str());
      fprintf(stderr,"Options: reduce: %d periodicity: %d boundary-name: %s \n",
              reduceFactor, periodicity, dName.c_str());
  }
 
  std::vector<real4>   bodyPositions;
  std::vector<real4>   bodyVelocities;
  std::vector<IDType>  bodiesIDs;
  std::vector<real2>   bodyDensRho;
  std::vector<real4>   bodyDrvt;
  std::vector<real4>   bodyHydro;
 
  double data_time = 0.0;
  float3 min_range = {NAN,NAN,NAN};
  float3 max_range = {NAN,NAN,NAN};

  readPhantomSPHFile(bodyPositions, bodyVelocities, bodiesIDs, bodyDensRho, bodyDrvt, bodyHydro,
                     data_time, min_range, max_range, baseName, reduceFactor);
  
  fprintf(stderr,"From IC file: Time: %f \nx-bounds: %f\t%f\ny-bounds: %f\t%f\nz-bounds: %f\t%f\n",
                    data_time, min_range[0], max_range[0], min_range[1], max_range[1], min_range[2], max_range[2]);

  if(!dName.empty())
  {
    readDomainBoundaries(dName, min_range, max_range);
    fprintf(stderr,"From boundaries file: Time: %f \nx-bounds: %f\t%f\ny-bounds: %f\t%f\nz-bounds: %f\t%f\n",
                    data_time, min_range[0], max_range[0], min_range[1], max_range[1], min_range[2], max_range[2]);
  }

  if(periodicity == 0)
  {
      if(min_range[0] != NAN)
      {
          fprintf(stderr,"*********************************************************************************\n");
          fprintf(stderr,"*                                                                               *\n");
          fprintf(stderr,"*  WARNING: You have set periodic boundaries but not specified the periodicity  *\n");
          fprintf(stderr,"*                                                                               *\n");
          fprintf(stderr,"*********************************************************************************\n");
      }
  }


  if(rank == 0)
  {
    fprintf(stderr, " nTotal= %ld \n", bodyPositions.size());
    fprintf(stderr, "Boundaries: \n");
    fprintf(stderr, "X: %f\tto\t%f\tactive: %d\n", min_range[0], max_range[0], (periodicity & 1) > 0);
    fprintf(stderr, "Y: %f\tto\t%f\tactive: %d\n", min_range[1], max_range[1], (periodicity & 2) > 0);
    fprintf(stderr, "Z: %f\tto\t%f\tactive: %d\n", min_range[2], max_range[2], (periodicity & 4) > 0);
  }

  const double tAll = MPI_Wtime();
  {
    const double tOpen = MPI_Wtime();
    BonsaiIO::Core out(rank, nranks, comm, BonsaiIO::WRITE, outputName);
    if (rank == 0)
      fprintf(stderr, "Time= %g\n", data_time);
    out.setTime(data_time); //start at t=0

	//Write the boundary information
	out.setDomainInfo(min_range[0], min_range[1], min_range[2],
					  max_range[0], max_range[1], max_range[2], periodicity);


    double dtOpenLoc = MPI_Wtime() - tOpen;
    double dtOpenGlb;
    MPI_Allreduce(&dtOpenLoc, &dtOpenGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "open file in %g sec \n", dtOpenGlb);

    double dtWrite = 0;
  
    std::array<size_t,10> ntypeloc, ntypeglb;
    std::fill(ntypeloc.begin(), ntypeloc.end(), 0);

    
    if (rank == 0) fprintf(stderr, " write Stars\n");
    MPI_Barrier(comm);
    dtWrite += writeStars(bodyPositions, bodyVelocities, bodiesIDs,
                          bodyDensRho, bodyDrvt, bodyHydro,
                          out,ntypeloc);
   





	//BonsaiIO statistics 
    MPI_Reduce(&ntypeloc, &ntypeglb, ntypeloc.size(), MPI_LONG_LONG, MPI_SUM, 0, comm);
    if (rank == 0)
    {
      size_t nsum = 0;
      for (int type = 0; type < (int)ntypeloc.size(); type++)
      {
        nsum += ntypeglb[type];
        if (ntypeglb[type] > 0)
          fprintf(stderr, "ptype= %d:  np= %zu \n",type, ntypeglb[type]);
      }
    }
  

    double dtWriteGlb;
    MPI_Allreduce(&dtWrite, &dtWriteGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "write file in %g sec \n", dtWriteGlb);
  

    const double tClose = MPI_Wtime();
    out.close();
    double dtCloseLoc = MPI_Wtime() - tClose;
    double dtCloseGlb;
    MPI_Allreduce(&dtCloseLoc, &dtCloseGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
    if (rank == 0) fprintf(stderr, "close time in %g sec \n", dtCloseGlb);

    if (rank == 0)
    {
      out.getHeader().printFields();
      fprintf(stderr, " Bandwidth= %g MB/s\n", out.computeBandwidth()/1e6);
    }
  }
  double dtAllLoc = MPI_Wtime() - tAll;
  double dtAllGlb;
  MPI_Allreduce(&dtAllLoc, &dtAllGlb, 1, MPI_DOUBLE, MPI_MAX,comm);
  if (rank == 0)
    fprintf(stderr, "All operations done in   %g sec \n", dtAllGlb);


  MPI_Finalize();


  return 0;
}


