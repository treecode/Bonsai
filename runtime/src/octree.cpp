#include "octree.h"

#ifndef WIN32
#include <sys/time.h>
#endif

/*********************************/
/*********************************/
/*********************************/



void octree::set_src_directory(string src_dir) {                                                                                                                                 
    this->src_directory = (char*)src_dir.c_str();                                                                                                                                
}   

double octree::get_time() {
#ifdef WIN32
  if (sysTimerFreq.QuadPart == 0)
  {
    return -1.0;
  }
  else
  {
    LARGE_INTEGER c;
    QueryPerformanceCounter(&c);
    return static_cast<double>( (double)(c.QuadPart - sysTimerAtStart.QuadPart) / sysTimerFreq.QuadPart );
  }
#else
  struct timeval Tvalue;
  struct timezone dummy;
  
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +1.e-6*((double) Tvalue.tv_usec));
#endif
}

int octree::getAllignmentOffset(int n)
{
  const int allignBoundary = 128*sizeof(uint); //Fermi,128 bytes 
  
  int offset = 0;
  //Compute the number of bytes  
  offset = n*sizeof(uint); 
  //Compute number of 256 byte blocks  
  offset = (offset / allignBoundary) + (((offset % allignBoundary) > 0) ? 1 : 0); 
  //Compute the number of bytes padded / offset 
  offset = (offset * allignBoundary) - n*sizeof(uint); 
  //Back to the actual number of elements
  offset = offset / sizeof(uint);   
  
  return offset;
}

int octree::getTextureAllignmentOffset(int n, int size)
{
    const int texBoundary = TEXTURE_BOUNDARY; //Fermi
  
    int textOffset = 0;
    //Compute the number of bytes  
    textOffset = n*size; 
    //Compute number of texBoundary byte blocks  
    textOffset = (textOffset / texBoundary) + (((textOffset % texBoundary) > 0) ? 1 : 0); 
    //Compute the number of bytes padded / offset 
    textOffset = (textOffset * texBoundary) - n*size; 
    //Back to the actual number of elements
    textOffset = textOffset / size; 
    
    return textOffset;
}   


/*********************************/
/******** Input functions  ******/
/*********************************/


/*********************************/
/******** Output functions  ******/
/*********************************/



void octree::write_snapshot_per_process(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds,
                                        int n, string fileName, float time)
{
    NTotal = n;
    NFirst = NSecond = NThird = 0;

    for(int i=0; i < n; i++)
    {
      if(bodyIds[i] >= DISKID  && bodyIds[i] < BULGEID)       NThird++;
      if(bodyIds[i] >= BULGEID && bodyIds[i] < DARKMATTERID)  NSecond++;
      if(bodyIds[i] >= DARKMATTERID)                          NFirst++;
    }

    //Sync them to process 0
    int NCombTotal, NCombFirst, NCombSecond, NCombThird;
    NCombTotal  = (NTotal);
    NCombFirst  = (NFirst);
    NCombSecond = (NSecond);
    NCombThird  = (NThird);

    //Since the bulge and disk particles are star particles, lets add them
    NCombSecond += NCombThird;


    ofstream outputFile;
    outputFile.open(fileName.c_str(), ios::out | ios::binary);

    dumpV2  h;

    if(!outputFile.is_open())
    {
      cout << "Can't open output file: "<< fileName << std::endl;
      ::exit(0);
    }

    //Create Tipsy header
    h.time    = time;
    h.nbodies = NCombTotal;
    h.ndim    = 3;
    h.ndark   = NCombFirst;
    h.nstar   = NCombSecond;    //Incase of disks we have to finish this
    h.nsph    = 0;
    h.version = 2;
    outputFile.write((char*)&h, sizeof(h));

    //First write the dark matter particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(bodyIds[i] >= DARKMATTERID)
      {
        //Set particle properties
        dark_particleV2 d;
        //d.eps = bodyVelocities[i].w;
        d.mass = bodyPositions[i].w;
        d.pos[0] = bodyPositions[i].x;
        d.pos[1] = bodyPositions[i].y;
        d.pos[2] = bodyPositions[i].z;
        d.vel[0] = bodyVelocities[i].x;
        d.vel[1] = bodyVelocities[i].y;
        d.vel[2] = bodyVelocities[i].z;
        d.setID(bodyIds[i]);      //Custom change to Tipsy format

        outputFile.write((char*)&d, sizeof(d));
      } //end if
    } //end i loop

    //Next write the star particles
    for(int i=0; i < NCombTotal ; i++)
    {
      //Set particle properties
      star_particleV2 s;
      //s.eps = bodyVelocities[i].w;
      s.mass = bodyPositions[i].w;
      s.pos[0] = bodyPositions[i].x;
      s.pos[1] = bodyPositions[i].y;
      s.pos[2] = bodyPositions[i].z;
      s.vel[0] = bodyVelocities[i].x;
      s.vel[1] = bodyVelocities[i].y;
      s.vel[2] = bodyVelocities[i].z;
      s.setID(bodyIds[i]);      //Custom change to tipsy format

      s.metals = 0;
      s.tform = 0;
      outputFile.write((char*)&s, sizeof(s));

    } //end i loop

    outputFile.close();

    LOGF(stderr,"Wrote %d bodies to tipsy file \n", NCombTotal);

}

void octree::write_dumbp_snapshot_parallel_tipsyV2(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds, int n, string fileName,
                                                 int NCombTotal, int NCombFirst, int NCombSecond, int NCombThird, float time)
{
  #ifdef TIPSYOUTPUT
  //Rank 0 does the writing
  if(mpiGetRank() == 0)
  {
    ofstream outputFile;
    outputFile.open(fileName.c_str(), ios::out | ios::binary);

    dumpV2  h;

    if(!outputFile.is_open())
    {
      cout << "Can't open output file: "<< fileName << std::endl;
      ::exit(0);
    }

    //Create Tipsy header
    h.time    = time;
    h.nbodies = NCombTotal;
    h.ndim    = 3;
    h.ndark   = NCombFirst;
    h.nstar   = NCombSecond;    //In case of disks we have to finish this
    h.nsph    = 0;
    h.version = 2;

    outputFile.write((char*)&h, sizeof(h));

    //Buffer to store complete snapshot
    vector<real4>   allPositions;
    vector<real4>   allVelocities;
    vector<ullong>  allIds;

    allPositions.insert(allPositions.begin(), &bodyPositions[0], &bodyPositions[n]);
    allVelocities.insert(allVelocities.begin(), &bodyVelocities[0], &bodyVelocities[n]);
    allIds.insert(allIds.begin(), &bodyIds[0], &bodyIds[n]);

    //Now receive the data from the other processes
    vector<real4>   extPositions;
    vector<real4>   extVelocities;
    vector<ullong>  extIds;

    for(int recvFrom=1; recvFrom < mpiGetNProcs(); recvFrom++)
    {
      ICRecv(recvFrom, extPositions, extVelocities,  extIds);

      allPositions.insert(allPositions.end(), extPositions.begin(), extPositions.end());
      allVelocities.insert(allVelocities.end(), extVelocities.begin(), extVelocities.end());
      allIds.insert(allIds.end(), extIds.begin(), extIds.end());
    }



    //First write the dark matter particles
    for(int i=0; i < NCombTotal ; i++)
    {
      if(allIds[i] >= DARKMATTERID)
      {
        //Set particle properties
        dark_particleV2 d;
        //d.eps = allVelocities[i].w;
        d.mass = allPositions[i].w;
        d.pos[0] = allPositions[i].x;
        d.pos[1] = allPositions[i].y;
        d.pos[2] = allPositions[i].z;
        d.vel[0] = allVelocities[i].x;
        d.vel[1] = allVelocities[i].y;
        d.vel[2] = allVelocities[i].z;
        //d.ID[0] = allIds[i];      //Custom change to tipsy format
        d.setID(allIds[i]);

        outputFile.write((char*)&d, sizeof(d));
      } //end if
    } //end i loop


    //Next write the star particles
    for(int i=0; i < NCombTotal ; i++)
    {
      {
        //Set particle properties
        star_particleV2 s;
        //s.eps = allVelocities[i].w;
        s.mass = allPositions[i].w;
        s.pos[0] = allPositions[i].x;
        s.pos[1] = allPositions[i].y;
        s.pos[2] = allPositions[i].z;
        s.vel[0] = allVelocities[i].x;
        s.vel[1] = allVelocities[i].y;
        s.vel[2] = allVelocities[i].z;
        s.setID(allIds[i]);

        s.metals = 0;
        s.tform = 0;
        outputFile.write((char*)&s, sizeof(s));
//         outputFile << s;
     } //end if
    } //end i loop

    outputFile.close();

//    fprintf(stdout, "Wrote %d bodies to tipsy file \n", NCombTotal);
//    fprintf(stdout, "Test: %d %d  and long: %d \t header: %d %d  test: %d\n",
//        sizeof(dark_particle), sizeof(dark_particleV2),
//        sizeof(long), sizeof(dump), sizeof(dumpV2), sizeof(ulonglong1));
  }
  else
  {
    //All other ranks send their data to proess 0
    ICSend(0,  bodyPositions, bodyVelocities,  bodyIds, n);
  }
 #endif
}

void octree::write_dumbp_snapshot_parallel(real4 *bodyPositions, real4 *bodyVelocities, ullong* bodyIds, int n, string fileName, float time)
{
    NTotal = n;
    NFirst = NSecond = NThird = 0;
    
    for(int i=0; i < n; i++)
    { 
      //Specific for JB his files
      if(bodyIds[i] >= DISKID  && bodyIds[i] < BULGEID)       NThird++;
      if(bodyIds[i] >= BULGEID && bodyIds[i] < DARKMATTERID)  NSecond++;
      if(bodyIds[i] >= DARKMATTERID)                          NFirst++;
    }
    
    //Sync them to process 0
    int NCombTotal, NCombFirst, NCombSecond, NCombThird;
    NCombTotal  = SumOnRootRank(NTotal);
    NCombFirst  = SumOnRootRank(NFirst);
    NCombSecond = SumOnRootRank(NSecond);
    NCombThird  = SumOnRootRank(NThird);
    
    //Since the dust and disk particles are star particles
    //lets add them
    NCombSecond += NCombThird;    

    char fullFileName[256];
    sprintf(fullFileName, "%s", fileName.c_str());
    string tempName; tempName.assign(fullFileName);
    write_dumbp_snapshot_parallel_tipsyV2(bodyPositions, bodyVelocities, bodyIds, n, tempName,
                                        NCombTotal, NCombFirst, NCombSecond, NCombThird, time);
    return;

};

//Version that writes the BD dump format
//void octree::write_dumbp_snapshot(real4 *bodyPositions, real4 *bodyVelocities, int* bodyIds, int n, string fileName) {
//  char fullFileName[256];
//  sprintf(fullFileName, "%s", fileName.c_str());
//
//  cout << "Trying to write to file: " << fullFileName << endl;
//
//  ofstream outputFile(fullFileName, ios::out);
//
//  if(!outputFile.is_open())
//  {
//    cout << "Can't open output file \n";
//    exit(0);
//  }
//
//  outputFile.precision(16);
//
//  //Write the header
//  outputFile << NTotal << "\t" << NFirst << "\t" << NSecond << "\t" << NThird << endl;
//
//  for(int i=0; i < n ; i++)
//  {
//    outputFile << bodyIds[i] << "\t" << bodyPositions[i].w << "\t" << bodyPositions[i].x << "\t" <<
//                       bodyPositions[i].y       << "\t" << bodyPositions[i].z << "\t" <<
//                       bodyVelocities[i].x      << "\t" << bodyVelocities[i].y << "\t" <<
//                       bodyVelocities[i].z      << "\t" << bodyVelocities[i].w << endl;
//  }
//
//
//  outputFile.close();
//
//  fprintf(stdout, "Wrote %d bodies to dump file \n", n);
//};
//void octree::write_dumbp_snapshot_parallel_tipsy(real4 *bodyPositions, real4 *bodyVelocities, int* bodyIds, int n, string fileName,
//                                                 int NCombTotal, int NCombFirst, int NCombSecond, int NCombThird, float time)
//{
//  #ifdef TIPSYOUTPUT
//  //Rank 0 does the writing
//  if(mpiGetRank() == 0)
//  {
//    ofstream outputFile;
//    outputFile.open(fileName.c_str(), ios::out | ios::binary);
//
//    dump  h;
//
//    if(!outputFile.is_open())
//    {
//      cout << "Can't open output file: "<< fileName << std::endl;
//      exit(0);
//    }
//
//    //Create tipsy header
//    h.time = time;
//    h.nbodies = NCombTotal;
//    h.ndim = 3;
//    h.ndark = NCombFirst;
//    h.nstar = NCombSecond;    //Incase of disks we have to finish this
//    h.nsph = 0;
//
//    outputFile.write((char*)&h, sizeof(h));
//
//    //Buffer to store complete snapshot
//    vector<real4> allPositions;
//    vector<real4> allVelocities;
//    vector<int> allIds;
//
//    allPositions.insert(allPositions.begin(), &bodyPositions[0], &bodyPositions[n]);
//    allVelocities.insert(allVelocities.begin(), &bodyVelocities[0], &bodyVelocities[n]);
//    allIds.insert(allIds.begin(), &bodyIds[0], &bodyIds[n]);
//
//    //Now receive the data from the other processes
//    vector<real4> extPositions;
//    vector<real4> extVelocities;
//    vector<int>   extIds;
//
//    for(int recvFrom=1; recvFrom < mpiGetNProcs(); recvFrom++)
//    {
//      ICRecv(recvFrom, extPositions, extVelocities,  extIds);
//
//      allPositions.insert(allPositions.end(), extPositions.begin(), extPositions.end());
//      allVelocities.insert(allVelocities.end(), extVelocities.begin(), extVelocities.end());
//      allIds.insert(allIds.end(), extIds.begin(), extIds.end());
//    }
//
//    //Frist write the dark matter particles
//    for(int i=0; i < NCombTotal ; i++)
//    {
//      if(allIds[i] >= 200000000 && allIds[i] < 300000000)
//      {
//        //Set particle properties
//        dark_particle d;
//        d.eps = allVelocities[i].w;
//        d.mass = allPositions[i].w;
//        d.pos[0] = allPositions[i].x;
//        d.pos[1] = allPositions[i].y;
//        d.pos[2] = allPositions[i].z;
//        d.vel[0] = allVelocities[i].x;
//        d.vel[1] = allVelocities[i].y;
//        d.vel[2] = allVelocities[i].z;
//        d.phi = allIds[i];      //Custom change to tipsy format
//
//        outputFile.write((char*)&d, sizeof(d));
//      } //end if
//    } //end i loop
//
//
//    //Next write the star particles
//    for(int i=0; i < NCombTotal ; i++)
//    {
////       if(allIds[i] >= 100000000 && allIds[i] < 200000000)
//      if(allIds[i] >= 0 && allIds[i] < 200000000)
//      {
//        //Set particle properties
//        star_particle s;
//        s.eps = allVelocities[i].w;
//        s.mass = allPositions[i].w;
//        s.pos[0] = allPositions[i].x;
//        s.pos[1] = allPositions[i].y;
//        s.pos[2] = allPositions[i].z;
//        s.vel[0] = allVelocities[i].x;
//        s.vel[1] = allVelocities[i].y;
//        s.vel[2] = allVelocities[i].z;
//        s.phi = allIds[i];      //Custom change to tipsy format
//
//        s.metals = 0;
//        s.tform = 0;
//        outputFile.write((char*)&s, sizeof(s));
////         outputFile << s;
//     } //end if
//    } //end i loop
//
//    outputFile.close();
//
//    fprintf(stdout, "Wrote %d bodies to tipsy file \n", NCombTotal);
//  }
//  else
//  {
//    //All other ranks send their data to proess 0
//    ICSend(0,  bodyPositions, bodyVelocities,  bodyIds, n);
//  }
// #endif
//}


/*********************************/
/*********************************/
/*********************************/

//Some old utility functions

void octree::to_binary(int key) {
  char binary[128];
  int n = sizeof(key)*8;
  sprintf(binary, "0b00 000 000 000 000 000 000 000 000 000 000 ");
  for (int i = n-1; i >= 0; i--) {
    if (i%3 == 2) 
      sprintf(binary, "%s ", binary);
    if (key & (1 << i)) {
      sprintf(binary, "%s1", binary);
    } else {
      sprintf(binary, "%s0", binary);
    }
  }
}

void octree::to_binary(uint2 key) {
  char binary[256];
  int n = sizeof(key.x)*8;
  sprintf(binary, "0b");
  for (int i = n-1; i >= 0; i--) {
    if (i%3 == 2) 
      sprintf(binary, "%s ", binary);
    if (key.x & (1 << i)) {
      sprintf(binary, "%s1", binary);
    } else {
      sprintf(binary, "%s0", binary);
    }
  }
  sprintf(binary, "%s ", binary);
  for (int i = n-1; i >= 0; i--) {
    if (i%3 == 2) 
      sprintf(binary, "%s ", binary);
    if (key.x & (1 << i)) {
      sprintf(binary, "%s1", binary);
    } else {
      sprintf(binary, "%s0", binary);
    }
  }
}

/*********************************/
/*********************************/
/*********************************/

uint2 octree::dilate3(int value) {
  unsigned int x;
  uint2 key;
  
  // dilate first 10 bits

  x = value & 0x03FF;
  x = ((x << 16) + x) & 0xFF0000FF;
  x = ((x <<  8) + x) & 0x0F00F00F;
  x = ((x <<  4) + x) & 0xC30C30C3;
  x = ((x <<  2) + x) & 0x49249249;
  key.y = x;

  // dilate second 10 bits

  x = (value >> 10) & 0x03FF;
  x = ((x << 16) + x) & 0xFF0000FF;
  x = ((x <<  8) + x) & 0x0F00F00F;
  x = ((x <<  4) + x) & 0xC30C30C3;
  x = ((x <<  2) + x) & 0x49249249;
  key.x = x;

  return key;
}

int octree::undilate3(uint2 key) {
  int x, value = 0;
  
  key.x = key.x & 0x09249249;
  key.y = key.y & 0x09249249;
  
  // undilate first 10 bits

  x = key.y & 0x3FFFF;
  x = ((x <<  4) + (x << 2) + x) & 0x0E070381;
  x = ((x << 12) + (x << 6) + x) & 0x0FF80001;
  x = ((x << 18) + x) & 0x0FFC0000;
  value = value | (x >> 18);
  
  x = (key.y >> 18) & 0x3FFFF;
  x = ((x <<  4) + (x << 2) + x) & 0x0E070381;
  x = ((x << 12) + (x << 6) + x) & 0x0FF80001;
  x = ((x << 18) + x) & 0x0FFC0000;
  value = value | (x >> 12);
  

  // undilate second 10 bits

  x = key.x & 0x3FFFF;
  x = ((x <<  4) + (x << 2) + x) & 0x0E070381;
  x = ((x << 12) + (x << 6) + x) & 0x0FF80001;
  x = ((x << 18) + x) & 0x0FFC0000;
  value = value | ((x >> 18) << 10);
  
  x = (key.x >> 18) & 0x3FFFF;
  x = ((x <<  4) + (x << 2) + x) & 0x0E070381;
  x = ((x << 12) + (x << 6) + x) & 0x0FF80001;
  x = ((x << 18) + x) & 0x0FFC0000;
  value = value | ((x >> 12) << 10);
  
  return value;
}


/*********************************/
/*********************************/
/*********************************/

uint2 octree::get_key(int3 crd) {
  uint2 key, key1;
  key  = dilate3(crd.x);

  key1 = dilate3(crd.y);
  key.x = key.x | (key1.x << 1);
  key.y = key.y | (key1.y << 1);

  key1 = dilate3(crd.z);
  key.x = key.x | (key1.x << 2);
  key.y = key.y | (key1.y << 2);

  return key;
}

int3 octree::get_crd(uint2 key) {

  int3 crd = make_int3(undilate3(key),
	                   undilate3(make_uint2(key.x >> 1, key.y >> 1)),
	                   undilate3(make_uint2(key.x >> 2, key.y >> 2)));

  return crd;
}

real4 octree::get_pos(uint2 key, real size, tree_structure &tree) {
  real4 pos;
  pos.w = size;
  
  int3 crd = get_crd(key);
  pos.x = crd.x*tree.domain_fac + tree.corner.x;
  pos.y = crd.y*tree.domain_fac + tree.corner.y;
  pos.z = crd.z*tree.domain_fac + tree.corner.z;

  return pos;
}

/*********************************/
/*********************************/
/*********************************/

//uint4 octree::get_mask(int level) {
//  int mask_levels = 3*std::max(MAXLEVELS - level, 0);
//  uint4 mask = {0x3FFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,0xFFFFFFFF};
//
//  if (mask_levels > 60)
//  {
//    mask.z = 0;
//    mask.y = 0;
//    mask.x = (mask.x >> (mask_levels - 60)) << (mask_levels - 60);
//  }
//  else if (mask_levels > 30) {
//    mask.z = 0;
//    mask.y = (mask.y >> (mask_levels - 30)) << (mask_levels - 30);
//  } else {
//    mask.z = (mask.z >> mask_levels) << mask_levels;
//  }
//
//  return mask;
//}

//int octree::cmp_uint4(uint4 a, uint4 b) {
//  if      (a.x < b.x) return -1;
//  else if (a.x > b.x) return +1;
//  else {
//    if       (a.y < b.y) return -1;
//    else  if (a.y > b.y) return +1;
//    else {
//      if       (a.z < b.z) return -1;
//      else  if (a.z > b.z) return +1;
//      return 0;
//    } //end z
//  }  //end y
//} //end x, function
//
////Binary search of the key within certain bounds (cij.x, cij.y)
//int octree::find_key(uint4 key, uint2 cij, uint4 *keys) {
//  int l = cij.x;
//  int r = cij.y - 1;
//  while (r - l > 1) {
//    int m = (r + l) >> 1;
//    int cmp = cmp_uint4(keys[m], key);
//    if (cmp == -1) {
//      l = m;
//    } else {
//      r = m;
//    }
//  }
//  if (cmp_uint4(keys[l], key) >= 0) return l;
//
//  return r;
//}



/*
uint2 octree::get_mask(int level) {
  int mask_levels = 3*max(MAXLEVELS - level, 0);
  uint2 mask = {0x3FFFFFFF, 0xFFFFFFFF};
    
  if (mask_levels > 30) {
    mask.y = 0;
    mask.x = (mask.x >> (mask_levels - 30)) << (mask_levels - 30);
  } else {
    mask.y = (mask.y >> mask_levels) << mask_levels;
  }
  
  return mask;
}

uint2 octree::get_imask(uint2 mask) {
  return make_uint2(0x3FFFFFFF ^ mask.x, 0xFFFFFFFF ^ mask.y);
}
*/
/*********************************/
/*********************************/
/*********************************/
/*
int octree::find_key(uint2          key, 
		     vector<uint2> &keys,
		     int l,
		     int r) {
  r--;
  while (r - l > 1) {
    int m = (r + l) >> 1;
    int cmp = cmp_uint2(keys[m], key);
    if (cmp == -1) {
      l = m;
    } else { 
      r = m;
    }
  }
  
  if (cmp_uint2(keys[l], key) >= 0) return l;
  
  return r;
}

int octree::find_key(uint2                  key, 
		     vector<morton_struct> &keys,
		     int l,
		     int r) {
  r--;
  while (r - l > 1) {
    int m = (r + l) >> 1;
    int cmp = cmp_uint2(keys[m].key, key);
    if (cmp == -1) {
      l = m;
    } else { 
      r = m;
    }
  }
  
  if (cmp_uint2(keys[l].key, key) >= 0) return l;
  
  return r;
}*/

/*********************************/
/*********************************/
/*********************************/

