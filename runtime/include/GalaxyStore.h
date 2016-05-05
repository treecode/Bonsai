/*
 * GalaxyStore.h
 *
 *  Created on: May 3, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef GALAXYSTORE_H_
#define GALAXYSTORE_H_

#include "Galaxy.h"
#include <vector>
#include <dirent.h>

class GalaxyStore
{
  public:
    GalaxyStore(std::string const& path)
    {
//	    DIR *dirp = opendir(path.c_str());
//	    dirent *dp;
//	    while ((dp = readdir(dirp)))
//	    {
//            if (dp->d_name != "tipsy") continue;
//            std::vector<real4> pos;
//            std::vector<real4> vel;
//            std::vector<int> ids;
//            std::vector<real4> pos_dust;
//            std::vector<real4> vel_dust;
//            std::vector<int> ids_dust;
//            //read_tipsy_file_parallel(pos, vel, ids, 0, dp->d_name, 0, 1, 0, 0, 0, 0, this,
//            //    pos_dust, vel_dust, ids_dust, 1, 1, false);
//            galaxies.push_back(Galaxy(pos, vel, ids));
//        }
//        closedir(dirp);
    }

    std::vector<Galaxy> galaxies;
};

#endif /* GALAXYSTORE_H_ */
