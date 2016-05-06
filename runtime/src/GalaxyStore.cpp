/*
 * GalaxyStore.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "FileIO.h"
#include "GalaxyStore.h"
#include <stdexcept>
#include <dirent.h>

void GalaxyStore::init(std::string const& path, octree *tree)
{
	DIR *dirp = opendir(path.c_str());
	if (!dirp) throw ::std::runtime_error("Can't open GalaxyStore directory " + path);
	dirent *dp;
	while ((dp = readdir(dirp)))
	{
		if (dp->d_name != "tipsy") continue;

		Galaxy galaxy;
		int Total2 = 0;
		int NFirst = 0;
		int NSecond = 0;
		int NThird = 0;

		read_tipsy_file_parallel(galaxy.pos, galaxy.vel, galaxy.ids,
			0.0, dp->d_name, 0, 1, Total2, NFirst, NSecond, NThird, tree,
			galaxy.pos_dust, galaxy.vel_dust, galaxy.ids_dust, 1, 1, false);

		galaxies.push_back(galaxy);
	}
	closedir(dirp);
}