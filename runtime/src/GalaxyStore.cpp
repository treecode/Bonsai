/*
 * GalaxyStore.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "FileIO.h"
#include "GalaxyStore.h"
#include <stdexcept>
#include <set>
#include <string>

const double pi = std::acos(-1);

void GalaxyStore::init(std::string const& path, octree *tree)
{
	for (int i = 0;; ++i)
	{
		std::string filename = path + "/galaxy_type_" + std::to_string(i) + ".tipsy";
		if (access(filename.c_str(), F_OK) == -1) break;
		std::cout << "Read file " << filename << " into GalaxyStore." << std::endl;

		Galaxy galaxy;
		int Total2 = 0;
		int NFirst = 0;
		int NSecond = 0;
		int NThird = 0;

		read_tipsy_file_parallel(galaxy.pos, galaxy.vel, galaxy.ids,
			0.0, filename.c_str(), 0, 1, Total2, NFirst, NSecond, NThird, tree,
			galaxy.pos_dust, galaxy.vel_dust, galaxy.ids_dust, 50, 1, false);

		real4 cm = galaxy.getCenterOfMass();
		std::cout << "Center of mass = " << cm.x << " " << cm.y << " " << cm.z << std::endl;
		real4 tv = galaxy.getTotalVelocity();
		std::cout << "Total_velocity = " << tv.x << " " << tv.y << " " << tv.z << std::endl;

		galaxy.centering();
		galaxy.steady();

		cm = galaxy.getCenterOfMass();
		std::cout << "Center of mass = " << cm.x << " " << cm.y << " " << cm.z << std::endl;
		tv = galaxy.getTotalVelocity();
		std::cout << "Total_velocity = " << tv.x << " " << tv.y << " " << tv.z << std::endl;

		galaxies.push_back(galaxy);
	}
}

Galaxy GalaxyStore::getGalaxy(int galaxy_id) const
{
	if (galaxy_id < 0 or galaxy_id >= galaxies.size())
		std::cout << "WARNING: Requested qalaxy " << galaxy_id << " is not available." << std::endl;
	return galaxies[galaxy_id];
}
