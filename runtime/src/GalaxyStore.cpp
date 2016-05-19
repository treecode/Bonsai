/*
 * GalaxyStore.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "FileIO.h"
#include "GalaxyStore.h"
#include <stdexcept>
#include <string>

const double pi = std::acos(-1);

void GalaxyStore::init(std::string const& path, octree *tree)
{
	std::vector<std::string> filenames;
	filenames.push_back("galaxy_type_1.tipsy");
	filenames.push_back("galaxy_type_2.tipsy");
	filenames.push_back("galaxy_type_3.tipsy");
	filenames.push_back("galaxy_type_4.tipsy");

	for (std::vector<std::string>::const_iterator iterFileCur(filenames.begin()), iterFileEnd(filenames.end());
	    iterFileCur != iterFileEnd; ++iterFileCur)
	{
		std::cout << "Read file " << *iterFileCur << " into GalaxyStore." << std::endl;

		Galaxy galaxy;
		int Total2 = 0;
		int NFirst = 0;
		int NSecond = 0;
		int NThird = 0;

		read_tipsy_file_parallel(galaxy.pos, galaxy.vel, galaxy.ids,
			0.0, (path + "/" + *iterFileCur).c_str(), 0, 1, Total2, NFirst, NSecond, NThird, tree,
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

Galaxy GalaxyStore::getGalaxy(int user_id, int galaxy_id, double angle, double velocity) const
{
	Galaxy galaxy(galaxies[galaxy_id - 1]);

	galaxy.set_id(user_id - 1);

	double sinus = sin(angle * pi / 180);
	double cosinus = cos(angle * pi / 180);

	if (user_id == 1) {
	  galaxy.translate( 100,  100, 0);
      galaxy.accelerate(velocity * -sinus, velocity * -cosinus, 0);
	} else if (user_id == 2) {
      galaxy.translate( 100, -100, 0);
      galaxy.accelerate(velocity * -cosinus, velocity * sinus, 0);
	} else if (user_id == 3) {
      galaxy.translate(-100, -100, 0);
      galaxy.accelerate(velocity * sinus, velocity * cosinus, 0);
	} else if (user_id == 4) {
      galaxy.translate(-100,  100, 0);
      galaxy.accelerate(velocity * cosinus, velocity * -sinus, 0);
	}

    return galaxy;
}
