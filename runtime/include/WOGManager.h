/*
 * WOGSocketManager.h
 *
 *  Created on: May 11, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#pragma once

#include "Galaxy.h"
#include "octree.h"
#include "jsoncons/json.hpp"
#include <array>
#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <stdexcept>

struct sockaddr_in;

/**
 * War of Galaxies (WOG)
 * - Store galaxy type which can be released
 * - Control interconnection via UNIX socket
 * - Add and remove particles to running simulation
 */
class WOGManager
{
 public:

  /// Constructor opening sockets and reading input galaxies
  WOGManager(octree *tree, std::string const& path, int port, int window_width, int window_height, real fovy,
	real farZ, real camera_distance, real deletion_radius_factor);

  /// Constructor closing the sockets
  ~WOGManager();

  /// Execute a client request
  void execute();

  /// Must be called by glutReshapeFunc
  void reshape(int width, int height);

 private:

  /// Read all galaxy types
  void read_galaxies(std::string const& path);

  /// Remove particles continuously
  void remove_particles();

  /// Execute a client request
  jsoncons::json execute_json(std::string const& buffer);

  octree *tree;

  int server_socket;

  int client_socket;

  /// Number of users
  static constexpr auto number_of_users = 4;

  /// Buffer size for socket data transmission
  static constexpr auto buffer_size = 1024;

  /// Maximal number of particles of a user
  static constexpr auto max_number_of_particles_of_user = 100000;

  /// Number of particles of user
  my_dev::dev_mem<uint> user_particles;

  /// Dimension of the window
  int window_width;
  int window_height;

  /// OpenGL viewing angle
  real fovy;

  /// OpenGL distance of clipping plane
  real farZ;

  /// Distance of the OpenGL camera
  real camera_distance;

  /// Dimension of the window
  real simulation_plane_width;
  real simulation_plane_height;

  /// Scaling factor for deletion sphere.
  real deletion_radius_factor;

  /// Squared radius of deletion sphere. Particles leaving this sphere will be removed.
  real deletion_radius_square;

  /// Galaxy types which can be released
  std::vector<Galaxy> galaxies;

};
