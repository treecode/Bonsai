/*
 * WOGSocketManager.h
 *
 *  Created on: May 11, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef WOGSOCKETMANAGER_H_
#define WOGSOCKETMANAGER_H_

#include "octree.h"
#include "GalaxyStore.h"
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

/// Control interconnection via UNIX socket
class WOGSocketManager
{
 public:

  /// Constructor opening sockets and reading input galaxies
  WOGSocketManager(int port);

  /// Constructor closing the sockets
  ~WOGSocketManager();

  /// Execute a client request
  void execute(octree *tree, GalaxyStore const& galaxyStore);

 private:

  int server_socket;

  int client_socket;

  /// Buffer size for socket data transmission
  static const int buffer_size = 1024;

  /// Number of users
  static const int number_of_users = 4;

  /// Maximal number of particles of a user
  static const int max_number_of_particles_of_user = 100000;

  /// Number of particles of user
  std::array<int, number_of_users> user_particles;

};

#endif /* WOGSOCKETMANAGER_H_ */
