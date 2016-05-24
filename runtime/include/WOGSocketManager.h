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

#define BUFFERSIZE 1024
#define NUMBER_OF_USERS 4
#define DUMMY_PARTICLES 1
#define MAX_NUMBER_OF_PARTICLES_OF_USER 100000

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

  /// Number of particles of user
  std::array<int, NUMBER_OF_USERS> user_particles;

};

#endif /* WOGSOCKETMANAGER_H_ */
