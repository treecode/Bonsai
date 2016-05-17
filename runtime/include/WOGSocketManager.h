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

class WOGSocketManager
{
 public:

  // Constructor opening sockets and reading input galaxies
  WOGSocketManager(int port);

  // Constructor closing the sockets
  ~WOGSocketManager();

  // Release Galaxy if requested by socket
  void release(octree *tree, GalaxyStore const& galaxyStore);

 private:

  int serverSocket;
  int transferSocket;

};

#endif /* WOGSOCKETMANAGER_H_ */
