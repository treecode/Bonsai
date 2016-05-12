/*
 * WOGSocketManager.h
 *
 *  Created on: May 11, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#ifndef WOGSOCKETMANAGER_H_
#define WOGSOCKETMANAGER_H_

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

struct sockaddr_in;

class WOGSocketManager
{
 public:
  WOGSocketManager(int port)
  {
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
      perror("socket");
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = port;
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr)) == -1) {
      perror("bind");
    }

    // wait for a client
    /* listen (this socket, request queue length) */
    if (listen(serverSocket, 5) == -1) {
        perror("listen");
    }

    sockaddr_in clientAddr;
    socklen_t sin_size = sizeof(struct sockaddr_in);
    transferSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &sin_size);
  	if (transferSocket == -1) perror("accept");

    fcntl(transferSocket, F_SETFL, O_NONBLOCK);
  }

  ~WOGSocketManager()
  {
    close(transferSocket);
    close(serverSocket);
  }


  int n = recv(wogSocket, buffer, sizeof(buffer), 0);
  if (n > 0) {
    buffer[n] = '\0';
    std::cout << "The string is: " << buffer << std::endl;
    if (send(wogSocket, "Hello, world!\n", 14, 0) == -1) perror("send");
    m_tree->releaseGalaxy(theDemo->m_galaxyStore.galaxies[0]);
  }

 private:

  int serverSocket;
  int transferSocket;

};

#endif /* WOGSOCKETMANAGER_H_ */
