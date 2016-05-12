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
#include <stdexcept>

struct sockaddr_in;

#define BUFFERSIZE 1024

class WOGSocketManager
{
 public:
  WOGSocketManager(int port)
  {
    serverSocket = socket(AF_INET, SOCK_STREAM, 0);
    if (serverSocket == -1) {
      perror("socket");
      throw std::runtime_error("socket error");
    }

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = htons(port);
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    if (bind(serverSocket, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr)) == -1) {
      perror("bind");
      throw std::runtime_error("bind error");
    }

    // wait for a client
    if (listen(serverSocket, 5) == -1) {
      perror("listen");
      throw std::runtime_error("listen error");
    }

    sockaddr_in clientAddr;
    socklen_t sin_size = sizeof(struct sockaddr_in);
    transferSocket = accept(serverSocket, (struct sockaddr*)&clientAddr, &sin_size);
  	if (transferSocket == -1) {
      perror("accept");
      throw std::runtime_error("accept error");
  	}

    fcntl(transferSocket, F_SETFL, O_NONBLOCK);
  }

  ~WOGSocketManager()
  {
    close(transferSocket);
    close(serverSocket);
  }

  bool getRequest()
  {
	char buffer[BUFFERSIZE];
    int n = recv(transferSocket, buffer, sizeof(buffer), 0);
    if (n > 0) {
      buffer[n] = '\0';
      std::cout << "The string is: " << buffer << std::endl;
      std::string message = "1";
      if (send(transferSocket, message.c_str(), message.size(), 0) == -1) perror("send");
      return true;
    }
    return false;
  }

 private:

  int serverSocket;
  int transferSocket;

};

#endif /* WOGSOCKETMANAGER_H_ */
