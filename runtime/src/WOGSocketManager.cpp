/*
 * WOGSocketManager.cpp
 *
 *  Created on: May 17, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "WOGSocketManager.h"

auto split(std::string const& s, char separator) -> std::vector<std::string>
{
  std::vector<std::string> result;
  std::string::size_type p = 0;
  std::string::size_type q;
  while ((q = s.find(separator, p)) != std::string::npos) {
    result.emplace_back(s, p, q - p);
    p = q + 1;
  }
  result.emplace_back(s, p);
  return result;
}

WOGSocketManager::WOGSocketManager(int port)
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

WOGSocketManager::~WOGSocketManager()
{
  close(transferSocket);
  close(serverSocket);
}

void WOGSocketManager::release(octree *tree, GalaxyStore const& galaxyStore)
{
  std::string buffer;
  buffer.resize(BUFFERSIZE);
  int n = recv(transferSocket, &buffer[0], BUFFERSIZE, 0);
  if (n <= 0) return;
  std::cout << "The string is: " << buffer << std::endl;

  auto values = split(buffer, '|');
  int user_id = std::stoi(values[0]);
  int galaxy_id = std::stoi(values[1]);
  double angle = std::stod(values[2]);
  double velocity = std::stod(values[3]);

  std::cout << "user_id: " << user_id << std::endl;
  std::cout << "galaxy_id: " << galaxy_id << std::endl;
  std::cout << "angle: " << angle << std::endl;
  std::cout << "velocity: " << velocity << std::endl;

  tree->releaseGalaxy(galaxyStore.getGalaxy(user_id, galaxy_id, angle, velocity));

  std::string message = "1";
  if (send(transferSocket, message.c_str(), message.size(), 0) == -1) perror("send");
}
