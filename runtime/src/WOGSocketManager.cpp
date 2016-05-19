/*
 * WOGSocketManager.cpp
 *
 *  Created on: May 17, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "WOGSocketManager.h"
#include "jsoncons/json.hpp"

using jsoncons::json;

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
  char buffer[BUFFERSIZE];
  int n = recv(transferSocket, buffer, BUFFERSIZE, 0);
  if (n <= 0) return;

  buffer[n] = '\0';
  std::string buffer_string(buffer);
  std::cout << "The string is: " << buffer_string << std::endl;

  std::istringstream iss(buffer_string);
  json recv_data;
  iss >> recv_data;

  std::cout << "task: " << recv_data["task"].as<std::string>() << std::endl;
  std::cout << "user_id: " << recv_data["user_id"].as<int>() << std::endl;
  std::cout << "galaxy_id: " << recv_data["galaxy_id"].as<int>() << std::endl;
  std::cout << "angle: " << recv_data["angle"].as<double>() << std::endl;
  std::cout << "velocity: " << recv_data["velocity"].as<double>() << std::endl;

  if (recv_data["task"].as<std::string>() == "release")
  {
    tree->releaseGalaxy(galaxyStore.getGalaxy(
      recv_data["user_id"].as<int>(),
      recv_data["galaxy_id"].as<int>(),
      recv_data["angle"].as<double>(),
      recv_data["velocity"].as<double>()));
  }
  else if (recv_data["task"].as<std::string>() == "remove")
  {
	tree->removeGalaxy(recv_data["user_id"].as<int>());
  }

  json send_data;
  send_data["last_operation"] = recv_data["task"].as<std::string>();

  std::ostringstream oss;
  oss << send_data;
  std::string send_data_string = oss.str();

  if (send(transferSocket, send_data_string.c_str(), send_data_string.size(), 0) == -1) perror("send");
}
