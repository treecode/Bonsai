/*
 * WOGSocketManager.cpp
 *
 *  Created on: May 17, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "WOGSocketManager.h"
#include "jsoncons/json.hpp"

using jsoncons::json;

#define DEBUG_PRINT

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
 : user_particles{{0, 0, 0, 0}}
{
  server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket == -1) {
    perror("socket");
    throw std::runtime_error("socket error");
  }

  sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(port);
  serverAddr.sin_addr.s_addr = INADDR_ANY;

  if (bind(server_socket, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr)) == -1) {
    perror("bind");
    throw std::runtime_error("bind error");
  }

  // wait for a client
  if (listen(server_socket, 5) == -1) {
    perror("listen");
    throw std::runtime_error("listen error");
  }

  sockaddr_in clientAddr;
  socklen_t sin_size = sizeof(struct sockaddr_in);
  client_socket = accept(server_socket, (struct sockaddr*)&clientAddr, &sin_size);
  if (client_socket == -1) {
    perror("accept");
    throw std::runtime_error("accept error");
  }

  fcntl(client_socket, F_SETFL, O_NONBLOCK);
}

WOGSocketManager::~WOGSocketManager()
{
  close(client_socket);
  close(server_socket);
}

void WOGSocketManager::execute(octree *tree, GalaxyStore const& galaxyStore)
{
  char buffer[BUFFERSIZE];
  int n = recv(client_socket, buffer, BUFFERSIZE, 0);
  if (n <= 0) return;

  buffer[n] = '\0';
  std::string buffer_string(buffer);
#ifdef DEBUG_PRINT
  std::cout << "The string is: " << buffer_string << std::endl;
#endif

  std::istringstream iss(buffer_string);
  json recv_data;
  iss >> recv_data;

  std::string task = recv_data["task"].as<std::string>();
  int user_id = recv_data["user_id"].as<int>();

#ifdef DEBUG_PRINT
  std::cout << "task: " << task << std::endl;
  std::cout << "user_id: " << user_id << std::endl;
#endif

  if (task == "release")
  {
    int galaxy_id = recv_data["galaxy_id"].as<int>();
    double angle = recv_data["angle"].as<double>();
    double velocity = recv_data["velocity"].as<double>();

    #ifdef DEBUG_PRINT
      std::cout << "galaxy_id: " << galaxy_id << std::endl;
      std::cout << "angle: " << angle << std::endl;
      std::cout << "velocity: " << velocity << std::endl;
    #endif

	Galaxy galaxy = galaxyStore.getGalaxy(user_id, galaxy_id, angle, velocity);

	for (auto & id : galaxy.ids) id = id - id % 10 + user_id;

    tree->releaseGalaxy(galaxy);
	user_particles[user_id - 1] += galaxy.pos.size();
  }
  else if (task == "remove")
  {
    tree->removeGalaxy(user_id);
    user_particles[user_id - 1] = 0;
  }

  json send_data;
  send_data["last_operation"] = task;

  std::ostringstream oss;
  oss << send_data;
  std::string send_data_string = oss.str();

  if (send(client_socket, send_data_string.c_str(), send_data_string.size(), 0) == -1) perror("send");
}
