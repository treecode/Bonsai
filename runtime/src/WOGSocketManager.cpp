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

  std::cout << "Bind server socket to port " << std::to_string(port) << std::endl;

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
  // Remove particles
  //remove_particles(tree);

  // Check for user request
  char buffer[buffer_size];
  int n = recv(client_socket, buffer, buffer_size, 0);
  if (n <= 0) return;
  buffer[n] = '\0';

  try {
    execute_json(tree, galaxyStore, buffer);
  } catch ( ... ) {
    json send_data;
	send_data["response"] = "Can't interpret last request";

    std::ostringstream oss;
    oss << send_data;
    std::string send_data_string = oss.str();

    if (send(client_socket, send_data_string.c_str(), send_data_string.size(), 0) == -1) perror("send");
  }
}

void WOGSocketManager::remove_particles(octree *tree)
{
  tree->removeParticles();
}

void WOGSocketManager::execute_json(octree *tree, GalaxyStore const& galaxyStore, std::string buffer)
{
  #ifdef DEBUG_PRINT
    std::cout << "The string is: " << buffer << std::endl;
  #endif

  std::istringstream iss(buffer);
  json recv_data;
  iss >> recv_data;

  std::string task = recv_data["task"].as<std::string>();

  #ifdef DEBUG_PRINT
    std::cout << "task: " << task << std::endl;
  #endif

  json send_data;

  if (task == "release")
  {
    int user_id = recv_data["user_id"].as<int>();
    int galaxy_id = recv_data["galaxy_id"].as<int>();
    std::vector<double> vector_position = recv_data["position"].as<std::vector<double>>();
    std::vector<double> vector_velocity = recv_data["velocity"].as<std::vector<double>>();

    double camera_distance = 500.0;
    double fovy = 60.0;
    double screen_high = 2 * camera_distance * tan(fovy * M_PI / 360.0);
    double screen_width = screen_high * m_windowDims.x / m_windowDims.y;

    real4 position;
    position.x = vector_position.size() > 0 ?  vector_position[0] * screen_width : 0.0;
    position.y = vector_position.size() > 1 ? -vector_position[1] * screen_high : 0.0;
    position.z = vector_position.size() > 2 ?  vector_position[2] : 0.0;

    // Shift center to upper left corner
    position.x -= screen_high * 0.5;
    position.y += screen_width * 0.5;

    real4 velocity;
    velocity.x = vector_velocity.size() > 0 ?  vector_velocity[0] * 100 : 0.0;
    velocity.y = vector_velocity.size() > 1 ? -vector_velocity[1] * 100 : 0.0;
    velocity.z = vector_velocity.size() > 2 ?  vector_velocity[2] * 100 : 0.0;

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
      std::cout << "galaxy_id: " << galaxy_id << std::endl;
      std::cout << "position: " << position.x << " " << position.y << " " << position.z << std::endl;
      std::cout << "velocity: " << velocity.x << " " << velocity.y << std::endl;
    #endif

	Galaxy galaxy = galaxyStore.getGalaxy(galaxy_id);
	galaxy.translate(position);
    galaxy.accelerate(velocity);

	for (auto & id : galaxy.ids) id = id - id % 10 + user_id;

	// Remove particles first if user exceeded his limit
//	int particles_to_remove = user_particles[user_id - 1] + galaxy.pos.size() - max_number_of_particles_of_user;
//	if (particles_to_remove > 0) {
//      tree->removeGalaxy(user_id, particles_to_remove);
//      user_particles[user_id - 1] -= particles_to_remove;
//	}

    tree->releaseGalaxy(galaxy);
	user_particles[user_id - 1] += galaxy.pos.size();

	send_data["response"] = "Galaxy was released.";
  }
  else if (task == "remove")
  {
    int user_id = recv_data["user_id"].as<int>();

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
    #endif

    tree->removeGalaxy(user_id, user_particles[user_id - 1]);
    user_particles[user_id - 1] = 0;

	send_data["response"] = "All particles of user " + std::to_string(user_id) + " were removed.";
  }
  else if (task == "report")
  {
	send_data["user_1"] = user_particles[0];
	send_data["user_2"] = user_particles[1];
	send_data["user_3"] = user_particles[2];
	send_data["user_4"] = user_particles[3];
  }
  else
  {
    throw std::runtime_error("Unkown task: " + task);
  }

  std::ostringstream oss;
  oss << send_data;
  std::string send_data_string = oss.str();

  if (send(client_socket, send_data_string.c_str(), send_data_string.size(), 0) == -1) perror("send");
}
