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

WOGSocketManager::WOGSocketManager(int port, int window_width, int window_height, real fovy,
  real farZ, real camera_distance, real deletion_radius_factor)
 : user_particles{{0, 0, 0, 0}},
   window_width(window_width),
   window_height(window_height),
   fovy(fovy),
   farZ(farZ),
   camera_distance(camera_distance),
   simulation_plane_width(0.0),
   simulation_plane_height(0.0),
   deletion_radius_factor(deletion_radius_factor),
   deletion_radius_square(0.0)
{
  reshape(window_width, window_height);

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
  remove_particles(tree);

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

void WOGSocketManager::reshape(int width, int height)
{
  window_width = width;
  window_height = height;
  real aspect_ratio = static_cast<real>(window_width) / window_height;

  simulation_plane_height = 2 * camera_distance * tan(fovy * M_PI / 360.0);
  simulation_plane_width = simulation_plane_height * aspect_ratio;

  std::cout << "window_width = " << window_width << std::endl;
  std::cout << "window_height = " << window_height << std::endl;
  std::cout << "aspect_ratio = " << aspect_ratio << std::endl;
  std::cout << "simulation_plane_height = " << simulation_plane_height << std::endl;
  std::cout << "simulation_plane_width = " << simulation_plane_width << std::endl;

  real4 rear_corner;
  rear_corner.y = camera_distance * tan(fovy * M_PI / 360.0);
  rear_corner.x = rear_corner.y * aspect_ratio;
  rear_corner.z = farZ - camera_distance;
  deletion_radius_square = rear_corner.x * rear_corner.x + rear_corner.y * rear_corner.y + rear_corner.z * rear_corner.z;
  deletion_radius_square *= deletion_radius_factor * deletion_radius_factor;
  #ifdef DEBUG_PRINT
    std::cout << "deletion_radius_factor = " << deletion_radius_factor << std::endl;
    std::cout << "deletion_radius = " << std::sqrt(deletion_radius_square) << std::endl;
  #endif
}

void WOGSocketManager::remove_particles(octree *tree)
{
  tree->removeParticles(deletion_radius_square, user_particles);
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

    real4 position;
    position.x = vector_position.size() > 0 ? vector_position[0] * simulation_plane_width : 0.0;
    position.y = vector_position.size() > 1 ? vector_position[1] * simulation_plane_height : 0.0;
    position.z = vector_position.size() > 2 ? vector_position[2] : 0.0;

    // Shift center to lower left corner
    position.x -= simulation_plane_width * 0.5;
    position.y -= simulation_plane_height * 0.5;

    real4 velocity;
    velocity.x = vector_velocity.size() > 0 ? vector_velocity[0] * window_width / simulation_plane_width : 0.0;
    velocity.y = vector_velocity.size() > 1 ? vector_velocity[1] * window_height / simulation_plane_height : 0.0;
    velocity.z = vector_velocity.size() > 2 ? vector_velocity[2] : 0.0;

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
      std::cout << "galaxy_id: " << galaxy_id << std::endl;
      std::cout << "position: " << position.x << " " << position.y << " " << position.z << std::endl;
      std::cout << "velocity: " << velocity.x << " " << velocity.y << " " << velocity.z << std::endl;
    #endif

	Galaxy galaxy = galaxyStore.getGalaxy(galaxy_id);
	galaxy.translate(position);
    galaxy.accelerate(velocity);

    // Since the particle ids are not needed for the simulation, we use them to store the user_id in the first digit.
	for (auto & id : galaxy.ids) id = id - id % 10 + user_id;

    tree->releaseGalaxy(galaxy);
	user_particles[user_id] += galaxy.pos.size();

	send_data["response"] = "Galaxy with " + std::to_string(galaxy.pos.size()) + " particles of user " + std::to_string(user_id) + " was released.";
  }
  else if (task == "remove")
  {
    int user_id = recv_data["user_id"].as<int>();

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
    #endif

    tree->removeGalaxy(user_id);
    user_particles[user_id] = 0;

	send_data["response"] = "All particles of user " + std::to_string(user_id) + " were removed.";
  }
  else if (task == "report")
  {
    // Simulation time in MYears, for factor see renderloop.cpp, search for MYears
	send_data["simulation time"] = tree->getTime() * 9.78;
	send_data["user 0"] = user_particles[0];
	send_data["user 1"] = user_particles[1];
	send_data["user 2"] = user_particles[2];
	send_data["user 3"] = user_particles[3];
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
