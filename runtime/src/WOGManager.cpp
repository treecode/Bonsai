/*
 * WOGSocketManager.cpp
 *
 *  Created on: May 17, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include "read_tipsy_file_parallel.h"
#include "WOGManager.h"

using jsoncons::json;

#define DEBUG_PRINT

WOGManager::WOGManager(octree *tree, std::string const& path, int port, int window_width, int window_height, real fovy,
  real farZ, real camera_distance, real deletion_radius_factor)
 : tree(tree),
   server_socket(-1),
   client_socket(-1),
   user_particles(),
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
  user_particles.ccalloc(number_of_users);

  read_galaxies(path);
  reshape(window_width, window_height);

  server_socket = socket(AF_INET, SOCK_STREAM, 0);
  if (server_socket == -1) {
    perror("socket");
    throw std::runtime_error("socket error");
  }

  int enable = 1;
  if (setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0) {
    perror("setsockopt");
    throw std::runtime_error("setsockopt(SO_REUSEADDR) failed");
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

  // Set server_socket to non-blocking
  long save_fd = fcntl(server_socket, F_GETFL);
  save_fd |= O_NONBLOCK;
  fcntl(server_socket, F_SETFL, save_fd);
}

WOGManager::~WOGManager()
{
  close(client_socket);
  close(server_socket);
}

void WOGManager::read_galaxies(std::string const& path)
{
  for (int i = 0;; ++i)
  {
	std::string filename = path + "/galaxy_type_" + std::to_string(i) + ".tipsy";
	if (access(filename.c_str(), F_OK) == -1) break;
	std::cout << "Read file " << filename << " into GalaxyStore." << std::endl;

	Galaxy galaxy;
	int Total2 = 0;
	int NFirst = 0;
	int NSecond = 0;
	int NThird = 0;

	read_tipsy_file_parallel(galaxy.pos, galaxy.vel, galaxy.ids,
	  0.0, filename.c_str(), 0, 1, Total2, NFirst, NSecond, NThird, nullptr,
	  galaxy.pos_dust, galaxy.vel_dust, galaxy.ids_dust, 1, 1, false);

	real4 cm = galaxy.getCenterOfMass();
	std::cout << "Center of mass = " << cm.x << " " << cm.y << " " << cm.z << std::endl;
	real4 tv = galaxy.getCenterOfMassVelocity();
	std::cout << "Center_of_mass_velocity = " << tv.x << " " << tv.y << " " << tv.z << std::endl;

	galaxy.centering();
	galaxy.steady();

	cm = galaxy.getCenterOfMass();
	std::cout << "Center of mass = " << cm.x << " " << cm.y << " " << cm.z << std::endl;
	tv = galaxy.getCenterOfMassVelocity();
	std::cout << "Center_of_mass_velocity = " << tv.x << " " << tv.y << " " << tv.z << std::endl;

	galaxies.push_back(galaxy);
  }
}

void WOGManager::execute()
{
  // Remove particles
  remove_particles();

  // Check for a client
  sockaddr_in clientAddr;
  socklen_t sin_size = sizeof(struct sockaddr_in);
  int new_client_socket = accept(server_socket, (struct sockaddr*)&clientAddr, &sin_size);
  if (new_client_socket > 0) {
	  close(client_socket);
	  client_socket = new_client_socket;
  }

  // Return if no client is connected
  if (client_socket == -1) return;

  // Set client_socket to non-blocking
  long save_fd = fcntl(client_socket, F_GETFL);
  save_fd |= O_NONBLOCK;
  fcntl(client_socket, F_SETFL, save_fd);

  // Check for user request
  char buffer[buffer_size];
  int n = recv(client_socket, buffer, buffer_size, 0);
  if (n <= 0) return;
  buffer[n] = '\0';

  json json_response;

  try {
	json_response = execute_json(buffer);
  } catch (std::exception const& e) {
	std::cerr << "Error: " << e.what() << std::endl;
	json_response["response"] = std::string("Error: ") + e.what();

  } catch ( ... ) {
	std::cerr << "Error: Unknown failure" << std::endl;
    json_response["response"] = "Error: Unknown failure";
  }

  std::ostringstream oss;
  oss << json_response;
  std::string json_response_string = oss.str();

  if (send(client_socket, json_response_string.c_str(), json_response_string.size(), 0) == -1) perror("send");
}

void WOGManager::reshape(int width, int height)
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

void WOGManager::remove_particles()
{
  tree->removeParticles(deletion_radius_square, user_particles, number_of_users);
}

json WOGManager::execute_json(std::string const& json_request_string)
{
  json json_response;

  #ifdef DEBUG_PRINT
    std::cout << "The JSON request is: " << json_request_string << std::endl;
  #endif

  std::istringstream iss(json_request_string);
  json json_request;
  iss >> json_request;

  std::string task = json_request["task"].as<std::string>();

  #ifdef DEBUG_PRINT
    std::cout << "task: " << task << std::endl;
  #endif

  if (task == "release")
  {
    int user_id = json_request["user_id"].as<int>();
    if (user_id < 0 or user_id >= number_of_users) throw std::runtime_error("Invalid user_id");

    int galaxy_id = json_request["galaxy_id"].as<int>();
    if (galaxy_id < 0 or galaxy_id >= galaxies.size()) throw std::runtime_error("Invalid galaxy_id");

    std::vector<double> vector_position = json_request["position"].as<std::vector<double>>();
    if (vector_position.size() > 3) throw std::runtime_error("Invalid dimension of position vector");

    std::vector<double> vector_velocity = json_request["velocity"].as<std::vector<double>>();
    if (vector_velocity.size() > 3) throw std::runtime_error("Invalid dimension of velocity vector");

    real4 position = make_real4(0.0, 0.0, 0.0, 0.0);

    if (vector_position.size() > 0) {
      if (vector_position[0] < 0.0 or vector_position[0] > 1.0) throw std::runtime_error("position.x out of range");
      position.x = vector_position[0] * simulation_plane_width;
    }
    if (vector_position.size() > 1) {
      if (vector_position[1] < 0.0 or vector_position[1] > 1.0) throw std::runtime_error("position.y out of range");
      position.y = vector_position[1] * simulation_plane_height;
    }
    if (vector_position.size() > 2) {
      if (vector_position[2] < -1.0 or vector_position[2] > 1.0) throw std::runtime_error("position.z out of range");
      if (vector_position[2] < 0.0)
        position.z = vector_position[2] * camera_distance;
      else
        position.z = vector_position[2] * (farZ - camera_distance);
    }

    // Shift center to lower left corner
    position.x -= simulation_plane_width * 0.5;
    position.y -= simulation_plane_height * 0.5;

    real4 velocity = make_real4(0.0, 0.0, 0.0, 0.0);

    if (vector_velocity.size() > 0) {
      velocity.x = vector_velocity[0] * window_width / simulation_plane_width;
    }
    if (vector_velocity.size() > 1) {
      velocity.y = vector_velocity[1] * window_height / simulation_plane_height;
    }
    if (vector_velocity.size() > 2) {
      velocity.z = vector_velocity[2] * window_height / simulation_plane_height;
    }

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
      std::cout << "galaxy_id: " << galaxy_id << std::endl;
      std::cout << "position: " << position.x << " " << position.y << " " << position.z << std::endl;
      std::cout << "velocity: " << velocity.x << " " << velocity.y << " " << velocity.z << std::endl;
    #endif

	Galaxy galaxy = galaxies[galaxy_id];
	galaxy.translate(position);
    galaxy.accelerate(velocity);

    // Since the particle ids are not needed for the simulation, we use them to store the user_id in the first digit.
	for (auto & id : galaxy.ids) id = id - id % 10 + user_id;

    tree->releaseGalaxy(galaxy);
	user_particles[user_id] += galaxy.pos.size();

	std::cout << "Galaxy with " + std::to_string(galaxy.pos.size()) + " particles of user " + std::to_string(user_id) + " was released.";
	json_response["response"] = task;
  }
  else if (task == "remove")
  {
    int user_id = json_request["user_id"].as<int>();
    if (user_id < 0 or user_id >= number_of_users) throw std::runtime_error("Invalid user_id");

    #ifdef DEBUG_PRINT
      std::cout << "user_id: " << user_id << std::endl;
    #endif

    tree->removeGalaxy(user_id);
    user_particles[user_id] = 0;

	std::cout << "All particles of user " + std::to_string(user_id) + " were removed.";
	json_response["response"] = task;
  }
  else if (task == "report")
  {
	std::cout << "Reporting current status.";
	json_response["response"] = task;
  }
  else
  {
    throw std::runtime_error("Unknown task: " + task);
  }

  // Always return the status information
  // Simulation time in MYears, for factor see renderloop.cpp, search for MYears
  json_response["simulation_time"] = tree->getTime() * 9.78;
  json up = json::make_array<1>(number_of_users,0);
  for (int ind = 0; ind < number_of_users; ind++) 
    up[ind] = user_particles[ind];
  json_response["user_particles"] = up;
  std::cout << "user_particles: " << up;

  return json_response;
}
