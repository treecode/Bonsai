/*
 * server.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include <arpa/inet.h>
#include <fcntl.h> /* Added for the nonblocking socket */
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

using namespace std;

#define SERVER_PORT htons(50007)
#define BACKLOG 10
#define BUFFERSIZE 1024

int main() {

    char buffer[BUFFERSIZE];
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
    	perror("socket");
        exit(1);
    }

    int last_fd = sockfd;

    sockaddr_in serverAddr;
    serverAddr.sin_family = AF_INET;
    serverAddr.sin_port = SERVER_PORT;
    serverAddr.sin_addr.s_addr = INADDR_ANY;

    /* bind (this socket, local address, address length)
       bind server socket (sockfd) to server address (serverAddr).
       Necessary so that server can use a specific port */
    if (bind(sockfd, (struct sockaddr*)&serverAddr, sizeof(struct sockaddr)) == -1) {
        perror("bind");
        exit(1);
    }

    // wait for a client
    /* listen (this socket, request queue length) */
    if (listen(sockfd, BACKLOG) == -1) {
        perror("listen");
        exit(1);
    }

    sockaddr_in clientAddr;
    socklen_t sin_size = sizeof(struct sockaddr_in);
    int new_fd = accept(sockfd, (struct sockaddr*)&clientAddr, &sin_size);
  	if (new_fd == -1) perror("accept");

    fcntl(new_fd, F_SETFL, O_NONBLOCK);
    fcntl(last_fd, F_SETFL, O_NONBLOCK);

    for(;;) {
		for (int i = sockfd; i <= last_fd; ++i) {
			std::cout << "Round number " << i << std::endl;
       		if (i != sockfd) {
		 		sin_size = sizeof(struct sockaddr_in);
        		if ((new_fd = accept(sockfd, (struct sockaddr *)&clientAddr, &sin_size)) == -1)
        			perror("accept");
         		printf("server: got connection from %s\n", inet_ntoa(clientAddr.sin_addr));
    	    	fcntl(new_fd, F_SETFL, O_NONBLOCK);
				last_fd = new_fd;
			}
			else {
	    		std::cout << "Wait for message ... " << std::endl;
	    		int n = recv(new_fd, buffer, sizeof(buffer), 0);
				if (n < 1) {
					perror("recv - non blocking \n");
	    			printf("Round %d, and the data read size is: n=%d \n",i,n);
				}
				else {
			        buffer[n] = '\0';
	    			printf("The string is: %s \n", buffer);
            		if (send(new_fd, "Hello, world!\n", 14, 0) == -1)
                        perror("send");
				}
			}
		}
		sleep(1);
    }
}
