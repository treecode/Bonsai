/*
 * server.cpp
 *
 *  Created on: May 6, 2016
 *      Author: Bernd Doser <bernd.doser@h-its.org>
 */

#include <sys/types.h>
#include <sys/socket.h>

int main()
{
    int socket = socket(AF_UNIX, SOCK_STREAM, 0);
}
