import socket
import sys

HOST, PORT = "localhost", 50007
BUFFERSIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while 1:
    print "Please define the next galaxy to release:"

    user_id = -1
    while not(user_id >= 1 and user_id <= 4):
        user_id = int(raw_input("Which user? [1-4]: "))
        print "user id = ", user_id

    galaxy_type = -1
    while not(galaxy_type >= 1 and galaxy_type <= 4):
        galaxy_type = int(raw_input("Which galaxy type? [1-4]: "))
        print "galaxy type = ", galaxy_type

    angle = -1.0
    while not(angle >= 0.0 and angle <= 90.0):
        angle = float(raw_input("Which angle? [0 <= angle <= 90]: "))
        print "angle = ", angle

    velocity = -1.0
    while not(velocity >= 0.0 and velocity <= 100.0):
        velocity = float(raw_input("Which velocity? [0 <= velocity <= 100]: "))
        print "velocity = ", velocity
    
    send_data = str(user_id) + "|" + str(galaxy_type) + "|" + str(angle) + "|" + str(velocity)
    print "Galaxy (", send_data, ") was released."
    s.send(send_data)
    
    recv_data = s.recv(BUFFERSIZE)
    print "Received data:", recv_data

s.shutdown(socket.SHUT_RDWR)
s.close()
