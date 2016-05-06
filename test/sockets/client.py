import socket
import sys

HOST, PORT = "localhost", 50007

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((HOST, PORT))

while 1:
    line = raw_input("Which galaxy? ")
    print line
    #s.send(line)
