import json
import socket
import sys

HOST = 'localhost'
PORT = 50007
BUFFERSIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while 1:

    send_json = raw_input("JSON string: ")
    if send_json == "exit":
       break
    s.send(send_json)

    recv_json = s.recv(BUFFERSIZE)
    print "Received data:", recv_json

s.shutdown(socket.SHUT_RDWR)
s.close()

