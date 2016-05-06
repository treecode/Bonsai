import socket
import sys

HOST, PORT = "localhost", 50007

#s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#s.connect((HOST, PORT))

for line in sys.stdin:
    print line
    #s.send(line)
