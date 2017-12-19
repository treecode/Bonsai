#!/usr/bin/python

import json
import socket
import sys

HOST = 'localhost'
PORT = 50008
BUFFERSIZE = 1024

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))

while 1:

    send_data = {}

    while not('task' in send_data and (send_data['task'] == 'release' or send_data['task'] == 'remove' or send_data['task'] == 'report')):
        send_data['task'] = raw_input("Which task? ['release', 'remove', 'report']: ")

    if send_data['task'] == 'release' or send_data['task'] == 'remove':

        while not('user_id' in send_data and send_data['user_id'] >= 1 and send_data['user_id'] <= 4):
            send_data['user_id'] = int(raw_input("Which user? [1-4]: "))
    
        if send_data['task'] == 'release':
    
            while not('galaxy_id' in send_data and send_data['galaxy_id'] >= 1 and send_data['galaxy_id'] <= 4):
                send_data['galaxy_id'] = int(raw_input("Which galaxy? [1-4]: "))
          
            while not('angle' in send_data and send_data['angle'] >= 0.0 and send_data['angle'] <= 90.0):
                send_data['angle'] = float(raw_input("Which angle? [0 <= angle <= 90]: "))
          
            while not('velocity' in send_data and send_data['velocity'] >= 0.0 and send_data['velocity'] <= 100.0):
                send_data['velocity'] = float(raw_input("Which velocity? [0 <= velocity <= 100]: "))
        
    send_data_json = json.dumps(send_data)
    print "Send data:", send_data_json
    s.send(send_data_json)

    recv_data_json = s.recv(BUFFERSIZE)
    print "Received data:", recv_data_json

s.shutdown(socket.SHUT_RDWR)
s.close()

