import socket
import sys

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('', 10000)
print (sys.stderr, 'starting up on %s port %s' % server_address)
sock.bind(server_address)

# Listen for incoming connections
sock.listen(1)

while True: 
    connection, client_address = sock.accept()
    while True:
        data = connection.recv(2048) #recv(buffer_size) --> 1024, 2048, 4096,default  
        data = data.decode("utf-8")
        print(data)