
import socket

pc_ip = "192.168.1.1"

s = socket.socket()
s.connect((pc_ip, 5000))

while True:
	s.send("Hello from pi".encode())
	print("Pc says:", s.recv(1024).decode())


