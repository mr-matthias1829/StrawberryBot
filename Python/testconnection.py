import socket
import json

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

packet = {
    "lx": 0,
    "ly": 0,
    "rx": 0,
    "ry": 0,
    "grip": 0
}

sock.sendto(
    json.dumps(packet).encode(),
    ("141.252.29.51", 5005)
)

print("UDP-pakket verzonden")