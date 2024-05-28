from xmlrpc.client import ServerProxy

server_url = "http://localhost:8000"
server = ServerProxy(server_url)

# Call a method on the server
result = server.some_method("Francisco", 21)
print("Result:", result)

# Call the open_Camera method
camera_result = server.open_Camera()
print("Camera Result:", camera_result)
