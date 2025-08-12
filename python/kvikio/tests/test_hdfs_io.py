# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.


import pytest
# from pytest_httpserver import HTTPServer
from kvikio import remote_file
import json
import pytest_httpserver

# @pytest.fixture
# def remote_file_path() -> str:
#     return "/home/test_user/test_file.bin"


# @pytest.fixture
# def register_request_for_file_size(httpserver: HTTPServer, remote_file_path: str) -> HTTPServer:
#     """
#         Minimal WebHDFS OPEN mock:
#         GET /webhdfs/v1/test.txt?op=OPEN&user.name=testuser  -> 307 with Location
#         GET <Location> (same server, with &datanode=true)    -> 200 with body
#         Optionally allow a HEAD on the first URL.
#         """
#     # Optional HEAD some clients issue
#     # httpserver.expect_request(
#     #     "/webhdfs/v1/home/user/test.bin",
#     #     method="HEAD",
#     #     query_string={"op": "OPEN", "user.name": "testuser"},
#     # ).respond_with_data(b"", status=200)

#     # # Coordinator GET: send 307 redirect with Location
#     # redirect_target_path = "/webhdfs/v1/home/user/test.bin"
#     # redirect_qs = "op=OPEN&user.name=testuser&datanode=true"
#     # redirect_url = httpserver.url_for(redirect_target_path) + "?" + redirect_qs

#     # httpserver.expect_request(
#     #     redirect_target_path,
#     #     method="GET",
#     #     query_string={"op": "OPEN", "user.name": "testuser"},
#     # ).respond_with_data(
#     #     b"", status=307, headers={"Location": redirect_url}
#     # )

#     json_str = json.dumps({"length": 1234})
#     print(json_str)
#     # httpserver.expect_request(
#     #     f"/webhdfs/v1{remote_file_path}",
#     #     method="GET",
#     #     query_string={"op": "GETFILESTATUS"},
#     # ).respond_with_json(response_json=json_str)

#     return httpserver


# def test_webhdfs_get_file_size(register_request_for_file_size: HTTPServer, remote_file_path: str):
#     url = register_request_for_file_size.url_for(
#         f"/webhdfs/v1{remote_file_path}")
#     print("-->", url)

#     # url = f"{base}?op=OPEN&user.name=testuser"

#     # file = remote_file.RemoteFile.open_webhdfs(url)
#     register_request_for_file_size.check_assertions()

#     # assert n == 11
#     # assert bytes(buf) == b"Hello World!"


# def test(httpserver: HTTPServer):
#     response_json = {
#         "FileStatus": {
#             "length": 24930,
#             "type": "FILE"
#         }
#     }

#     # Use expect_oneshot_request for debugging
#     httpserver.expect_oneshot_request(
#         "/webhdfs/v1/path/to/file",
#         query_string="op=GETFILESTATUS"
#     ).respond_with_json(response_json, status=200)

#     url = httpserver.url_for("/webhdfs/v1/path/to/file")
#     print(f"URL being passed: {url}")  # Debug output

#     try:
#         rf = remote_file.RemoteFile.open_webhdfs(url)
#         assert rf.size == 24930
#     finally:
#         # Check what requests were actually made
#         print("Request log:")
#         for request in httpserver.log:
#             print(f"  {request.method} {request.path}?{request.query_string}")

#         # This will show unmatched requests
#         httpserver.check_assertions()

def test_with_simple_server():
    import threading
    import http.server
    import json
    from time import sleep

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            print(f"Received request: {self.path}")
            response = {
                "FileStatus": {
                    "length": 24930,
                    "type": "FILE"
                }
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        def log_message(self, format, *args):
            print(f"Server log: {format % args}")

    # Start server in thread
    server = http.server.HTTPServer(('127.0.0.1', 8888), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()

    sleep(0.5)  # Give server time to start

    url = "http://127.0.0.1:8888/webhdfs/v1/path/to/file"
    print(f"Testing with URL: {url}")

    try:
        rf = remote_file.RemoteFile.open_webhdfs(url)
        print(f"Success! Size: {rf.size}")
    finally:
        server.shutdown()


test_with_simple_server()
