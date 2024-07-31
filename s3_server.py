# Copyright (c) 2024, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import socket
import time

import boto3
import moto.server
import numpy as np

ThreadedMotoServer = moto.server.ThreadedMotoServer


def get_endpoint_ip():
    return "127.0.0.1"


def get_endpoint_port():
    # Return a free port
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main():
    # Fake aws credentials exported to prevent botocore looking for
    # system aws credentials, https://github.com/spulec/moto/issues/1793
    os.environ["AWS_ACCESS_KEY_ID"] = "foobar_key"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "foobar_secret"
    os.environ["S3FS_LOGGING_LEVEL"] = "DEBUG"
    os.environ["AWS_SECURITY_TOKEN"] = "foobar_security_token"
    os.environ["AWS_SESSION_TOKEN"] = "foobar_session_token"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

    # Launching moto in server mode, i.e., as a separate process
    # with an S3 endpoint on localhost
    endpoint_ip = get_endpoint_ip()
    endpoint_port = get_endpoint_port()
    endpoint_port = "42000"

    endpoint_uri = f"http://{endpoint_ip}:{endpoint_port}/"
    server = ThreadedMotoServer(ip_address=endpoint_ip, port=endpoint_port)
    server.start()
    print(f"starting S3 server: {endpoint_uri}")
    try:
        client = boto3.client("s3", endpoint_url=endpoint_uri)
        client.create_bucket(Bucket="test-bucket", ACL="public-read-write")

        client.put_object(Bucket="test-bucket", Key="a1", Body=bytes(np.arange(2 ** 20, dtype=np.int32)))
        print(f"put_object: s3://{endpoint_ip}:{endpoint_port}/test-bucket/a1")
        time.sleep(100000)
    finally:
        client.delete_object(Bucket="test-bucket", Key="a1")
        server.stop()


if __name__ == "__main__":
    main()
