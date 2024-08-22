# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# See file LICENSE for terms.

import os

import boto3
import cupy

import kvikio
from kvikio.benchmarks.aws_s3_io import get_local_port, local_s3_server


def main():
    a = cupy.arange(100)
    b = cupy.empty_like(a)

    # In this example, we launch and use a local S3 server with the
    # following available address:
    endpoint_url = f"http://127.0.0.1:{get_local_port()}"

    # In order use a local server instead of an official Amazon S3 server,
    # we set the AWS_ENDPOINT_URL environment variable.
    os.environ["AWS_ENDPOINT_URL"] = endpoint_url

    # Start a local S3 server
    with local_s3_server(lifetime=100):
        # Create the bucket "my-bucket" and the object "data"
        client = boto3.client("s3", endpoint_url=endpoint_url)
        client.create_bucket(Bucket="my-bucket", ACL="public-read-write")
        client.put_object(Bucket="my-bucket", Key="data", Body=bytes(a))

        # Create a S3 context that connects to AWS_ENDPOINT_URL
        context = kvikio.S3Context()

        # Using the context, we can open "data" as if it was a regular CuFile
        with kvikio.RemoteFile(context, "my-bucket", "data") as f:
            f.read(b)
        print(a)
        print(b)


if __name__ == "__main__":
    main()
