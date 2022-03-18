import multiprocessing as mp
import subprocess
from typing import Iterable

import pytest

mp = mp.get_context("spawn")  # type: ignore


def command_server(conn):
    """Server to run commands given through `conn`"""
    while True:
        # Get the next command to run
        cmd, cwd, verbose = conn.recv()
        # Run command
        res: subprocess.CompletedProcess = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=cwd
        )  # type: ignore
        if verbose:
            print(f"{cwd}$ " + " ".join(res.args))
            print(res.stdout.decode(), end="")
        # Send return code back to client
        conn.send(res.returncode)


@pytest.fixture(scope="session", autouse=True)
def run_cmd():
    """Provide a `run_cmd` function to run commands in a separate process

    Use `run_cmd(cmd, cwd, verbose)` to run a command.

    Notice, the server that runs the commands are spawned before CUDA initialization.
    """

    # Start the command server before the very first test
    client_conn, server_conn = mp.Pipe()
    p = mp.Process(target=command_server, args=(server_conn,),)
    p.start()

    def run_cmd(cmd: Iterable[str], cwd, verbose=True):
        client_conn.send((cmd, cwd, verbose))
        return client_conn.recv()

    yield run_cmd

    # Kill the command server after the last test
    p.kill()
