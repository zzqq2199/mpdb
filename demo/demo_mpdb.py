import argparse
import multiprocessing
import os
from pathlib import Path
import socket
import shutil
import sys
import tempfile
import time

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import mpdb


def single_process():
    value = "single-process"
    for step in range(10):
        if step == 5:
            mpdb.set_trace(not_distributed=True)
    print(f"[After mpdb] value={value}")


def worker_for_file_sync(rank, world_size, sync_dir):
    print(f"[FileSync Demo] Rank {rank}/{world_size} started. PID: {os.getpid()}")
    for step in range(10):
        print(f"Rank {rank}: step = {step}")
        time.sleep(0.1 * rank)
        if step == 5:
            print(f"Rank {rank} is hitting the debugger.")
            mpdb.set_trace(
                sync_method="file",
                sync_dir=sync_dir,
                rank=rank,
                world_size=world_size,
            )
    print(f"[FileSync Demo] Rank {rank} finished.")


def multi_process_by_file_sync():
    world_size = 3
    sync_dir = tempfile.mkdtemp(prefix="mpdb-demo-")
    print("--- Starting FileSync Demo ---")
    print(f"Shared sync directory: {sync_dir}")
    print(f"World Size: {world_size}")

    processes = []
    for rank in range(1, world_size):
        process = multiprocessing.Process(
            target=worker_for_file_sync,
            args=(rank, world_size, sync_dir),
        )
        processes.append(process)
        process.start()

    worker_for_file_sync(rank=0, world_size=world_size, sync_dir=sync_dir)

    for process in processes:
        process.join()

    print(f"Cleaning up sync directory: {sync_dir}")
    shutil.rmtree(sync_dir)
    print("--- FileSync Demo Finished ---")


def worker_for_torch(rank, world_size, init_method):
    print(f"[TorchSync Demo] Rank {rank}/{world_size} started. PID: {os.getpid()}")

    import torch

    torch.distributed.init_process_group(
        backend="gloo",
        init_method=init_method,
        world_size=world_size,
        rank=rank,
    )
    torch.distributed.barrier()

    for step in range(10):
        print(f"Rank {rank}: step = {step}")
        time.sleep(0.1 * rank)
        if step == 5:
            if rank == 0:
                try:
                    sys.stdin = open("/dev/tty")
                except OSError:
                    pass
            torch.distributed.barrier()
            print(f"Rank {rank} is hitting the debugger.")
            mpdb.set_trace(
                sync_method="torch",
                rank=rank,
                world_size=world_size,
                web_mode=True,
            )
            print(f"Rank {rank} has exited the debugger.")

    mpdb.set_trace(rank=rank, world_size=world_size)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()
    print(f"[TorchSync Demo] Rank {rank} finished.")


def _get_free_local_tcp_init_method():
    # Pick an ephemeral loopback port so the demo does not fail if a fixed
    # rendezvous port is already occupied on the local machine.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()
    return f"tcp://127.0.0.1:{port}"


def multi_process_by_torch():
    world_size = 2
    init_method = _get_free_local_tcp_init_method()
    print(f"[TorchSync Demo] Using init_method={init_method}")
    processes = []

    for rank in range(world_size):
        process = multiprocessing.Process(
            target=worker_for_torch,
            args=(rank, world_size, init_method),
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def main():
    parser = argparse.ArgumentParser(description="Run mpdb demo scenarios.")
    parser.add_argument(
        "--mode",
        type=str,
        default="file",
        choices=["single", "file", "torch"],
        help="Choose a demo: single / file / torch",
    )
    args = parser.parse_args()

    if args.mode == "single":
        print("Running in single process mode.")
        single_process()
    elif args.mode == "file":
        print("Running in multi-process mode with file sync.")
        multi_process_by_file_sync()
    elif args.mode == "torch":
        print("Running in multi-process mode with torch sync.")
        multi_process_by_torch()


if __name__ == "__main__":
    main()
