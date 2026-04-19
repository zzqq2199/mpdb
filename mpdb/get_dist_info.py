import abc
import os
import fcntl
import time
from functools import lru_cache

class DistSyncBase(abc.ABC):
    @abc.abstractmethod
    def get_rank(self):
        pass

    @abc.abstractmethod
    def get_world_size(self):
        pass

    @abc.abstractmethod
    def barrier(self):
        pass

    @abc.abstractmethod
    def broadcast_object_list(self, obj_list, src=0):
        pass

    @abc.abstractmethod
    def gather_object_list(self, obj_list, dst=0):
        pass

    @abc.abstractmethod
    def is_distributed(self):
        pass

class TorchSync(DistSyncBase):
    @lru_cache()
    def is_distributed(self):
        try:
            import torch
            return torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        except ImportError:
            return False

    def get_rank(self):
        if not self.is_distributed(): return 0
        import torch
        return torch.distributed.get_rank()

    def get_world_size(self):
        if not self.is_distributed(): return 1
        import torch
        return torch.distributed.get_world_size()

    def barrier(self):
        if self.is_distributed():
            import torch
            torch.distributed.barrier()

    def broadcast_object_list(self, *args, **kwargs):
        if self.is_distributed():
            import torch
            torch.distributed.broadcast_object_list(*args, **kwargs)

    def gather_object_list(self, obj_list, dst=0):
        if not self.is_distributed():
            return
        import torch
        gathered = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered, obj_list[0])
        if torch.distributed.get_rank() == dst:
            obj_list[0] = gathered

import pickle

class FileSync(DistSyncBase):
    def __init__(self, sync_dir, rank=None, world_size=None):
        if not sync_dir or not os.path.isdir(sync_dir):
            raise ValueError("sync_dir must be a valid directory for file sync method.")
        self.sync_dir = sync_dir

        # Prioritize arguments over environment variables
        if rank is not None:
            self.rank = int(rank)
        else:
            self.rank = int(os.environ.get('DPDB_RANK', 0))

        if world_size is not None:
            self.world_size = int(world_size)
        else:
            self.world_size = int(os.environ.get('DPDB_WORLD_SIZE', 1))

        self.command_file = os.path.join(self.sync_dir, 'command.pkl') # Use .pkl for pickle files
        self.command_version_file = os.path.join(self.sync_dir, 'command_version.txt')
        self.barrier_version_file = os.path.join(self.sync_dir, 'barrier_version.txt')
        self._last_seen_version = 0 # Start at version 0

        if self.rank == 0:
            self._write_file(self.command_version_file, '0')
            self._write_file(self.barrier_version_file, '0')
            with open(self.command_file, 'wb') as f:
                pickle.dump('', f) # Pre-create command file with empty content

    def _write_file(self, path, content):
        with open(path, 'w') as f:
            f.write(content)

    def _read_file(self, path):
        try:
            with open(path, 'r') as f:
                return f.read().strip()
        except FileNotFoundError:
            return None

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def is_distributed(self):
        return self.world_size > 1

    def barrier(self):
        if not self.is_distributed():
            return

        barrier_version = None
        while barrier_version is None:
            barrier_version = self._read_file(self.barrier_version_file)
            if barrier_version is None:
                time.sleep(0.01)

        done_file = os.path.join(self.sync_dir, f'barrier_v{barrier_version}_rank_{self.rank}.done')
        self._write_file(done_file, '')

        while True:
            done_files = [f for f in os.listdir(self.sync_dir) if f.startswith(f'barrier_v{barrier_version}_rank_') and f.endswith('.done')]
            if len(done_files) == self.world_size:
                break
            time.sleep(0.01)

        if self.rank == 0:
            # Small sleep to ensure all workers have exited the loop before master cleans up
            time.sleep(0.05)
            for f in done_files:
                try:
                    os.remove(os.path.join(self.sync_dir, f))
                except FileNotFoundError:
                    pass # Another process might have cleaned up
            self._write_file(self.barrier_version_file, str(int(barrier_version) + 1))

    def broadcast_object_list(self, obj_list, src=0):
        if not self.is_distributed():
            return

        if self.rank == src: # Master
            command = obj_list[0]
            with open(self.command_file, 'wb') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                pickle.dump(command, f)
                fcntl.flock(f, fcntl.LOCK_UN)
            
            current_version = int(self._read_file(self.command_version_file) or 0)
            self._write_file(self.command_version_file, str(current_version + 1))
        else: # Worker
            while True:
                current_version_str = self._read_file(self.command_version_file)
                if current_version_str is None:
                    time.sleep(0.01)
                    continue
                
                current_version = int(current_version_str)
                if current_version > self._last_seen_version:
                    with open(self.command_file, 'rb') as f:
                        fcntl.flock(f, fcntl.LOCK_SH)
                        command_obj = pickle.load(f)
                        fcntl.flock(f, fcntl.LOCK_UN)
                    
                    self._last_seen_version = current_version
                    obj_list[0] = command_obj
                    break
                time.sleep(0.01)

    def gather_object_list(self, obj_list, dst=0):
        if not self.is_distributed():
            return

        barrier_version = None
        while barrier_version is None:
            barrier_version = self._read_file(self.barrier_version_file)
            if barrier_version is None:
                time.sleep(0.01)

        path = os.path.join(self.sync_dir, f'gather_v{barrier_version}_rank_{self.rank}.pkl')
        with open(path, 'wb') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            pickle.dump(obj_list[0], f)
            fcntl.flock(f, fcntl.LOCK_UN)

        if self.rank == dst:
            while True:
                files = [f for f in os.listdir(self.sync_dir) if f.startswith(f'gather_v{barrier_version}_rank_') and f.endswith('.pkl')]
                if len(files) == self.world_size:
                    break
                time.sleep(0.01)

            gathered = [None for _ in range(self.world_size)]
            for r in range(self.world_size):
                p = os.path.join(self.sync_dir, f'gather_v{barrier_version}_rank_{r}.pkl')
                with open(p, 'rb') as f:
                    fcntl.flock(f, fcntl.LOCK_SH)
                    gathered[r] = pickle.load(f)
                    fcntl.flock(f, fcntl.LOCK_UN)
            obj_list[0] = gathered

            for r in range(self.world_size):
                try:
                    os.remove(os.path.join(self.sync_dir, f'gather_v{barrier_version}_rank_{r}.pkl'))
                except FileNotFoundError:
                    pass
        else:
            while os.path.exists(path):
                time.sleep(0.01)

_sync_backend = None

class GetDistInfo:
    _FLAG_NOT_DISTRIUBTED_ = False
    _THREAD_HEARTBEAT_ = None
    
    @staticmethod
    def keep_alive(keep_alive_interval=300):
        if keep_alive_interval <= 0:
            return
        import threading
        if GetDistInfo._THREAD_HEARTBEAT_ is not None:
            return
        if _sync_backend is None:
            return
        def heartbeat():
            while True:
                time.sleep(keep_alive_interval)
                _sync_backend.barrier()
        GetDistInfo._THREAD_HEARTBEAT_ = threading.Thread(target=heartbeat, daemon=True)
        GetDistInfo._THREAD_HEARTBEAT_.start()

    @staticmethod
    def initialize(sync_method='torch', sync_dir=None, rank=None, world_size=None):
        global _sync_backend
        if GetDistInfo._FLAG_NOT_DISTRIUBTED_:
            _sync_backend = None
            return

        if sync_method == 'torch':
            _sync_backend = TorchSync()
        elif sync_method == 'file':
            _sync_backend = FileSync(sync_dir, rank=rank, world_size=world_size)
        else:
            raise ValueError(f"Unknown sync_method: {sync_method}")

    @staticmethod
    def get_rank():
        if _sync_backend is None: return 0
        return _sync_backend.get_rank()

    @staticmethod
    def get_world_size():
        if _sync_backend is None: return 1
        return _sync_backend.get_world_size()

    @staticmethod
    def is_distributed():
        if GetDistInfo._FLAG_NOT_DISTRIUBTED_: return False
        if _sync_backend is None: return False
        return _sync_backend.is_distributed()

    @staticmethod
    def barrier():
        if _sync_backend is not None:
            _sync_backend.barrier()

    @staticmethod
    def broadcast_object_list(*args, **kwargs):
        if _sync_backend is not None:
            _sync_backend.broadcast_object_list(*args, **kwargs)

    @staticmethod
    def gather_object_list(*args, **kwargs):
        if _sync_backend is not None:
            _sync_backend.gather_object_list(*args, **kwargs)
