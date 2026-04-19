import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import tempfile
import shutil
import time
from io import StringIO
import multiprocessing
import mpdb.__main__ as mpdb_main
from mpdb.__main__ import set_trace, _init_pdb, _eval_in_context, _exec_in_context
from mpdb.get_dist_info import GetDistInfo, FileSync
from demo.demo_mpdb import _get_free_local_tcp_init_method


def _filesync_barrier_worker(rank, sync_dir, world_size):
    os.environ['DPDB_RANK'] = str(rank)
    os.environ['DPDB_WORLD_SIZE'] = str(world_size)
    fs = FileSync(sync_dir)
    fs.barrier()


def _filesync_broadcast_worker(rank, sync_dir, world_size, commands, received_commands, master_rank=0):
    os.environ['DPDB_RANK'] = str(rank)
    os.environ['DPDB_WORLD_SIZE'] = str(world_size)
    fs = FileSync(sync_dir)
    fs.barrier()

    if rank == master_rank:
        for cmd in commands:
            fs.broadcast_object_list([cmd], src=master_rank)
            time.sleep(0.1)
    else:
        for _ in commands:
            cmd_list = [None]
            fs.broadcast_object_list(cmd_list, src=master_rank)
            received_commands.put(cmd_list[0])

class TestMpdb(unittest.TestCase):

    def setUp(self):
        # Clean up any backend left from previous tests
        GetDistInfo.initialize(sync_method='torch') # Default to torch
        GetDistInfo._FLAG_NOT_DISTRIUBTED_ = False
        self._orig_web_state = dict(mpdb_main._WEB_STATE)
        mpdb_main._WEB_STATE["sticky_web_mode"] = False
        mpdb_main._WEB_STATE["last_web_port"] = None

    def tearDown(self):
        mpdb_main._WEB_STATE.clear()
        mpdb_main._WEB_STATE.update(self._orig_web_state)

    @patch('mpdb.__main__._init_pdb')
    def test_set_trace_non_distributed(self, mock_init_pdb):
        """
        Tests that set_trace() works in a non-distributed environment.
        """
        # Arrange
        mock_pdb_instance = MagicMock()
        mock_init_pdb.return_value = mock_pdb_instance

        # Act
        def func_to_debug():
            a = 1
            b = 2
            set_trace(not_distributed=True)
            c = a + b
            return c

        result = func_to_debug()

        # Assert
        self.assertEqual(result, 3)
        mock_init_pdb.assert_called_once()
        mock_pdb_instance.set_trace.assert_called_once()

    def test_init_pdb_reuses_sticky_web_mode(self):
        created = {}

        class DummyDebugger:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)
                self.rcLines = []

        mpdb_main._WEB_STATE["sticky_web_mode"] = True

        with patch('mpdb.__main__._get_debugger_cls', return_value=DummyDebugger):
            with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=True):
                _init_pdb(sync_method='torch')

        self.assertTrue(created["web_mode"])

    def test_init_pdb_falls_back_to_web_mode_without_tty(self):
        created = {}

        class DummyDebugger:
            def __init__(self, *args, **kwargs):
                created.update(kwargs)
                self.rcLines = []

        with patch('mpdb.__main__._get_debugger_cls', return_value=DummyDebugger):
            with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=False):
                with patch('sys.stderr', new_callable=StringIO) as stderr:
                    _init_pdb(sync_method='torch', web_mode=False)

        self.assertTrue(created["web_mode"])
        self.assertIn("falling back to web mode", stderr.getvalue())

    @patch('mpdb.get_dist_info.TorchSync')
    def test_interaction_master_torch(self, MockTorchSync):
        """
        Tests the interaction method for the master process (rank 0) with torch sync.
        """
        # Arrange
        mock_sync_instance = MockTorchSync.return_value
        mock_sync_instance.get_rank.return_value = 0
        mock_sync_instance.is_distributed.return_value = True

        with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=True):
            with patch('sys.stdout', new_callable=StringIO):
                pdb_instance = _init_pdb(sync_method='torch')

        pdb_instance.cmdloop = MagicMock()
        pdb_instance.forget = MagicMock()
        frame = sys._getframe()
        pdb_instance.botframe = frame

        # Act
        pdb_instance.interaction(frame, None)

        # Assert
        pdb_instance.cmdloop.assert_called_once()
        mock_sync_instance.broadcast_object_list.assert_called_with([None], src=0)
        self.assertEqual(pdb_instance.forget.call_count, 2)

    @patch('mpdb.get_dist_info.TorchSync')
    def test_interaction_worker_torch(self, MockTorchSync):
        """
        Tests the interaction method for a worker process (rank > 0) with torch sync.
        """
        # Arrange
        mock_sync_instance = MockTorchSync.return_value
        mock_sync_instance.get_rank.return_value = 1
        mock_sync_instance.is_distributed.return_value = True

        commands_from_master = ['a = 1', 'q']
        def broadcast_side_effect(obj_list, src):
            if commands_from_master:
                obj_list[0] = commands_from_master.pop(0)
            else:
                obj_list[0] = None
        mock_sync_instance.broadcast_object_list.side_effect = broadcast_side_effect

        with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=True):
            with patch('sys.stdout', new_callable=StringIO):
                pdb_instance = _init_pdb(sync_method='torch')

        pdb_instance.forget = MagicMock()
        frame = sys._getframe()
        pdb_instance.botframe = frame

        # Act
        pdb_instance.interaction(frame, None)

        # Assert
        self.assertEqual(mock_sync_instance.broadcast_object_list.call_count, 2)
        self.assertEqual(mock_sync_instance.barrier.call_count, 2)
        self.assertEqual(pdb_instance.forget.call_count, 2)


class TestMpdbComprehensionScope(unittest.TestCase):
    def _make_local_frame(self):
        def frame_holder():
            some = [1, 2, 3]
            ddd = 10
            return sys._getframe()
        return frame_holder()

    def _make_pdb_instance(self):
        with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=True):
            with patch('sys.stdout', new_callable=StringIO):
                return _init_pdb(sync_method='torch')

    def test_eval_in_context_supports_function_local_list_comprehension(self):
        frame = self._make_local_frame()
        value = _eval_in_context('[x + ddd for x in some]', frame.f_globals, frame.f_locals)
        self.assertEqual(value, [11, 12, 13])

    def test_exec_in_context_supports_function_local_list_comprehension(self):
        frame = self._make_local_frame()
        output = StringIO()
        with patch('sys.stdout', output):
            _exec_in_context('print([x + ddd for x in some])', frame.f_globals, frame.f_locals)
        self.assertIn('[11, 12, 13]', output.getvalue())

    def test_default_supports_function_local_list_comprehension(self):
        frame = self._make_local_frame()
        pdb_instance = self._make_pdb_instance()
        pdb_instance.curframe = frame
        pdb_instance.curframe_locals = frame.f_locals
        pdb_instance.stdout = StringIO()
        pdb_instance.stdin = StringIO()

        pdb_instance.default('print([x + ddd for x in some])')

        self.assertIn('[11, 12, 13]', pdb_instance.stdout.getvalue())

    def test_watch_values_support_function_local_list_comprehension(self):
        frame = self._make_local_frame()
        pdb_instance = self._make_pdb_instance()
        pdb_instance.curframe = frame
        pdb_instance.curframe_locals = frame.f_locals
        pdb_instance.watch_exprs = ['[x + ddd for x in some]']

        self.assertEqual(pdb_instance._get_watch_values(), ['[11, 12, 13]'])


class TestMpdbCommandAssignmentAmbiguity(unittest.TestCase):
    def _make_local_frame(self):
        def frame_holder():
            b = "orig_b"
            d = "orig_d"
            ddd = "orig_ddd"
            return sys._getframe()
        return frame_holder()

    def _make_pdb_instance(self):
        with patch('mpdb.__main__._reopen_stdin_to_tty', return_value=True):
            with patch('sys.stdout', new_callable=StringIO):
                return _init_pdb(sync_method='torch')

    def test_single_letter_assignments_use_python_execution(self):
        frame = self._make_local_frame()
        pdb_instance = self._make_pdb_instance()
        pdb_instance.curframe = frame
        pdb_instance.curframe_locals = frame.f_locals
        pdb_instance.stdout = StringIO()
        pdb_instance.stdin = StringIO()
        pdb_instance.do_b = MagicMock(return_value=False)
        pdb_instance.do_d = MagicMock(return_value=False)

        pdb_instance.onecmd('b=123')
        pdb_instance.onecmd('d=456')
        pdb_instance.onecmd('ddd=789')

        self.assertEqual(pdb_instance.curframe_locals['b'], 123)
        self.assertEqual(pdb_instance.curframe_locals['d'], 456)
        self.assertEqual(pdb_instance.curframe_locals['ddd'], 789)
        self.assertEqual(pdb_instance.do_b.call_count, 0)
        self.assertEqual(pdb_instance.do_d.call_count, 0)
        self.assertEqual(pdb_instance.stdout.getvalue(), '')

    def test_spaced_single_letter_assignments_use_python_execution(self):
        frame = self._make_local_frame()
        pdb_instance = self._make_pdb_instance()
        pdb_instance.curframe = frame
        pdb_instance.curframe_locals = frame.f_locals
        pdb_instance.stdout = StringIO()
        pdb_instance.stdin = StringIO()
        pdb_instance.do_b = MagicMock(return_value=False)
        pdb_instance.do_d = MagicMock(return_value=False)

        pdb_instance.onecmd('b = 123')
        pdb_instance.onecmd('d = 456')

        self.assertEqual(pdb_instance.curframe_locals['b'], 123)
        self.assertEqual(pdb_instance.curframe_locals['d'], 456)
        self.assertEqual(pdb_instance.do_b.call_count, 0)
        self.assertEqual(pdb_instance.do_d.call_count, 0)
        self.assertEqual(pdb_instance.stdout.getvalue(), '')

    def test_real_break_and_down_commands_still_dispatch(self):
        frame = self._make_local_frame()
        pdb_instance = self._make_pdb_instance()
        pdb_instance.curframe = frame
        pdb_instance.curframe_locals = frame.f_locals
        pdb_instance.stdout = StringIO()
        pdb_instance.stdin = StringIO()
        pdb_instance.do_b = MagicMock(return_value=False)
        pdb_instance.do_d = MagicMock(return_value=False)

        pdb_instance.onecmd('b 123')
        pdb_instance.onecmd('d 2')

        pdb_instance.do_b.assert_called_once_with('123')
        pdb_instance.do_d.assert_called_once_with('2')

class TestFileSync(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_file_sync_barrier(self):
        """Tests the barrier implementation of FileSync."""
        world_size = 3
        processes = []

        for i in range(world_size):
            p = multiprocessing.Process(target=_filesync_barrier_worker, args=(i, self.temp_dir, world_size))
            processes.append(p)
            p.start()

        for p in processes:
            p.join(timeout=5)
            self.assertFalse(p.is_alive(), f"Process {p.pid} timed out.")

    def test_file_sync_broadcast(self):
        """Tests the broadcast implementation of FileSync."""
        master_rank = 0
        commands = ["a=1", "b=2", None]
        received_commands = multiprocessing.Queue()
        p_master = multiprocessing.Process(target=_filesync_broadcast_worker, args=(master_rank, self.temp_dir, 2, commands, received_commands, master_rank))
        p_worker = multiprocessing.Process(target=_filesync_broadcast_worker, args=(1, self.temp_dir, 2, commands, received_commands, master_rank))

        p_master.start()
        p_worker.start()

        p_master.join(timeout=5)
        p_worker.join(timeout=5)

        self.assertFalse(p_master.is_alive())
        self.assertFalse(p_worker.is_alive())

        results = []
        while not received_commands.empty():
            results.append(received_commands.get())
        
        self.assertEqual(results, ['a=1', 'b=2', None])

    def test_file_sync_broadcast_with_none_string(self):
        """Tests that broadcasting a string 'None' is not mistaken for the None object."""
        master_rank = 0
        commands = ["print(1)", "None", "print(2)", None]
        received_commands = multiprocessing.Queue()
        p_master = multiprocessing.Process(target=_filesync_broadcast_worker, args=(master_rank, self.temp_dir, 2, commands, received_commands, master_rank))
        p_worker = multiprocessing.Process(target=_filesync_broadcast_worker, args=(1, self.temp_dir, 2, commands, received_commands, master_rank))

        p_master.start()
        p_worker.start()

        p_master.join(timeout=5)
        p_worker.join(timeout=5)

        self.assertFalse(p_master.is_alive())
        self.assertFalse(p_worker.is_alive())

        results = []
        while not received_commands.empty():
            results.append(received_commands.get())
        
        self.assertEqual(results, ["print(1)", "None", "print(2)", None])

    @patch.dict('os.environ', {'DPDB_RANK': '100', 'DPDB_WORLD_SIZE': '200'})
    def test_file_sync_init_with_args(self):
        """Tests that FileSync prioritizes arguments over environment variables."""
        # Initialize with direct arguments
        fs = FileSync(self.temp_dir, rank=5, world_size=10)

        self.assertEqual(fs.get_rank(), 5)
        self.assertEqual(fs.get_world_size(), 10)


class TestDemoTorchInitMethod(unittest.TestCase):
    def test_get_free_local_tcp_init_method_uses_loopback_tcp_url(self):
        init_method = _get_free_local_tcp_init_method()

        self.assertRegex(init_method, r"^tcp://127\.0\.0\.1:\d+$")
        port = int(init_method.rsplit(":", 1)[1])
        self.assertGreater(port, 0)
        self.assertLess(port, 65536)

if __name__ == "__main__":
    unittest.main()