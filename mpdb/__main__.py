# Copyright (c) 2011-2016 Godefroid Chapelle and ipdb development team
#
# This file is part of ipdb.
# Redistributable under the revised BSD license
# https://opensource.org/licenses/BSD-3-Clause

from __future__ import print_function
import ast
import os
import sys
import rlcompleter

from .get_dist_info import GetDistInfo

from contextlib import contextmanager

__version__ = '0.13.13'

from IPython import get_ipython
from IPython.core.debugger import BdbQuit_excepthook
from IPython.terminal.ipapp import TerminalIPythonApp
from IPython.terminal.embed import InteractiveShellEmbed
try:
    import configparser
except:
    import ConfigParser as configparser

try:
    from . import web_pdb
except ImportError:
    import web_pdb
    
__prompt__ = "[mpdb]>>>"

_WEB_STATE = {
    "watch_exprs": [],
    "sticky_web_mode": False,
    "last_web_port": None,
}


def _stdin_is_tty(stream=None):
    stream = sys.stdin if stream is None else stream
    try:
        return bool(stream) and bool(stream.isatty())
    except Exception:
        return False


def _reopen_stdin_to_tty():
    if _stdin_is_tty():
        return True
    try:
        tty = open('/dev/tty')
    except Exception:
        return False
    if not _stdin_is_tty(tty):
        try:
            tty.close()
        except Exception:
            pass
        return False
    sys.stdin = tty
    return True


def _resolve_web_mode(web_mode):
    if web_mode is True:
        return True
    if web_mode is False:
        return False
    if os.environ.get("Z_DPDB_WEB_MODE") == "1":
        return True
    return bool(_WEB_STATE.get("sticky_web_mode"))


def _build_exec_namespace(globals_dict, locals_dict):
    namespace = {}
    try:
        namespace.update(globals_dict or {})
    except Exception:
        pass
    try:
        namespace.update(locals_dict or {})
    except Exception:
        pass
    return namespace


def _sync_exec_locals(locals_dict, namespace):
    if locals_dict is None:
        return
    for key, value in namespace.items():
        if key == "__builtins__":
            continue
        try:
            locals_dict[key] = value
        except Exception:
            pass


def _eval_in_context(expression, globals_dict, locals_dict):
    namespace = _build_exec_namespace(globals_dict, locals_dict)
    return eval(expression, namespace, namespace)


def _exec_in_context(source, globals_dict, locals_dict, mode='exec'):
    namespace = _build_exec_namespace(globals_dict, locals_dict)
    code = compile(source, '<stdin>', mode)
    exec(code, namespace, namespace)
    _sync_exec_locals(locals_dict, namespace)
    return namespace


def _is_assignment_statement(source):
    try:
        tree = ast.parse(source, mode='exec')
    except SyntaxError:
        return False
    if len(tree.body) != 1:
        return False
    stmt = tree.body[0]
    return isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign))

"""
Here are some of the other methods at the same level as precmd that you can use to customize the behavior of the debugger:
    - postcmd(stop, line): This method is called after a command has been executed. The stop argument is a flag that indicates whether the interpreter should stop, and line is the command that was just executed.
    - preloop(): This is called once when the command loop is entered.
    - postloop(): This is called once when the command loop is about to be exited.
    - emptyline(): This method is called when the user enters an empty line. By default, it re-runs the last command.
    - default(line): This is called when the command entered by the user doesn't match any of the defined do_* commands.
    - do_<command>(args): You can define your own commands by creating methods with names that start with do_. For example, a method named do_hello(self, arg) would handle the command hello.
"""


def _get_debugger_cls():
    shell = get_ipython()
    if shell is None:
        # Not inside IPython
        # Build a terminal app in order to force ipython to load the
        # configuration
        ipapp = TerminalIPythonApp()
        # Avoid output (banner, prints)
        ipapp.interact = False
        ipapp.initialize(["--no-term-title"])
        shell = ipapp.shell
    else:
        # Running inside IPython

        # Detect if embed shell or not and display a message
        if isinstance(shell, InteractiveShellEmbed):
            sys.stderr.write(
                "\nYou are currently into an embedded ipython shell,\n"
                "the configuration will not be loaded.\n\n"
            )

    # Let IPython decide about which debugger class to use
    # This is especially important for tools that fiddle with stdout
    
    debugger_cls = shell.debugger_cls

    class DistributedPdb(debugger_cls):
        def __init__(self, *args, **kwargs):
            # Pop sync arguments to avoid passing them to the parent class
            kwargs.pop('sync_method', None)
            kwargs.pop('sync_dir', None)
            self.web_mode = kwargs.pop('web_mode', False)
            self.web_port = kwargs.pop('web_port', None)

            super(DistributedPdb, self).__init__(*args, **kwargs)
            self.reset()
            self.prompt = __prompt__
            self.rank = GetDistInfo.get_rank()
            if self.rank == 0:
                is_dist = bool(GetDistInfo.is_distributed())
                if self.web_mode:
                    print("Web mode enabled")
                tip_parts = []
                if is_dist:
                    tip_parts.append("You are in distributed debugging mode.")
                else:
                    tip_parts.append("You are in single-process debugging mode.")
                if not self.web_mode:
                    tip_parts.append("Use `z web` to switch to web mode.")
                if tip_parts:
                    print("Tip: " + " ".join(tip_parts))
            self.all = True
            if self.web_mode:
                self.ui_mode = 'web'
            else:
                self.ui_mode = 'cli'
            self._ui_switch_requested = False
            self.watch_exprs = list(_WEB_STATE.get("watch_exprs") or [])
        
        def do_all(self, arg):
            """toggle all ranks
            """
            try:
                if arg in ['on', '1']:
                    self.all = True
                elif arg in ['off', '0']:
                    self.all = False
            except:
                print(f"invalid args, set all to True")
                self.all = True
            

        def precmd(self, line):
            if self.rank == 0:
                if self.all:
                    bcast_data = [line]
                else:
                    bcast_data = ["None"]
                GetDistInfo.broadcast_object_list(bcast_data, src=0)
            return line
        
        def postcmd(self, stop, line):
            # print(f"postcmd: line={line}, rank={self.rank}")
            GetDistInfo.barrier()
            return stop

        def web_complete(self, line, cursor):
            if line is None:
                line = ''
            if cursor is None:
                cursor = len(line)
            cursor = max(0, min(int(cursor), len(line)))
            prefix = line[:cursor]

            start = cursor
            while start > 0 and not prefix[start - 1].isspace():
                start -= 1
            text = prefix[start:cursor]

            py_mode = prefix.lstrip().startswith('!')
            if start == 0 and text.startswith('!'):
                start = 1
                text = text[1:]

            matches = []

            if not py_mode and prefix.lstrip().startswith('z '):
                z_start = prefix.find('z ')
                if z_start != -1:
                    after = prefix[z_start + 2:]
                    arg_prefix = after.lstrip()
                    if arg_prefix:
                        start = z_start + 2 + (len(after) - len(arg_prefix))
                        text = arg_prefix
                    else:
                        start = cursor
                        text = ''
                matches.extend([m for m in ['web', 'terminal', 'console', 'shell'] if m.startswith(text)])
                uniq = sorted(set(matches))
                return start, cursor, uniq

            if not py_mode and start == 0 and (' ' not in prefix) and ('\t' not in prefix):
                cmds = []
                for name in dir(self):
                    if name.startswith('do_') and len(name) > 3:
                        cmds.append(name[3:])
                cmds.extend(['q', 'quit', 'exit'])
                matches.extend([c for c in cmds if c.startswith(text)])

            frame = getattr(self, 'curframe', None)
            if frame is not None:
                ns = {}
                try:
                    ns.update(frame.f_globals or {})
                except Exception:
                    pass
                try:
                    ns.update(frame.f_locals or {})
                except Exception:
                    pass
                completer = rlcompleter.Completer(ns)
                state = 0
                while True:
                    item = completer.complete(text, state)
                    if item is None:
                        break
                    matches.append(item)
                    state += 1

            uniq = sorted(set(matches))
            return start, cursor, uniq

        def do_z(self, arg):
            target = (arg or '').strip().split(' ', 1)[0]
            if target == 'web':
                if not self.web_mode:
                    self.web_mode = True
                _WEB_STATE["sticky_web_mode"] = True
                if self.ui_mode == 'web':
                    print("Already in web mode")
                    return False
                self.ui_mode = 'web'
                self._ui_switch_requested = True
                print("Switching to web mode")
                return True
            if target in ['terminal', 'console', 'shell', "cli"]:
                _WEB_STATE["sticky_web_mode"] = False
                if self.ui_mode == 'cli':
                    print("Already in terminal mode")
                    return False
                self.ui_mode = 'cli'
                self._ui_switch_requested = True
                print("Switching to terminal mode")
                return True
            print("Usage: z web | z terminal | z console | z shell | z cli")
            return False
        
        def _error_exc(self):
            import traceback
            traceback.print_exc()

        def _get_exec_locals(self, frame=None):
            if frame is None:
                frame = getattr(self, 'curframe', None)
            if frame is getattr(self, 'curframe', None):
                return getattr(self, 'curframe_locals', None)
            return getattr(frame, 'f_locals', None)

        def _eval_in_frame(self, expression, frame=None):
            if frame is None:
                frame = getattr(self, 'curframe', None)
            if frame is None:
                raise RuntimeError("No current frame for evaluation")
            return _eval_in_context(expression, frame.f_globals, self._get_exec_locals(frame))

        def _exec_in_frame(self, source, frame=None, mode='exec'):
            if frame is None:
                frame = getattr(self, 'curframe', None)
            if frame is None:
                raise RuntimeError("No current frame for execution")
            return _exec_in_context(source, frame.f_globals, self._get_exec_locals(frame), mode=mode)

        def default(self, line):
            if line[:1] == '!':
                line = line[1:]
            try:
                save_stdout = sys.stdout
                save_stdin = sys.stdin
                save_displayhook = sys.displayhook
                try:
                    sys.stdin = self.stdin
                    sys.stdout = self.stdout
                    sys.displayhook = self.displayhook
                    self._exec_in_frame(line + '\n', mode='single')
                finally:
                    sys.stdout = save_stdout
                    sys.stdin = save_stdin
                    sys.displayhook = save_displayhook
            except Exception:
                self._error_exc()

        def _getval(self, arg):
            try:
                return self._eval_in_frame(arg)
            except Exception:
                self._error_exc()
                raise

        def _getval_except(self, arg, frame=None):
            import traceback
            try:
                return self._eval_in_frame(arg, frame=frame)
            except Exception:
                exc_info = sys.exc_info()[:2]
                err = traceback.format_exception_only(*exc_info)[-1].strip()
                return '** raised %s **' % err
            

        def inject_command(self, cmd):
            """Inject command into stdin (CLI mode support)"""
            try:
                import fcntl
                import termios
                for char in cmd:
                    fcntl.ioctl(sys.stdin, termios.TIOCSTI, char.encode())
                fcntl.ioctl(sys.stdin, termios.TIOCSTI, b'\n')
            except Exception as e:
                print(f"Failed to inject command: {e}")

        def _get_watch_values(self):
            exprs = list(self.watch_exprs or [])
            if not exprs:
                return []
            frame = getattr(self, 'curframe', None)
            if frame is None:
                return ['<no frame>' for _ in exprs]
            out = []
            for expr in exprs:
                e = (expr or '').strip()
                if not e:
                    out.append('')
                    continue
                try:
                    val = self._eval_in_frame(e, frame=frame)
                    try:
                        out.append(repr(val))
                    except Exception as re:
                        out.append(f'<repr error: {re}>')
                except Exception as ee:
                    out.append(f'<eval error: {ee}>')
            return out

        def do_watch(self, arg):
            expr = (arg or '').strip()
            if getattr(self, 'ui_mode', 'cli') != 'web':
                print("watch is not supported in terminal mode")
                return False
            if not expr:
                if not self.watch_exprs:
                    print("watch list is empty")
                    return False
                for i, e in enumerate(self.watch_exprs, start=1):
                    print(f"[{i}] {e}")
                return False
            if expr in self.watch_exprs:
                print(f"watch already exists: {expr}")
                return False
            self.watch_exprs.append(expr)
            _WEB_STATE["watch_exprs"] = list(self.watch_exprs)
            print(f"watch added: [{len(self.watch_exprs)}] {expr}")
            return False

        def do_unwatch(self, arg):
            target = (arg or '').strip()
            if getattr(self, 'ui_mode', 'cli') != 'web':
                print("unwatch is not supported in terminal mode")
                return False
            if not target:
                print("Usage: unwatch <expr> | unwatch <index>")
                if self.watch_exprs:
                    for i, e in enumerate(self.watch_exprs, start=1):
                        print(f"[{i}] {e}")
                else:
                    print("watch list is empty")
                return False
            try:
                idx = int(target)
            except Exception:
                idx = None
            if idx is not None:
                if idx < 1 or idx > len(self.watch_exprs):
                    print(f"invalid watch index: {idx}")
                    return False
                removed = self.watch_exprs.pop(idx - 1)
                _WEB_STATE["watch_exprs"] = list(self.watch_exprs)
                print(f"watch removed: [{idx}] {removed}")
                return False
            if target in self.watch_exprs:
                self.watch_exprs.remove(target)
                _WEB_STATE["watch_exprs"] = list(self.watch_exprs)
                print(f"watch removed: {target}")
                return False
            print(f"watch not found: {target}")
            return False

        def onecmd(self, line):
            """侵入式重写onecmd方法，添加详细的错误处理机制""" 
            # 检查是否在定义命令列表（IPython特有功能）
            if hasattr(self, 'commands_defining') and self.commands_defining:
                return self.handle_command_def(line)
            
            cmd, arg, line = self.parseline(line)
            if not line:
                return self.emptyline()
            if cmd is None:
                return self.default(line)
            self.lastcmd = line
            if line == 'EOF' :
                self.lastcmd = ''
            if cmd == '':
                return self.default(line) # pdb.default
            else:
                do_attr = f"do_{cmd}"
                if hasattr(self, do_attr) and _is_assignment_statement(line):
                    return self.default(line)
                if hasattr(self, do_attr):
                    func = getattr(self, do_attr)
                    return func(arg)
                else:
                    return self.default(line)

        @contextmanager
        def silent_if_worker(self):
            if self.rank == 0:
                yield
            else:
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                try:
                    yield
                finally:
                    sys.stdout.close()
                    sys.stderr.close()
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr

        def web_cmdloop(self):
            self.preloop()
            stop = None
            while not stop:
                try:
                    cmd = web_pdb.COMMAND_QUEUE.get()
                    
                    # Capture stdout
                    import io
                    from bdb import BdbQuit
                    stdout_capture = io.StringIO()
                    original_stdout = sys.stdout
                    original_stderr = sys.stderr
                    
                    # Also capture self.stdout if possible
                    original_self_stdout = getattr(self, 'stdout', None)
                    original_self_stderr = getattr(self, 'stderr', None)
                    
                    sys.stdout = stdout_capture
                    sys.stderr = stdout_capture
                    self.stdout = stdout_capture
                    try:
                        self.stderr = stdout_capture
                    except Exception:
                        pass
                    gathered = [""]
                    
                    try:
                        line = self.precmd(cmd)
                        if '\n' in cmd:
                            frame = getattr(self, 'curframe', None)
                            if frame is None:
                                raise RuntimeError("No current frame for multiline execution")
                            try:
                                self._exec_in_frame(cmd, frame=frame, mode='exec')
                            except Exception:
                                import traceback as _traceback
                                _traceback.print_exc()
                            local_output = stdout_capture.getvalue()
                            payload = {"out": local_output, "watch": self._get_watch_values()}
                            gathered = [payload]
                            if GetDistInfo.is_distributed():
                                GetDistInfo.gather_object_list(gathered, dst=0)
                            GetDistInfo.barrier()
                            stop = False
                        else:
                            stop = self.onecmd(line)
                            local_output = stdout_capture.getvalue()
                            payload = {"out": local_output, "watch": self._get_watch_values()}
                            gathered = [payload]
                            if GetDistInfo.is_distributed():
                                GetDistInfo.gather_object_list(gathered, dst=0)
                            stop = self.postcmd(stop, line)
                    except BdbQuit:
                        output = stdout_capture.getvalue()
                        payload = {"out": output, "watch": self._get_watch_values()}
                        gathered = [payload]
                        if GetDistInfo.is_distributed():
                            GetDistInfo.gather_object_list(gathered, dst=0)
                        if self.rank == 0:
                            if gathered[0] and isinstance(gathered[0], list):
                                by_rank = {str(i): ((gathered[0][i] or {}).get("out", "") if isinstance(gathered[0][i], dict) else (gathered[0][i] or "")) for i in range(len(gathered[0]))}
                                watch_by_rank = {str(i): ((gathered[0][i] or {}).get("watch", []) if isinstance(gathered[0][i], dict) else []) for i in range(len(gathered[0]))}
                            else:
                                by_rank = {"0": output}
                                watch_by_rank = {"0": payload.get("watch", [])}
                            web_pdb.RESPONSE_QUEUE.put({"result_by_rank": by_rank, "watch_exprs": list(self.watch_exprs or []), "watch_by_rank": watch_by_rank, "exited": True})
                        stop = True
                        self.quitting = True
                        break
                    except Exception as e:
                        print(f"Error executing command: {e}")
                        payload = {"out": stdout_capture.getvalue(), "watch": self._get_watch_values()}
                        gathered = [payload]
                    finally:
                        sys.stdout = original_stdout
                        sys.stderr = original_stderr
                        if original_self_stdout:
                            self.stdout = original_self_stdout
                        if original_self_stderr:
                            try:
                                self.stderr = original_self_stderr
                            except Exception:
                                pass
                        
                    if self.rank == 0:
                        if gathered[0] and isinstance(gathered[0], list):
                            by_rank = {str(i): ((gathered[0][i] or {}).get("out", "") if isinstance(gathered[0][i], dict) else (gathered[0][i] or "")) for i in range(len(gathered[0]))}
                            watch_by_rank = {str(i): ((gathered[0][i] or {}).get("watch", []) if isinstance(gathered[0][i], dict) else []) for i in range(len(gathered[0]))}
                        else:
                            by_rank = {"0": (gathered[0] or {}).get("out", stdout_capture.getvalue()) if isinstance(gathered[0], dict) else stdout_capture.getvalue()}
                            watch_by_rank = {"0": (gathered[0] or {}).get("watch", []) if isinstance(gathered[0], dict) else []}
                        web_pdb.RESPONSE_QUEUE.put({"result_by_rank": by_rank, "watch_exprs": list(self.watch_exprs or []), "watch_by_rank": watch_by_rank})
                    
                    if stop:
                        break
                except KeyboardInterrupt:
                    if self.rank == 0:
                        try:
                            web_pdb.RESPONSE_QUEUE.put({"result_by_rank": {"0": "Interrupted (Ctrl+C)\n"}, "exited": True})
                        except Exception:
                            pass
                        try:
                            web_pdb.stop_web_server("Interrupted by Ctrl+C")
                        except Exception:
                            pass
                        self.quitting = True
                        try:
                            GetDistInfo.broadcast_object_list([None], src=0)
                        except Exception:
                            pass
                    break
            self.postloop()
                

        def interaction(self, frame, traceback):
            self.setup(frame, traceback)
            if self.rank == 0:
                while True:
                    if self.web_mode and self.ui_mode == 'web':
                        if self.web_port is None:
                            web_pdb.start_web_server()
                        else:
                            web_pdb.start_web_server(port=int(self.web_port))
                        try:
                            port = web_pdb.get_server_port()
                            if port:
                                print(f"Web Pdb server running at http://localhost:{port}", flush=True)
                        except Exception:
                            pass
                        try:
                            web_pdb.set_debugger(self)
                        except Exception:
                            pass
                        try:
                            self.web_cmdloop()
                        finally:
                            if getattr(self, 'quitting', False):
                                try:
                                    web_pdb.set_debugger(None)
                                except Exception:
                                    pass
                                try:
                                    web_pdb.stop_web_server("Debugger quit")
                                except Exception:
                                    pass
                    else:
                        self.cmdloop()
                    if getattr(self, 'quitting', False):
                        break
                    if getattr(self, '_ui_switch_requested', False):
                        self._ui_switch_requested = False
                        continue
                    break
                # Tell workers to quit
                GetDistInfo.broadcast_object_list([None], src=0)
            else:
                # Worker processes receive and execute commands
                while True:
                    bcast_data = [None]
                    GetDistInfo.broadcast_object_list(bcast_data, src=0)
                    line = bcast_data[0]
                    if line is None:  # Quit signal from master
                        break
                    if self.web_mode and getattr(self, 'ui_mode', 'web') == 'web':
                        import io
                        from bdb import BdbQuit
                        stdout_capture = io.StringIO()
                        original_stdout = sys.stdout
                        original_stderr = sys.stderr
                        original_self_stdout = getattr(self, 'stdout', None)
                        original_self_stderr = getattr(self, 'stderr', None)
                        sys.stdout = stdout_capture
                        sys.stderr = stdout_capture
                        self.stdout = stdout_capture
                        try:
                            self.stderr = stdout_capture
                        except Exception:
                            pass
                        try:
                            try:
                                if '\n' in line:
                                    frame = getattr(self, 'curframe', None)
                                    if frame is None:
                                        raise RuntimeError("No current frame for multiline execution")
                                    try:
                                        self._exec_in_frame(line, frame=frame, mode='exec')
                                    except Exception:
                                        import traceback as _traceback
                                        _traceback.print_exc()
                                    stop = False
                                else:
                                    stop = self.onecmd(line)
                            except BdbQuit:
                                self.quitting = True
                                stop = True
                        finally:
                            sys.stdout = original_stdout
                            sys.stderr = original_stderr
                            if original_self_stdout:
                                self.stdout = original_self_stdout
                            if original_self_stderr:
                                try:
                                    self.stderr = original_self_stderr
                                except Exception:
                                    pass
                        payload = {"out": stdout_capture.getvalue(), "watch": self._get_watch_values()}
                        gathered = [payload]
                        if GetDistInfo.is_distributed():
                            GetDistInfo.gather_object_list(gathered, dst=0)
                    else:
                        with self.silent_if_worker():
                            stop = self.onecmd(line)
                    GetDistInfo.barrier()
                    if self.quitting:
                        break
                    if getattr(self, '_ui_switch_requested', False):
                        self._ui_switch_requested = False
            self.forget()
    return DistributedPdb



def _init_pdb(context=None, commands=[], sync_method='torch', sync_dir=None, rank=None, world_size=None, keep_alive_interval=0, web_mode=None, web_port=None):
    if context is None:
        context = os.getenv("IPDB_CONTEXT_SIZE", get_context_from_config())
    debugger_cls = _get_debugger_cls()
    
    GetDistInfo.initialize(sync_method=sync_method, sync_dir=sync_dir, rank=rank, world_size=world_size)
    GetDistInfo.keep_alive(keep_alive_interval)
    resolved_web_mode = _resolve_web_mode(web_mode)
    if resolved_web_mode and web_port is None:
        web_port = _WEB_STATE.get("last_web_port")
    if not resolved_web_mode and not _reopen_stdin_to_tty():
        resolved_web_mode = True
        _WEB_STATE["sticky_web_mode"] = True
        if GetDistInfo.get_rank() == 0:
            print("mpdb: stdin is not a TTY, falling back to web mode.", file=sys.stderr)
    if resolved_web_mode:
        _WEB_STATE["sticky_web_mode"] = True
        if web_port is not None:
            _WEB_STATE["last_web_port"] = web_port

    try:
        p = debugger_cls(context=context, web_mode=resolved_web_mode, web_port=web_port)
    except TypeError:
        p = debugger_cls(web_mode=resolved_web_mode, web_port=web_port)
    p.rcLines.extend(commands)
    return p


def wrap_sys_excepthook():
    # make sure we wrap it only once or we would end up with a cycle
    #  BdbQuit_excepthook.excepthook_ori == BdbQuit_excepthook
    if sys.excepthook != BdbQuit_excepthook:
        BdbQuit_excepthook.excepthook_ori = sys.excepthook
        sys.excepthook = BdbQuit_excepthook

def set_trace(frame=None, context=None, cond=True, *, not_distributed=False, sync_method='torch', sync_dir=None, rank=None, world_size=None, keep_alive_interval=0, web_mode=None, web_port=None):
    if not cond:
        return
    GetDistInfo._FLAG_NOT_DISTRIUBTED_ = not_distributed
    wrap_sys_excepthook()
    if frame is None:
        frame = sys._getframe().f_back
    p = _init_pdb(context=context, sync_method=sync_method, sync_dir=sync_dir, rank=rank, world_size=world_size, keep_alive_interval=keep_alive_interval, web_mode=web_mode, web_port=web_port).set_trace(frame)
    if p and hasattr(p, 'shell'):
        p.shell.restore_sys_module_state()


def get_context_from_config():
    parser = get_config()
    try:
        return parser.getint("ipdb", "context")
    except (configparser.NoSectionError, configparser.NoOptionError):
        return 3
    except ValueError:
        value = parser.get("ipdb", "context")
        raise ValueError(
            "In %s,  context value [%s] cannot be converted into an integer."
            % (parser.filepath, value)
        )


class ConfigFile(object):
    """
    Filehandle wrapper that adds a "[ipdb]" section to the start of a config
    file so that users don't actually have to manually add a [ipdb] section.
    Works with configparser versions from both Python 2 and 3
    """

    def __init__(self, filepath):
        self.first = True
        with open(filepath) as f:
            self.lines = f.readlines()

    # Python 2.7 (Older dot versions)
    def readline(self):
        try:
            return self.__next__()
        except StopIteration:
            return ''

    # Python 2.7 (Newer dot versions)
    def next(self):
        return self.__next__()

    # Python 3
    def __iter__(self):
        return self

    def __next__(self):
        if self.first:
            self.first = False
            return "[ipdb]\n"
        if self.lines:
            return self.lines.pop(0)
        raise StopIteration


def get_config():
    """
    Get ipdb config file settings.
    All available config files are read.  If settings are in multiple configs,
    the last value encountered wins.  Values specified on the command-line take
    precedence over all config file settings.
    Returns: A ConfigParser object.
    """
    parser = configparser.ConfigParser()

    filepaths = []

    # Low priority goes first in the list
    for cfg_file in ("setup.cfg", ".ipdb", "pyproject.toml"):
        cwd_filepath = os.path.join(os.getcwd(), cfg_file)
        if os.path.isfile(cwd_filepath):
            filepaths.append(cwd_filepath)

    # Medium priority (whenever user wants to set a specific path to config file)
    home = os.getenv("HOME")
    if home:
        default_filepath = os.path.join(home, ".ipdb")
        if os.path.isfile(default_filepath):
            filepaths.append(default_filepath)

    # High priority (default files)
    env_filepath = os.getenv("IPDB_CONFIG")
    if env_filepath and os.path.isfile(env_filepath):
        filepaths.append(env_filepath)

    if filepaths:
        # Python 3 has parser.read_file(iterator) while Python2 has
        # parser.readfp(obj_with_readline)
        try:
            read_func = parser.read_file
        except AttributeError:
            read_func = parser.readfp
        for filepath in filepaths:
            parser.filepath = filepath
            # Users are expected to put an [ipdb] section
            # only if they use setup.cfg
            if filepath.endswith('setup.cfg'):
                with open(filepath) as f:
                    parser.remove_section("ipdb")
                    read_func(f)
            # To use on pyproject.toml, put [tool.ipdb] section
            elif filepath.endswith('pyproject.toml'):
                try:
                    import tomllib
                    file_mode = "rb"
                except ImportError:
                    try:
                        import tomli as tomllib
                        file_mode = "rb"
                    except ImportError:
                        import toml as tomllib
                        file_mode = "r"
                with open(filepath, file_mode) as f:
                    toml_file = tomllib.load(f)
                    if "tool" in toml_file and "ipdb" in toml_file["tool"]:
                        if not parser.has_section("ipdb"):
                            parser.add_section("ipdb")
                        for key, value in toml_file["tool"]["ipdb"].items():
                            parser.set("ipdb", key, str(value))
            else:
                read_func(ConfigFile(filepath))
    return parser


def post_mortem(tb=None):
    wrap_sys_excepthook()
    p = _init_pdb()
    p.reset()
    if tb is None:
        # sys.exc_info() returns (type, value, traceback) if an exception is
        # being handled, otherwise it returns None
        tb = sys.exc_info()[2]
    if tb:
        p.interaction(None, tb)


def pm():
    post_mortem(sys.last_traceback)


def run(statement, globals=None, locals=None):
    _init_pdb().run(statement, globals, locals)

def runcall(*args, **kwargs):
    return _init_pdb().runcall(*args, **kwargs)


def runeval(expression, globals=None, locals=None):
    return _init_pdb().runeval(expression, globals, locals)


@contextmanager
def launch_ipdb_on_exception():
    try:
        yield
    except Exception:
        e, m, tb = sys.exc_info()
        print(m.__repr__(), file=sys.stderr)
        post_mortem(tb)
    finally:
        pass


# iex is a concise alias
iex = launch_ipdb_on_exception()


_usage = """\
usage: python -m ipdb [-m] [-c command] ... pyfile [arg] ...

Debug the Python program given by pyfile.

Initial commands are read from .pdbrc files in your home directory
and in the current directory, if they exist.  Commands supplied with
-c are executed after commands from .pdbrc files.

To let the script run until an exception occurs, use "-c continue".
To let the script run up to a given line X in the debugged file, use
"-c 'until X'"

Option -m is available only in Python 3.7 and later.

ipdb version %s.""" % __version__


def main():
    import traceback
    import sys
    import getopt

    try:
        from pdb import Restart
    except ImportError:
        class Restart(Exception):
            pass

    if sys.version_info >= (3, 7):
        opts, args = getopt.getopt(sys.argv[1:], 'mhc:', ['help', 'command='])
    else:
        opts, args = getopt.getopt(sys.argv[1:], 'hc:', ['help', 'command='])

    commands = []
    run_as_module = False
    for opt, optarg in opts:
        if opt in ['-h', '--help']:
            print(_usage)
            sys.exit()
        elif opt in ['-c', '--command']:
            commands.append(optarg)
        elif opt in ['-m']:
            run_as_module = True

    if not args:
        print(_usage)
        sys.exit(2)

    mainpyfile = args[0]     # Get script filename
    if not run_as_module and not os.path.exists(mainpyfile):
        print('Error:', mainpyfile, 'does not exist')
        sys.exit(1)

    sys.argv = args     # Hide "pdb.py" from argument list

    # Replace pdb's dir with script's dir in front of module search path.
    if not run_as_module:
        sys.path[0] = os.path.dirname(mainpyfile)

    # Note on saving/restoring sys.argv: it's a good idea when sys.argv was
    # modified by the script being debugged. It's a bad idea when it was
    # changed by the user from the command line. There is a "restart" command
    # which allows explicit specification of command line arguments.
    pdb = _init_pdb(commands=commands)
    while 1:
        try:
            import pdb as stdlib_pdb
            if hasattr(stdlib_pdb.Pdb, "_run"):
                # Looks like Pdb from Python 3.11+
                if run_as_module:
                    pdb._run(stdlib_pdb._ModuleTarget(mainpyfile))
                else:
                    pdb._run(stdlib_pdb._ScriptTarget(mainpyfile))
            else:
                if run_as_module:
                    pdb._runmodule(mainpyfile)
                else:
                    pdb._runscript(mainpyfile)
            if pdb._user_requested_quit:
                break
            print("The program finished and will be restarted")
        except Restart:
            print("Restarting", mainpyfile, "with arguments:")
            print("\t" + " ".join(sys.argv[1:]))
        except SystemExit:
            # In most cases SystemExit does not warrant a post-mortem session.
            print("The program exited via sys.exit(). Exit status: ", end='')
            print(sys.exc_info()[1])
        except:
            traceback.print_exc()
            print("Uncaught exception. Entering post mortem debugging")
            print("Running 'cont' or 'step' will restart the program")
            t = sys.exc_info()[2]
            pdb.interaction(None, t)
            print("Post mortem debugger finished. The " + mainpyfile +
                  " will be restarted")


if __name__ == '__main__':
    main()
