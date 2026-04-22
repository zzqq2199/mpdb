import threading
import queue
import json
import os
import errno
import atexit
import time
import signal
from http.server import HTTPServer, SimpleHTTPRequestHandler
from typing import Dict, Optional

COMMAND_QUEUE = queue.Queue()
RESPONSE_QUEUE = queue.Queue()

_SERVER_THREAD = None
_HTTPD = None
_SERVER_PORT = None
_SERVER_LOCK = threading.Lock()
_STATUS_LOCK = threading.Lock()
_STATUS = {"state": "starting", "message": "", "ts": 0.0}
_DEBUGGER_LOCK = threading.Lock()
_DEBUGGER = None
_DESCRIPTION_LOCK = threading.Lock()
_DESCRIPTION: Dict[str, Optional[str]] = {"title": None, "subtitle": None}

def set_description(title: Optional[str] = None, subtitle: Optional[str] = None):
    with _DESCRIPTION_LOCK:
        if title is not None:
            _DESCRIPTION["title"] = str(title)
        if subtitle is not None:
            _DESCRIPTION["subtitle"] = str(subtitle)

def _get_description():
    with _DESCRIPTION_LOCK:
        return dict(_DESCRIPTION)

def _set_status(state, message=""):
    with _STATUS_LOCK:
        _STATUS["state"] = state
        _STATUS["message"] = message
        _STATUS["ts"] = time.time()

def _get_status():
    with _STATUS_LOCK:
        return dict(_STATUS)

def set_debugger(debugger):
    global _DEBUGGER
    with _DEBUGGER_LOCK:
        _DEBUGGER = debugger

def _get_debugger():
    with _DEBUGGER_LOCK:
        return _DEBUGGER

class WebPdbHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            html = None
            try:
                from importlib import resources as importlib_resources
                html = importlib_resources.files(__package__).joinpath('templates/index.html').read_bytes()
            except Exception:
                html = None

            if html is None:
                try:
                    import pkgutil
                    pkg = __package__
                    if pkg:
                        html = pkgutil.get_data(pkg, 'templates/index.html')
                except Exception:
                    html = None

            if html is None:
                template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
                try:
                    with open(template_path, 'rb') as f:
                        html = f.read()
                except FileNotFoundError:
                    html = None

            if html is None:
                self.wfile.write(
                    b"Error: Template not found. This usually means mpdb/templates/index.html "
                    b"was not included in the wheel. Please add package_data/include_package_data "
                    b"for mpdb templates and rebuild the wheel."
                )
            else:
                self.wfile.write(html)
        elif self.path == '/status':
            payload = _get_status()
            debugger = _get_debugger()
            if debugger:
                payload['ui_mode'] = getattr(debugger, 'ui_mode', 'web')
                payload['watch_exprs'] = list(getattr(debugger, 'watch_exprs', []) or [])
            desc = _get_description()
            if desc.get('title') is not None:
                payload['title'] = desc['title']
            if desc.get('subtitle') is not None:
                payload['subtitle'] = desc['subtitle']
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode('utf-8'))
        else:
            # Serve static files if needed, or 404
            super().do_GET()

    def do_POST(self):
        if self.path == '/execute':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                command = data.get('command')
                
                if command:
                    debugger = _get_debugger()
                    if debugger and getattr(debugger, 'ui_mode', 'web') == 'cli':
                        if hasattr(debugger, 'inject_command'):
                             debugger.inject_command(command)
                             self.send_response(200)
                             self.send_header('Content-type', 'application/json')
                             self.end_headers()
                             self.wfile.write(json.dumps({'status': 'ok', 'result': 'Command injected'}).encode('utf-8'))
                             return

                    COMMAND_QUEUE.put(command)
                    # Block until result is available
                    result = RESPONSE_QUEUE.get()
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    if isinstance(result, dict):
                        payload = {'status': 'ok'}
                        payload.update(result)
                    else:
                        payload = {'status': 'ok', 'result': result}
                    self.wfile.write(json.dumps(payload).encode('utf-8'))
                else:
                    self.send_error(400, "No command provided")
            except Exception as e:
                self.send_error(500, str(e))
        elif self.path == '/complete':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                line = data.get('line', '')
                cursor = data.get('cursor', None)
                if cursor is None:
                    cursor = len(line)
                cursor = int(cursor)

                debugger = _get_debugger()
                if debugger is None or not hasattr(debugger, 'web_complete'):
                    payload = {'status': 'ok', 'start': cursor, 'end': cursor, 'matches': []}
                else:
                    start, end, matches = debugger.web_complete(line, cursor)
                    payload = {'status': 'ok', 'start': int(start), 'end': int(end), 'matches': list(matches)}

                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(json.dumps(payload).encode('utf-8'))
            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress logging to keep console clean
        pass

import socket

class DualStackServer(HTTPServer):
    address_family = socket.AF_INET6
    
    def server_bind(self):
        # Allow dual stack (IPv4 + IPv6)
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except (AttributeError, OSError):
            # If IPV6_V6ONLY is not available or fails, ignore
            pass
        super().server_bind()

def _serve_forever(httpd):
    httpd.serve_forever()

def start_web_server(port=25555, max_port_tries=50):
    global _SERVER_THREAD, _SERVER_PORT, _HTTPD
    with _SERVER_LOCK:
        if _SERVER_THREAD is not None and _SERVER_THREAD.is_alive():
            return _SERVER_THREAD

        for i in range(max_port_tries):
            candidate_port = int(port) + i
            # Try Dual Stack (IPv6 + IPv4) first
            try:
                HTTPServer.allow_reuse_address = True
                # Bind to :: to listen on all IPv6 and IPv4 interfaces (if V6ONLY=0)
                httpd = DualStackServer(('::', candidate_port), WebPdbHandler)
                server_type = "IPv6+IPv4"
            except OSError:
                # Fallback to IPv4 only
                try:
                    HTTPServer.allow_reuse_address = True
                    httpd = HTTPServer(('0.0.0.0', candidate_port), WebPdbHandler)
                    server_type = "IPv4"
                except OSError as e:
                    if getattr(e, 'errno', None) == errno.EADDRINUSE:
                        continue
                    raise

            _HTTPD = httpd
            _SERVER_PORT = candidate_port
            t = threading.Thread(target=_serve_forever, args=(httpd,), daemon=True)
            t.start()
            _SERVER_THREAD = t
            _set_status("running", "")
            
            # Print access URLs
            print(f"DPDB Web Server ({server_type}) running on port {candidate_port}")
            print(f"Access links:")
            print(f"  Local:   http://localhost:{candidate_port}")
            print(f"           http://127.0.0.1:{candidate_port}")
            print(f"           http://[::1]:{candidate_port}")
            try:
                hostname = socket.gethostname()
                print(f"  Network: http://{hostname}:{candidate_port}")
                fqdn = socket.getfqdn()
                if fqdn and fqdn != hostname:
                    print(f"           http://{fqdn}:{candidate_port}")
            except Exception:
                pass
            print(f"  Wait for connection...", flush=True)

            _install_signal_handlers()
            return t

        raise RuntimeError("Failed to start web server")

def get_server_port():
    return _SERVER_PORT

def stop_web_server(message="Debugger exited"):
    global _HTTPD, _SERVER_THREAD
    with _SERVER_LOCK:
        if _HTTPD is None:
            _set_status("stopped", message)
            return
        _set_status("stopping", message)
        try:
            _HTTPD.shutdown()
        except Exception:
            pass
        try:
            _HTTPD.server_close()
        except Exception:
            pass
        _HTTPD = None
        _SERVER_THREAD = None
        _set_status("stopped", message)

_ORIG_SIGINT = None
_ORIG_SIGTERM = None

def _signal_handler(signum, frame):
    try:
        stop_web_server("Interrupted by signal")
    finally:
        if signum == signal.SIGINT and callable(_ORIG_SIGINT):
            _ORIG_SIGINT(signum, frame)
        elif signum == signal.SIGTERM and callable(_ORIG_SIGTERM):
            _ORIG_SIGTERM(signum, frame)

def _install_signal_handlers():
    global _ORIG_SIGINT, _ORIG_SIGTERM
    try:
        if _ORIG_SIGINT is None:
            _ORIG_SIGINT = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, _signal_handler)
        if _ORIG_SIGTERM is None:
            _ORIG_SIGTERM = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

def _atexit_handler():
    stop_web_server("Backend process exited")

atexit.register(_atexit_handler)
