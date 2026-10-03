"""pii-sentinel serve: keep the model loaded and scan text sent over HTTP.

The server listens on a Unix socket (created with mode 600) or on a loopback address only. It is stateless and
never logs or stores the text it receives. Requests are handled one at a time.

    GET  /health  -> {"status": "ok", "model": ..., "version": ...}
    POST /scan    {"text": "...", "rules": true, "show_values": false} -> the report of `scan --json` for one text
"""

import json
import os
import signal
import socket
import socketserver
import stat
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .predict import analyse

LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
DEFAULT_MAX_BYTES = 1_000_000
DISCARD_LIMIT = 64_000_000


def package_version():
    try:
        return version("pii-sentinel")
    except PackageNotFoundError:
        return "unknown"


class Scanner:
    """The loaded model and how to call it; the lock keeps one inference at a time."""

    def __init__(self, model, tokenizer, meta, device, mask):
        self.model, self.tokenizer, self.meta, self.device, self.mask = model, tokenizer, meta, device, mask
        self.lock = threading.Lock()

    def scan(self, text, use_rules=True, show_values=False):
        with self.lock:
            res = analyse(
                self.model,
                self.tokenizer,
                text,
                self.device,
                max_length=self.meta.get("max_length", 512),
                use_rules=use_rules,
                doc_pooling=self.meta.get("doc_pooling", "per_window"),
            )
        if not show_values:
            for f in res["findings"] + res["secrets"]:
                f["value"] = self.mask(f["value"], f["type"])
        return res


def make_handler(scanner, max_bytes, allowed_hosts, verbose):
    class Handler(BaseHTTPRequestHandler):
        server_version = "pii-sentinel"
        sys_version = ""
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            # The request line holds only the method and path; bodies are never logged.
            if verbose:
                sys.stderr.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {format % args}\n")

        def send_json(self, status, body):
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status, message):
            self.close_connection = True
            self.send_json(status, {"error": message})

        def host_allowed(self):
            # Over TCP, refuse a Host header that is not a loopback name: a web page that rebinds its own domain
            # to 127.0.0.1 would otherwise be able to call the server and read the answers.
            if allowed_hosts is None:
                return True
            host = (self.headers.get("Host") or "").strip()
            name = host[1:].split("]")[0] if host.startswith("[") else host.rsplit(":", 1)[0]
            return name in allowed_hosts

        def do_GET(self):
            if not self.host_allowed():
                return self.error(HTTPStatus.FORBIDDEN, "host not allowed")
            if self.path != "/health":
                return self.error(HTTPStatus.NOT_FOUND, "not found")
            self.send_json(
                HTTPStatus.OK,
                {"status": "ok", "model": scanner.meta.get("model_name"), "version": package_version()},
            )

        def discard(self, length):
            # Read a refused body before answering, so the client is not cut off while it is still sending; past
            # DISCARD_LIMIT the connection is just closed.
            left = min(length, DISCARD_LIMIT)
            while left > 0:
                chunk = self.rfile.read(min(left, 65536))
                if not chunk:
                    break
                left -= len(chunk)

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self.error(HTTPStatus.LENGTH_REQUIRED, "Content-Length required")
            if length < 0:
                return self.error(HTTPStatus.BAD_REQUEST, "invalid Content-Length")
            if length > max_bytes:
                self.discard(length)
                return self.error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"body larger than {max_bytes} bytes")
            raw = self.rfile.read(length)
            if not self.host_allowed():
                return self.error(HTTPStatus.FORBIDDEN, "host not allowed")
            if self.path != "/scan":
                return self.error(HTTPStatus.NOT_FOUND, "not found")
            # A JSON content type cannot be sent cross-origin without a CORS preflight, which is never answered.
            if (self.headers.get("Content-Type") or "").split(";")[0].strip() != "application/json":
                return self.error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "Content-Type must be application/json")
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                return self.error(HTTPStatus.BAD_REQUEST, "body is not JSON")
            if not isinstance(body, dict) or not isinstance(body.get("text"), str):
                return self.error(HTTPStatus.BAD_REQUEST, 'expected {"text": "..."}')
            res = scanner.scan(
                body["text"],
                use_rules=body.get("rules", True) is not False,
                show_values=body.get("show_values") is True,
            )
            self.send_json(HTTPStatus.OK, res)

    return Handler


class UnixHTTPServer(socketserver.UnixStreamServer):
    allow_reuse_address = False

    def server_bind(self):
        # Create the socket with no permissions for others from the start, then make it owner read/write only.
        old = os.umask(0o177)
        try:
            super().server_bind()
        finally:
            os.umask(old)
        os.chmod(self.server_address, stat.S_IRUSR | stat.S_IWUSR)

    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)


def remove_stale_socket(path: Path):
    """Remove a socket left by a server that is gone; refuse to touch anything else or a live server."""
    if not path.exists() and not path.is_symlink():
        return
    if not stat.S_ISSOCK(path.lstat().st_mode):
        raise SystemExit(f"{path} exists and is not a socket")
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.connect(str(path))
    except OSError:
        path.unlink()
    else:
        raise SystemExit(f"another server is listening on {path}")
    finally:
        probe.close()


def build_server(
    scanner, socket_path=None, host="127.0.0.1", port=8765, max_bytes=DEFAULT_MAX_BYTES, verbose=False
):
    if socket_path is not None:
        path = Path(socket_path).expanduser()
        # sun_path holds 104 bytes on macOS and 108 on Linux, including the terminating NUL.
        if len(os.fsencode(path)) > 103:
            raise SystemExit(f"socket path too long ({len(os.fsencode(path))} bytes, at most 103): {path}")
        remove_stale_socket(path)
        server = UnixHTTPServer(str(path), make_handler(scanner, max_bytes, None, verbose))
        return server, f"unix:{path}"
    if host not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"--host must be a loopback address ({', '.join(sorted(LOOPBACK_HOSTS))}), got {host}"
        )

    class Server(HTTPServer):
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    server = Server((host, port), make_handler(scanner, max_bytes, LOOPBACK_HOSTS, verbose))
    return server, f"http://{'[' + host + ']' if ':' in host else host}:{server.server_address[1]}"


def serve(scanner, **kwargs):
    server, where = build_server(scanner, **kwargs)

    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"pii-sentinel {package_version()} listening on {where}", file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if isinstance(server, UnixHTTPServer):
            Path(server.server_address).unlink(missing_ok=True)
