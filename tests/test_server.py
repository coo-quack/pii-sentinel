import http.client
import json
import os
import socket
import stat
import tempfile
import threading
from pathlib import Path

import pytest

from pii_sentinel import server as S
from pii_sentinel.cli import mask


class FakeScanner:
    meta = {"model_name": "fake"}

    def __init__(self):
        self.calls = []

    def scan(self, text, use_rules=True, show_values=False):
        self.calls.append((text, use_rules, show_values))
        value = "Emily Carter" if show_values else mask("Emily Carter", "person_name")
        return {
            "sensitivity": {"level": "low"},
            "findings": [{"type": "person_name", "value": value}],
            "secrets": [],
        }


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost")
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self.path)


@pytest.fixture
def tmp_path():
    # tmp_path under macOS's $TMPDIR is longer than a Unix socket path may be.
    with tempfile.TemporaryDirectory(prefix="ps-", dir="/tmp") as d:
        yield Path(d)


@pytest.fixture
def running(tmp_path):
    servers = []

    def start(**kwargs):
        scanner = FakeScanner()
        server, where = S.build_server(scanner, **kwargs)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server, scanner

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def request(conn, method, path, body=None, headers=None):
    conn.request(method, path, body=body, headers=headers or {})
    res = conn.getresponse()
    return res.status, json.loads(res.read())


def test_unix_socket_is_private_and_scans(tmp_path, running):
    path = tmp_path / "ps.sock"
    running(socket_path=str(path))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    conn = UnixConnection(str(path))
    assert request(conn, "GET", "/health")[1]["status"] == "ok"
    status, body = request(
        conn, "POST", "/scan", json.dumps({"text": "Emily Carter"}), {"Content-Type": "application/json"}
    )
    assert status == 200 and body["findings"][0]["value"] == "E…"


def test_scan_options_are_passed(tmp_path, running):
    path = tmp_path / "ps.sock"
    _, scanner = running(socket_path=str(path))
    conn = UnixConnection(str(path))
    body = json.dumps({"text": "x", "rules": False, "show_values": True})
    request(conn, "POST", "/scan", body, {"Content-Type": "application/json"})
    assert scanner.calls == [("x", False, True)]


@pytest.mark.parametrize(
    ("body", "headers", "status"),
    [
        ('{"text": "x"}', {"Content-Type": "text/plain"}, 415),
        ("not json", {"Content-Type": "application/json"}, 400),
        ('{"txt": "x"}', {"Content-Type": "application/json"}, 400),
        ('{"text": 1}', {"Content-Type": "application/json"}, 400),
    ],
)
def test_bad_requests_are_refused(tmp_path, running, body, headers, status):
    path = tmp_path / "ps.sock"
    _, scanner = running(socket_path=str(path))
    assert request(UnixConnection(str(path)), "POST", "/scan", body, headers)[0] == status
    assert scanner.calls == []


def test_large_body_is_refused(tmp_path, running):
    path = tmp_path / "ps.sock"
    running(socket_path=str(path), max_bytes=10)
    body = json.dumps({"text": "x" * 100})
    assert (
        request(UnixConnection(str(path)), "POST", "/scan", body, {"Content-Type": "application/json"})[0]
        == 413
    )


def test_tcp_refuses_foreign_host_header(running):
    server, _ = running(host="127.0.0.1", port=0)
    port = server.server_address[1]
    assert request(http.client.HTTPConnection("127.0.0.1", port), "GET", "/health")[0] == 200
    conn = http.client.HTTPConnection("127.0.0.1", port)
    assert request(conn, "GET", "/health", headers={"Host": "evil.example.com"})[0] == 403


def test_verbose_log_escapes_control_characters(capsys):
    handler_class = S.make_handler(FakeScanner(), 10, None, True)
    handler = handler_class.__new__(handler_class)
    line = "GET /\x1b]0;x\x07\r\n2000-01-01T00:00:00 fake HTTP/1.1"
    handler.log_message('"%s" %s %s', line, "404", "-")
    err = capsys.readouterr().err
    assert "\x1b" not in err and "\x07" not in err and "\r" not in err
    assert err.count("\n") == 1
    assert r"/\x1b]0;x\x07\x0d\x0a2000" in err


def test_non_loopback_host_is_refused():
    with pytest.raises(SystemExit):
        S.build_server(FakeScanner(), host="0.0.0.0", port=0)


def test_stale_socket_is_replaced_but_live_one_and_files_are_not(tmp_path, running):
    stale = tmp_path / "stale.sock"
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(str(stale))
    s.close()
    running(socket_path=str(stale))
    assert request(UnixConnection(str(stale)), "GET", "/health")[0] == 200
    with pytest.raises(SystemExit, match="another server"):
        S.build_server(FakeScanner(), socket_path=str(stale))
    regular = tmp_path / "file"
    regular.write_text("keep me")
    with pytest.raises(SystemExit, match="not a socket"):
        S.build_server(FakeScanner(), socket_path=str(regular))
    assert regular.read_text() == "keep me"


def test_too_long_socket_path_is_refused():
    with pytest.raises(SystemExit, match="too long"):
        S.build_server(FakeScanner(), socket_path="/tmp/" + "x" * 120)


def test_scanner_masks_secrets_by_default_and_keeps_offsets_with_show_values(monkeypatch):
    from pii_sentinel.cli import masked_report

    text = "Hello\nSecond line\npassword=hunter22\n"
    secret = text.index("hunter22")

    def fake_analyse(*args, **kwargs):
        return {
            "sensitivity": {"level": "high"},
            "findings": [],
            "secrets": [
                {
                    "type": "secret",
                    "value": "hunter22",
                    "start": secret,
                    "end": secret + 8,
                    "rule": "generic-password",
                }
            ],
        }

    monkeypatch.setattr(S, "analyse", fake_analyse)
    scanner = S.Scanner(None, None, {}, None, masked_report)
    assert scanner.scan(text)["secrets"] == [
        {"type": "secret", "value": "…", "line": 3, "rule": "generic-password"}
    ]
    raw = scanner.scan(text, show_values=True)["secrets"][0]
    assert raw["value"] == "hunter22" and raw["start"] == secret and raw["end"] == secret + 8
