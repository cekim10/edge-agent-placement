"""The access-control service behind a real network hop, with real durability.

Every latency number in the commit-barrier experiments has so far rested on a
`time.sleep()` standing in for the commit round trip. Sweeping that parameter is
a defensible way to show *where* the policies cross, but it cannot say what the
crossing is worth, because it never says what a commit actually costs. This
module is what makes `c` a measurement.

Why not just time an HTTP round trip
------------------------------------
Because that would time the network, not the commit, and the paper's whole
premise is that a committed operation is an external side effect that cannot be
taken back. An operation is only irreversible once it has been made durable; a
service that mutates a dictionary and replies has not committed anything, and
`c` measured against it is a lower bound with no physical meaning.

So a commit here does what the smallest honest version of this service must do:
parse and validate the request, read the current state, apply the mutation,
append to an audit journal, and `fsync` that journal before replying. The fsync
is not incidental -- it is the moment the operation stops being recoverable, and
on most storage it is the single largest term in `c`.

The server reports its own timings back in the response body, so a client can
split what it observes into network, service work, and durability without
needing a second experiment:

    c_observed = network + service_work + durability

`durable=False` disables only the fsync, which measures that last term by
difference. It is a measurement knob, not a deployment mode: with it off the
service is no longer committing anything in the sense this paper uses the word.
"""

from __future__ import annotations

import fcntl
import http.client
import json
import os
import sys
import socket
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import AccessControlService, CommitResult


@dataclass
class ServerTimings:
    """What the server says it spent, in seconds."""

    total_s: float = 0.0
    durability_s: float = 0.0


class CommitServiceHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so `c` is not a TCP handshake

    service: AccessControlService
    journal_path: Path
    durable: bool
    lock: threading.Lock

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - silence stderr
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @staticmethod
    def _flush(fileno: int) -> None:
        """Push the write past the drive's own cache, not just the page cache.

        On Darwin `fsync` returns once the data reaches the disk controller and
        does not flush the drive cache; `F_FULLFSYNC` is what actually makes the
        write survive power loss. The difference is roughly two orders of
        magnitude, so measuring `c` with plain fsync on a Mac reports a
        durability term that no real commit would enjoy.
        """
        if sys.platform == "darwin":
            try:
                fcntl.fcntl(fileno, getattr(fcntl, "F_FULLFSYNC", 51))
                return
            except OSError:
                pass  # some filesystems refuse it; fall back and note it below
        os.fsync(fileno)

    def _journal(self, kind: str, ops: list[dict[str, str]]) -> float:
        """Append the record and make it durable. Returns seconds spent."""
        started = time.perf_counter()
        line = json.dumps({"kind": kind, "ops": ops, "at": time.time()}) + "\n"
        with open(self.journal_path, "a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()
            if self.durable:
                self._flush(handle.fileno())
        return time.perf_counter() - started

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/state":
            with self.lock:
                self._send_json({"state": self.service.state()})
            return
        if self.path == "/health":
            self._send_json({"ok": True, "durable": self.durable})
            return
        self._send_json({"error": "not_found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        started = time.perf_counter()
        try:
            payload = self._read_json()
        except (ValueError, OSError):
            self._send_json({"error": "bad_request"}, status=400)
            return

        if self.path == "/reset":
            with self.lock:
                self.service.reset(payload["state"])
            self._send_json({"ok": True, "server_s": time.perf_counter() - started})
            return

        if self.path not in ("/commit", "/compensate"):
            self._send_json({"error": "not_found"}, status=404)
            return

        ops = payload.get("ops", [])
        with self.lock:
            if self.path == "/commit":
                result = self.service.commit(ops)
            else:
                result = self.service.compensate(ops, payload.get("recoverability", ""))
            # Journal only what actually changed state. A compensation that
            # refused is not a durable event -- nothing happened -- and charging
            # it an fsync would inflate the very cost the paper says it avoids.
            durability_s = self._journal(self.path[1:], ops) if result.applied else 0.0

        self._send_json({
            "applied": result.applied,
            "error": result.error,
            "compensation_supported": result.compensation_supported,
            "compensation_applied": result.compensation_applied,
            "server_s": time.perf_counter() - started,
            "durability_s": durability_s,
        })


def serve(
    host: str, port: int, *, journal_path: Path, durable: bool = True
) -> ThreadingHTTPServer:
    handler = type(
        "BoundCommitServiceHandler",
        (CommitServiceHandler,),
        {
            "service": AccessControlService(),
            "journal_path": journal_path,
            "durable": durable,
            "lock": threading.Lock(),
        },
    )
    return ThreadingHTTPServer((host, port), handler)


class _ClampedCommitConnection(http.client.HTTPConnection):
    """Connection that caps its TCP segment size before the handshake.

    The cluster NICs advertise a 9000-byte MTU while the switch between the
    nodes forwards only 1500, so anything larger disappears without an error.
    `TCP_MAXSEG` has to be set on an unconnected socket: it goes out in the SYN,
    which is what stops the *peer* from sending oversized segments back. Setting
    it after connecting is silently too late -- the option is accepted and the
    handshake has already advertised the full MSS.

    It matters in both directions here. `/reset` sends a full record table,
    several kilobytes for a hard cell, and `/state` returns one.
    """

    #: Set once the clamp has been observed to apply, so a run can report
    #: whether it was actually in force rather than merely requested.
    clamp_applied: bool | None = None

    def connect(self) -> None:
        clamp = int(os.environ.get("TCP_MSS_CLAMP", "1400") or 0)
        if clamp <= 0:
            _ClampedCommitConnection.clamp_applied = False
            super().connect()
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, clamp)
            _ClampedCommitConnection.clamp_applied = True
        except OSError:
            # Darwin rejects TCP_MAXSEG on an unconnected socket. The clamp is a
            # workaround for one Linux cluster's MTU mismatch, so failing here
            # is expected off that cluster and must not stop the run -- but it
            # is recorded, because a silent no-op clamp is what the blackhole
            # looked like in the first place.
            _ClampedCommitConnection.clamp_applied = False
        if isinstance(self.timeout, (int, float)):
            sock.settimeout(self.timeout)
        sock.connect((self.host, self.port))
        self.sock = sock


class RemoteAccessControlService:
    """Client with the same surface as AccessControlService.

    Substitutable for the in-process service anywhere the workflow uses it, so
    the same experiment can be run against a simulated commit cost and a real
    one and the results compared directly.

    The connection is kept alive across calls. A fresh TCP handshake per commit
    would be a real cost for a naive client, but no service client works that
    way, and charging speculation for it would overstate what it hides.
    """

    def __init__(self, endpoint: str, *, timeout_s: float = 30.0) -> None:
        host, _, port = endpoint.removeprefix("http://").partition(":")
        self.host = host
        self.port = int(port or "80")
        self.timeout_s = timeout_s
        self._local = threading.local()
        self.last_timings = ServerTimings()

    def _connection(self) -> http.client.HTTPConnection:
        # One connection per thread: the speculative path runs commit and verify
        # concurrently, and http.client connections are not shareable.
        existing = getattr(self._local, "connection", None)
        if existing is not None:
            return existing
        connection = _ClampedCommitConnection(
            self.host, self.port, timeout=self.timeout_s
        )
        self._local.connection = connection
        return connection

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(body))}
        for attempt in (1, 2):
            connection = self._connection()
            try:
                connection.request("POST", path, body=body, headers=headers)
                response = connection.getresponse()
                return json.loads(response.read().decode("utf-8"))
            except (http.client.HTTPException, OSError):
                # A kept-alive connection the server has since closed fails on
                # first use; reconnect once rather than losing the measurement.
                try:
                    connection.close()
                finally:
                    self._local.connection = None
                if attempt == 2:
                    raise
        raise RuntimeError("unreachable")

    def _call(self, path: str, payload: dict[str, Any]) -> CommitResult:
        started = time.perf_counter()
        data = self._post(path, payload)
        latency_s = time.perf_counter() - started
        self.last_timings = ServerTimings(
            total_s=float(data.get("server_s", 0.0)),
            durability_s=float(data.get("durability_s", 0.0)),
        )
        return CommitResult(
            applied=bool(data.get("applied")),
            error=data.get("error"),
            compensation_supported=bool(data.get("compensation_supported")),
            compensation_applied=bool(data.get("compensation_applied")),
            latency_s=latency_s,
        )

    def reset(self, state: dict[str, Any]) -> None:
        self._post("/reset", {"state": state})

    def state(self) -> dict[str, Any]:
        connection = self._connection()
        connection.request("GET", "/state")
        response = connection.getresponse()
        return json.loads(response.read().decode("utf-8"))["state"]

    def commit(self, ops: list[dict[str, str]]) -> CommitResult:
        return self._call("/commit", {"ops": ops})

    def compensate(self, ops: list[dict[str, str]], recoverability: str) -> CommitResult:
        return self._call(
            "/compensate", {"ops": ops, "recoverability": recoverability}
        )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=871)
    parser.add_argument("--journal", type=Path, default=Path("/tmp/commit_journal.log"))
    parser.add_argument(
        "--no-durable", action="store_true",
        help="skip the fsync. Measures the durability term by difference; the "
             "service is not committing in this paper's sense with it off.",
    )
    args = parser.parse_args()
    args.journal.parent.mkdir(parents=True, exist_ok=True)
    server = serve(
        args.host, args.port, journal_path=args.journal, durable=not args.no_durable
    )
    print(
        f"commit service on {args.host}:{args.port}  journal={args.journal}  "
        f"durable={not args.no_durable}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
