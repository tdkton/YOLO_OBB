"""
Package transport: send one JSON line per board over a TCP socket.

Non-blocking-ish with lazy auto-reconnect so a temporarily-down host/PLC does
not crash the vision loop. Each package is newline-terminated for easy parsing
on the receiver side.
"""
from __future__ import annotations
import json
import socket
import time


class PackageSender:
    def __init__(self, host: str, port: int, reconnect_seconds: float = 2.0, enabled: bool = True):
        self.host = host
        self.port = int(port)
        self.reconnect_seconds = float(reconnect_seconds)
        self.enabled = enabled
        self.sock: socket.socket | None = None
        self._last_attempt = 0.0

    def _connect(self) -> None:
        now = time.time()
        if now - self._last_attempt < self.reconnect_seconds:
            return
        self._last_attempt = now
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1.0)
            s.connect((self.host, self.port))
            self.sock = s
            print(f"[comms] connected to {self.host}:{self.port}")
        except OSError as e:
            self.sock = None
            print(f"[comms] connect failed ({e}); will retry")

    def send(self, package: dict) -> None:
        """Print the package always; transmit over TCP if comms are enabled."""
        line = json.dumps(package)
        print(f"[PACKAGE] {line}")
        if not self.enabled:
            return
        if self.sock is None:
            self._connect()
        if self.sock is None:
            return
        try:
            self.sock.sendall((line + "\n").encode("utf-8"))
        except OSError as e:
            print(f"[comms] send failed ({e}); dropping connection")
            self.close()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            finally:
                self.sock = None
