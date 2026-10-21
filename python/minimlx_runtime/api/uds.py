from __future__ import annotations

import json
import socketserver
from pathlib import Path

from ..errors import error_response
from .worker import Worker


class _RequestHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        worker: Worker = self.server.worker  # type: ignore[attr-defined]
        for line in self.rfile:
            if not line.strip():
                continue
            request = None
            try:
                request = json.loads(line)
                responses = worker.handle_events(request)
                for response in responses:
                    self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
                    self.wfile.flush()
                continue
            except Exception as exc:  # keep the connection usable after one error
                command = request.get("command") if isinstance(request, dict) else None
                response = error_response(command, exc)
            self.wfile.write((json.dumps(response, ensure_ascii=False) + "\n").encode())
            self.wfile.flush()


class _UnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def serve_unix(worker: Worker, socket_path: str | Path) -> None:
    path = Path(socket_path).expanduser()
    if path.exists():
        path.unlink()
    server = _UnixServer(str(path), _RequestHandler)
    server.worker = worker  # type: ignore[attr-defined]
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if path.exists():
            path.unlink()
