"""Minimal local SMTP server (RFC 5321 subset) that stores received mail as .eml.

The agent's send_message tool speaks real SMTP via smtplib. In production you
point EDA_SMTP_HOST/PORT at your mail relay; for the demo this sink captures the
messages into data/outbox so they can be inspected.
"""
from __future__ import annotations

import socketserver
import threading
import time
from pathlib import Path

from . import config


class _Handler(socketserver.StreamRequestHandler):
    def _send(self, line: str):
        self.wfile.write((line + "\r\n").encode())

    def handle(self):
        self._send("220 eda-mailsink ESMTP ready")
        mail_from, rcpts = None, []
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            cmd = raw.decode(errors="replace").strip()
            verb = cmd[:4].upper()
            if verb in ("HELO", "EHLO"):
                self._send("250 eda-mailsink")
            elif verb == "MAIL":
                mail_from, rcpts = cmd[10:].strip("<> "), []
                self._send("250 OK")
            elif verb == "RCPT":
                rcpts.append(cmd[8:].strip("<> "))
                self._send("250 OK")
            elif verb == "DATA":
                self._send("354 End data with <CR><LF>.<CR><LF>")
                lines = []
                while True:
                    ln = self.rfile.readline()
                    if not ln or ln in (b".\r\n", b".\n"):
                        break
                    lines.append(ln[1:] if ln.startswith(b"..") else ln)
                outbox = Path(self.server.outbox)
                outbox.mkdir(parents=True, exist_ok=True)
                name = f"{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 100000:05d}.eml"
                (outbox / name).write_bytes(b"".join(lines))
                self._send(f"250 OK queued as {name}")
            elif verb == "RSET":
                mail_from, rcpts = None, []
                self._send("250 OK")
            elif verb == "NOOP":
                self._send("250 OK")
            elif verb == "QUIT":
                self._send("221 Bye")
                return
            else:
                self._send("502 Command not implemented")


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start(host=None, port=None, outbox=None):
    """Start the sink in a daemon thread. Returns the server, or None if the port is taken
    (typically because another sink is already running, which is fine)."""
    try:
        srv = _Server((host or config.SMTP_HOST, port or config.SMTP_PORT), _Handler)
    except OSError:
        return None
    srv.outbox = str(outbox or config.OUTBOX_DIR)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    s = start()
    print(f"mailsink listening on {config.SMTP_HOST}:{config.SMTP_PORT}, writing to {config.OUTBOX_DIR}" if s else "port busy")
    while s:
        time.sleep(3600)
