"""Minimal litellm CLI stub.

Provides a `litellm` command that starts a simple HTTP server on the specified port.
This mimics the real litellm proxy's behavior enough to satisfy supervisor health checks.
"""

import argparse
import http.server
import json
import signal
import sys
import threading


class LiteLLMStubHandler(http.server.BaseHTTPRequestHandler):
    """Simple HTTP handler that responds to /health and basic routes."""

    def do_GET(self):
        if self.path == "/health":
            self._respond(200, {"status": "healthy", "stub": True})
        elif self.path == "/":
            self._respond(200, {"message": "litellm stub proxy running"})
        else:
            self._respond(404, {"error": "not found"})

    def do_POST(self):
        # Accept any POST and return a stub response
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        self._respond(200, {
            "id": "stub-response",
            "object": "chat.completion",
            "choices": [{"message": {"role": "assistant", "content": "stub response"}}],
        })

    def _respond(self, status_code: int, body: dict):
        response = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, format, *args):
        # Suppress default logging to keep stdout clean for supervisor
        sys.stdout.write(f"[litellm-stub] {format % args}\n")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="LiteLLM Stub Proxy")
    parser.add_argument("--config", type=str, default=None, help="Config file path (ignored in stub)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers (ignored in stub)")
    args = parser.parse_args()

    server = http.server.HTTPServer((args.host, args.port), LiteLLMStubHandler)

    # Graceful shutdown on SIGTERM/SIGINT
    def shutdown_handler(signum, frame):
        print(f"[litellm-stub] Received signal {signum}, shutting down...")
        threading.Thread(target=server.shutdown).start()

    signal.signal(signal.SIGTERM, shutdown_handler)
    signal.signal(signal.SIGINT, shutdown_handler)

    print(f"[litellm-stub] Proxy running on {args.host}:{args.port}")
    sys.stdout.flush()
    server.serve_forever()


if __name__ == "__main__":
    main()
