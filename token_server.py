"""
token_server.py — Simple LiveKit token server for the AdaptiveCX demo frontend.
Run with: python token_server.py
Then open frontend/index.html in your browser.
"""
import os, time, json, hmac, hashlib, base64, struct
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

API_KEY    = os.getenv("LIVEKIT_API_KEY", "")
API_SECRET = os.getenv("LIVEKIT_API_SECRET", "")


def b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def make_token(room: str, identity: str) -> str:
    """Mint a minimal LiveKit JWT token."""
    now = int(time.time())
    header = b64url(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    payload = b64url(json.dumps({
        "iss": API_KEY,
        "sub": identity,
        "iat": now,
        "exp": now + 3600,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": True,
            "canSubscribe": True,
        },
        "metadata": "",
        "name": identity,
    }).encode())
    sig_input = f"{header}.{payload}".encode()
    sig = hmac.new(API_SECRET.encode(), sig_input, hashlib.sha256).digest()
    return f"{header}.{payload}.{b64url(sig)}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[token-server] {fmt % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path == "/token":
            room     = qs.get("room",     ["adaptivecx-demo-room"])[0]
            identity = qs.get("identity", ["dashboard-user"])[0]

            token = make_token(room, identity)
            body  = json.dumps({"token": token}).encode()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.end_headers()


if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        print("ERROR: LIVEKIT_API_KEY / LIVEKIT_API_SECRET not set in .env")
        exit(1)
    print(f"Token server running on http://localhost:7881")
    print(f"Using key: {API_KEY}")
    print(f"Open frontend/index.html in your browser, then click Join.")
    HTTPServer(("localhost", 7881), Handler).serve_forever()
