"""Live check for PyJWT GHSA-42vr-xj54-vc7v against a running backend.

Sends a forged bearer token whose payload is JSON nested thousands of levels
deep and reports the HTTP status. A safe server answers 401 (invalid token);
a vulnerable one answers 500 (unhandled RecursionError in token validation).

Usage (backend running, e.g. `python -m uvicorn app.main:app`):

    python scripts/check_nested_jwt.py
    python scripts/check_nested_jwt.py --base-url http://localhost:8001 --depth 4000

Exit code: 0 if every request got 401, 1 otherwise. Standard library only.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import urllib.error
import urllib.request


def b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def nested_token(depth: int) -> str:
    header = b64url(json.dumps({"alg": "RS256", "typ": "JWT", "kid": "x"}).encode())
    payload = b64url(b"[" * depth + b"]" * depth)
    return f"{header}.{payload}.{b64url(b'forged-signature')}"


def status_for(url: str, token: str) -> int:
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status
    except urllib.error.HTTPError as exc:
        return exc.code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--depth", type=int, default=6000, help="nesting depth (default 6000)")
    args = parser.parse_args()
    url = f"{args.base_url.rstrip('/')}/assistants"

    cases = [
        ("plain garbage token (baseline)", "not.a.jwt"),
        (f"nested payload, depth {args.depth}", nested_token(args.depth)),
    ]
    all_ok = True
    for label, token in cases:
        try:
            code = status_for(url, token)
        except urllib.error.URLError as exc:
            print(f"Could not reach {url}: {exc.reason}. Is the backend running?")
            return 1
        ok = code == 401
        all_ok &= ok
        verdict = "OK (rejected cleanly)" if ok else "VULNERABLE" if code == 500 else "UNEXPECTED"
        print(f"{label:<34} token {len(token):>6} bytes -> HTTP {code}  {verdict}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
