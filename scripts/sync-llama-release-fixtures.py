"""Serve fixture llama.cpp releases for test-sync-llama-release.sh.

Each case has its own path prefix:

    API:       http://127.0.0.1:<port>/<case>/releases/...
    Downloads: http://127.0.0.1:<port>/<prefix>/download/<tag>/<name>

A case named "a+b" serves its downloads from prefix "a",
so that its asset URLs equal those of case "a".
The server listens on a free port and writes the port to <port-file>.

Usage: python3 -I sync-llama-release-fixtures.py <archive> <port-file>
"""

import hashlib
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARCHIVE = sys.argv[1]
PORT_FILE = sys.argv[2]
with open(ARCHIVE, "rb") as f:
    ARCHIVE_BYTES = f.read()
DIGEST = "sha256:" + hashlib.sha256(ARCHIVE_BYTES).hexdigest()
VERSION = "v1.0.0"
NIGHTLY = "b12359"
PUBLISHED = "2026-11-13T10:20:19Z"
BASE = ""  # Set when the server has a port.


def download(case, tag, name):
    return f"{BASE}/{case.split('+')[0]}/download/{tag}/{name}"


def recent():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 120))


def latest(case, tag=VERSION, txt=True, published=PUBLISHED, extra=()):
    assets = []
    if txt:
        assets.append({"name": "nightly-tag.txt", "browser_download_url": download(case, tag, "nightly-tag.txt")})
    return {"tag_name": tag, "published_at": published, "assets": assets + list(extra)}


def xcframework(case, tag=NIGHTLY, **overrides):
    name = f"llama-{tag}-xcframework.zip"
    asset = {"name": name, "browser_download_url": download(case, tag, name), "digest": DIGEST}
    asset.update(overrides)
    return {k: v for k, v in asset.items() if v is not None}


def nightly(case, tag=NIGHTLY, assets=None):
    return {"tag_name": tag, "assets": [xcframework(case, tag)] if assets is None else assets}


def ok(body):
    return lambda case, tag: (200, {}, body(case, tag))


RATE_LIMIT = (403, {"x-ratelimit-remaining": "0"}, {"message": "API rate limit exceeded for 127.0.0.1."})
OTHER_URL = "https://example.com/"

# Each case can override "latest" and "nightly", functions of (case, tag)
# that return (status, headers, body); "txt", the nightly-tag.txt body
# or a (status, body) pair; and "archive", the archive's (status, body).
CASES = {
    "ok": {},
    "ok+no-digest": {"nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest=None)]))},
    "download-503": {
        "nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest=None)])),
        "archive": (503, "Service Unavailable"),
    },
    "not-json": {"latest": lambda c, t: (200, {}, "<html>oops</html>")},
    "no-tag-name": {"latest": ok(lambda c, t: {"assets": []})},
    "no-assets": {"latest": ok(lambda c, t: {"tag_name": VERSION, "published_at": PUBLISHED})},
    "not-semver": {"latest": ok(lambda c, t: latest(c, tag="b12400"))},
    "prerelease-semver": {"latest": ok(lambda c, t: latest(c, tag="v1.1.0-rc1"))},
    "no-nightly-txt-old": {"latest": ok(lambda c, t: latest(c, txt=False))},
    "no-nightly-txt-new": {"latest": ok(lambda c, t: latest(c, txt=False, published=recent()))},
    "duplicate-nightly-txt": {
        "latest": ok(lambda c, t: latest(c, extra=[{"name": "nightly-tag.txt", "browser_download_url": download(c, VERSION, "nightly-tag.txt")}]))
    },
    "wrong-nightly-txt-url": {
        "latest": ok(lambda c, t: latest(c, txt=False, extra=[{"name": "nightly-tag.txt", "browser_download_url": OTHER_URL + "nightly-tag.txt"}]))
    },
    "nightly-txt-url-not-string": {
        "latest": ok(lambda c, t: latest(c, txt=False, extra=[{"name": "nightly-tag.txt", "browser_download_url": ["x"]}]))
    },
    "crlf-nightly-txt": {"txt": "b12359\r\n"},
    "no-newline-nightly-txt": {"txt": "b12359"},
    "prefixed-nightly-txt": {"txt": "master-b12359-0123abc\n"},
    "multiline-nightly-txt": {"txt": "b12\n359\n"},
    "extra-line-nightly-txt": {"txt": "b12359\n\n"},
    "trailing-space-nightly-txt": {"txt": "b12359 \n"},
    "nul-nightly-txt": {"txt": "b12359\x00\n"},
    "long-nightly-txt": {"txt": "b" + "1" * 300 + "\n"},
    "malformed-nightly-txt": {"txt": "b12359; rm -rf /\n"},
    "empty-nightly-txt": {"txt": "\n"},
    "nightly-txt-404": {"txt": (404, "Not Found")},
    "nightly-not-json": {"nightly": lambda c, t: (200, {}, "[]")},
    "no-xcframework": {
        "nightly": ok(lambda c, t: nightly(c, t, [{"name": f"llama-{t}-bin-macos-arm64.tar.gz", "browser_download_url": download(c, t, "x.tar.gz")}]))
    },
    "duplicate-xcframework": {
        "nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t), xcframework(c, t, digest="sha256:" + "0" * 64)]))
    },
    "wrong-xcframework-url": {
        "nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, browser_download_url=OTHER_URL + f"llama-{t}-xcframework.zip")]))
    },
    "digest-not-string": {"nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest=12345)]))},
    "digest-wrong-algorithm": {"nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest="md5:" + "0" * 32)]))},
    "digest-not-hex": {"nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest="sha256:" + "Z" * 64)]))},
    "nightly-tag-name-mismatch": {"nightly": ok(lambda c, t: nightly(c, "b99999", [xcframework(c, t)]))},
    "nightly-release-404": {"nightly": lambda c, t: (404, {}, {"message": "Not Found"})},
    "digest-mismatch": {"nightly": ok(lambda c, t: nightly(c, t, [xcframework(c, t, digest="sha256:" + "0" * 64)]))},
    "older-than-newest": {"latest": ok(lambda c, t: latest(c, tag="v0.9.0")), "txt": "b12000\n"},
    "rate-limit-403": {"latest": lambda c, t: RATE_LIMIT},
    "rate-limit-429": {"latest": lambda c, t: (429, {"retry-after": "60"}, {"message": "Too many requests"})},
    "forbidden-403": {"latest": lambda c, t: (403, {}, {"message": "Resource not accessible by integration"})},
    "unauthorized-401": {"latest": lambda c, t: (401, {}, {"message": "Bad credentials"})},
    "server-error-502": {"latest": lambda c, t: (502, {}, "Bad Gateway")},
    "rate-limit-on-nightly": {"nightly": lambda c, t: RATE_LIMIT},
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, status, headers, body):
        if isinstance(body, bytes):
            data = body
        elif isinstance(body, str):
            data = body.encode()
        else:
            data = json.dumps(body).encode()
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        name, _, rest = self.path.strip("/").partition("/")
        case = CASES.get(name)
        if case is None:
            return self.send(404, {}, {"message": "Not Found"})
        if rest == "releases/latest":
            return self.send(*case.get("latest", ok(lambda c, t: latest(c)))(name, VERSION))
        if rest.startswith("releases/tags/"):
            tag = rest[len("releases/tags/"):]
            return self.send(*case.get("nightly", ok(nightly))(name, tag))
        if rest.startswith("download/") and rest.endswith("/nightly-tag.txt"):
            txt = case.get("txt", NIGHTLY + "\n")
            return self.send(txt[0], {}, txt[1]) if isinstance(txt, tuple) else self.send(200, {}, txt)
        if rest.startswith("download/") and rest.endswith("-xcframework.zip"):
            if "archive" in case:
                return self.send(case["archive"][0], {}, case["archive"][1])
            return self.send(200, {}, ARCHIVE_BYTES)
        return self.send(404, {}, "Not Found")


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
BASE = f"http://127.0.0.1:{server.server_address[1]}"
with open(PORT_FILE + ".tmp", "w") as f:
    f.write(str(server.server_address[1]))
# Rename, so that the test never reads a partly written file.
os.replace(PORT_FILE + ".tmp", PORT_FILE)
server.serve_forever()
