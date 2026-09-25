"""Small HTTP adapter; deploy behind a TLS reverse proxy, never register HTTP."""

import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .contracts import ContractError, validate_schema
from .retrieval.errors import ArchiveConflict, WorkInProgress


def verify_signature(headers, raw, keys, now=None):
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        ts = headers["X-SSA-Timestamp"]
        if abs((time.time() if now is None else now) - int(ts)) > 300:
            raise ValueError("timestamp outside acceptance window")
        key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(keys[headers["X-SSA-Key-Id"]], validate=True)
        )
        key.verify(
            base64.b64decode(headers["X-SSA-Signature"], validate=True),
            ts.encode() + b"." + raw,
        )
    except Exception:
        raise ValueError("invalid request signature") from None


def probe_response(envelope):
    """Explicit non-scored official probe fixtures; never used for season rounds."""
    validate_schema(envelope, "agent-api-request.schema.json")
    r = envelope["round"]
    expected = {
        "ssa-contract-test": "continuous_normal",
        "ssa-contract-test-profile": "profile_energy",
        "ssa-contract-test-ranking": "ranking_list",
    }
    if r["round_id"] not in expected:
        return None
    if (
        r["target_type"] != expected[r["round_id"]]
        or r["lock_at"] != "2099-01-01T00:00:00Z"
        or not r["question"].startswith("Non-scored contract test:")
    ):
        raise ContractError("invalid probe fixture")
    if r["target_type"] == "continuous_normal":
        fc = {"mean": 50, "sd": 5}
    elif r["target_type"] == "profile_energy":
        fc = {"profile": {c: {"mean": 50, "sd": 5} for c in r["cells"]}}
    else:
        fc = {"ranking": r["ranking"]["items"]}
    result = {"schema_version": "ssa-agent-api-v2", "forecast": fc}
    validate_schema(result, "agent-api-response.schema.json")
    return result


def make_server(
    pipeline,
    *,
    host="127.0.0.1",
    port=8787,
    keys=None,
    verify=True,
    max_concurrent=4,
    metadata=None,
    metadata_by_round=None,
    enable_probe_fixtures=False,
):
    if verify and not keys:
        raise ValueError("signature verification requires public keys")
    gate = threading.BoundedSemaphore(max_concurrent)

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, format, *args):
            pass

        def send(self, status, payload):
            raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header(
                "Access-Control-Allow-Headers",
                "Content-Type, X-SSA-Key-Id, X-SSA-Timestamp, X-SSA-Signature",
            )
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.end_headers()
            self.wfile.write(raw)

        def do_OPTIONS(self):
            self.send(200, {})

        def do_GET(self):
            (
                self.send(200, {"status": "ok"})
                if self.path == "/health"
                else self.send(404, {"error": "not_found"})
            )

        def do_POST(self):
            if self.path != "/forecast":
                return self.send(404, {"error": "not_found"})
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self.send(400, {"error": "invalid_content_length"})
            if not 0 < n <= 1_000_000:
                return self.send(413, {"error": "request_size"})
            if not gate.acquire(blocking=False):
                return self.send(503, {"error": "busy"})
            try:
                raw = self.rfile.read(n)
                if len(raw) != n:
                    return self.send(400, {"error": "incomplete_body"})
                if verify:
                    try:
                        verify_signature(self.headers, raw, keys)
                    except ValueError:
                        return self.send(401, {"error": "invalid_signature"})
                envelope = json.loads(raw)
                if enable_probe_fixtures:
                    probe = probe_response(envelope)
                    if probe is not None:
                        from .retrieval.utils import digest

                        pipeline.archive.bind_request(
                            pipeline.config.entrant,
                            envelope["request_id"],
                            digest(envelope),
                        )
                        return self.send(200, probe)
                task_meta = {
                    **(metadata or {}),
                    **(metadata_by_round or {}).get(envelope["round"]["round_id"], {}),
                }
                result = pipeline.run_envelope(envelope, metadata=task_meta)
                self.send(200, result["response"])
            except ArchiveConflict:
                self.send(409, {"error": "request_conflict"})
            except WorkInProgress:
                self.send(503, {"error": "request_in_progress"})
            except (ContractError, ValueError, KeyError, TypeError):
                self.send(400, {"error": "invalid_request"})
            except Exception:
                # No remote response, secret, or plaintext forecast in failure body.
                self.send(503, {"error": "forecast_failed"})
            finally:
                gate.release()

    return ThreadingHTTPServer((host, port), Handler)
