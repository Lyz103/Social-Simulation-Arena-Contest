"""Run pinned official network probe against an ephemeral local fixture endpoint.

Fetches a public, pinned standalone probe; sends no forecast to the arena.
Public test signing keys are intentionally public and not credentials.
"""

from pathlib import Path
import importlib.util
import json
import tempfile
import threading
from urllib.request import urlopen

from ssa_harness.archive import RunArchive
from ssa_harness.pipeline import ForecastPipeline, PipelineConfig
from ssa_harness.server import make_server


COMMIT = "e80c8b42c157dadc018441da2a9691b44b20affc"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "tools/probe_agent_api.py"
        path.parent.mkdir()
        schema_path = Path(tmp) / "schema/agent-api-response.schema.json"
        schema_path.parent.mkdir()
        from importlib.resources import files

        schema_path.write_text(
            files("ssa_harness")
            .joinpath("schemas", "agent-api-response.schema.json")
            .read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        with urlopen(
            f"https://raw.githubusercontent.com/Social-Atoms/social-sim-arena/{COMMIT}/tools/probe_agent_api.py",
            timeout=30,
        ) as r:
            path.write_bytes(r.read())
        spec = importlib.util.spec_from_file_location("official_probe", path)
        probe = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(probe)
        with urlopen("https://social-simulation-arena.com/keys.json", timeout=30) as r:
            keys = {row["key_id"]: row["public_key"] for row in json.load(r)["keys"]}
        p = ForecastPipeline(
            archive=RunArchive(Path(tmp) / "runs.sqlite3"),
            clients={},
            config=PipelineConfig(
                research_policy="disabled", allow_baseline_fallback=True
            ),
        )
        server = make_server(p, port=0, keys=keys, enable_probe_fixtures=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            signer = probe.make_signer(probe.TEST_PRIVATE_KEY, probe.TEST_KEY_ID)
            if signer is None:
                raise RuntimeError("cryptography is required for signature tests")
            result = probe.probe(
                f"http://127.0.0.1:{server.server_port}/forecast",
                signer=signer,
                retries=0,
            )
            print(
                json.dumps(
                    {"commit": COMMIT, "non_scored_probe": True, "result": result},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            if not all(check[1] for check in result):
                raise AssertionError("official contract probe failed")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(5)


if __name__ == "__main__":
    main()
