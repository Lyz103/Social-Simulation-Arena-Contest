"""Offline CLI smoke tests, including safe exports and replay."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ssa_harness.retrieval.cli import EXAMPLES, main


class CLITests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.db = self.root / "archive.sqlite3"
        self.output = self.root / "bundle.json"

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with patch(
                "ssa_harness.retrieval.providers.TavilyRetriever.search",
                side_effect=AssertionError("Offline command attempted live search"),
            ):
                status = main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def run_args(self, entrant="cli-test"):
        return [
            "run",
            "--request",
            str(EXAMPLES / "topline.json"),
            "--provider",
            "snapshot",
            "--corpus",
            str(EXAMPLES / "corpus.json"),
            "--entrant",
            entrant,
            "--db",
            str(self.db),
            "--output",
            str(self.output),
        ]

    def test_all_demo_commands_replay_offline_with_unicode(self):
        for kind in ("topline", "profile", "ranking"):
            with self.subTest(kind=kind):
                args = ["demo", "--kind", kind, "--db", str(self.db)]
                first = self.invoke(args)
                self.assertEqual(first[0], 0, first[2])
                self.assertIn("以下为外部检索证据", first[1])
                self.assertIn("虚构演示资料", first[1])
                self.assertEqual(first, self.invoke(args))

    def test_export_verify_and_exact_replay(self):
        first = self.invoke(self.run_args())
        self.assertEqual(first[0], 0, first[2])
        original = self.output.read_bytes()
        exported = json.loads(original.decode("utf-8"))
        self.assertEqual(exported, json.loads(first[1]))
        self.assertIn("以下为外部检索证据", exported["rendered_text"])
        self.assertEqual(first, self.invoke(self.run_args()))
        self.assertEqual(original, self.output.read_bytes())
        status, output, error = self.invoke(["verify", str(self.output)])
        self.assertEqual(status, 0, error)
        self.assertIn(exported["bundle_hash"], output)

    def test_different_bundle_cannot_overwrite_export(self):
        self.assertEqual(self.invoke(self.run_args())[0], 0)
        original = self.output.read_bytes()
        status, _, error = self.invoke(self.run_args(entrant="other-experiment"))
        self.assertEqual(status, 2)
        self.assertIn("已存在不同 bundle", error)
        self.assertEqual(original, self.output.read_bytes())

    def test_verify_reports_tampering_without_traceback(self):
        self.assertEqual(self.invoke(self.run_args())[0], 0)
        value = json.loads(self.output.read_text(encoding="utf-8"))
        value["rendered_text"] = "tampered"
        self.output.write_text(json.dumps(value), encoding="utf-8")
        status, _, error = self.invoke(["verify", str(self.output)])
        self.assertEqual(status, 2)
        self.assertIn("ArchiveCorruption", error)
        self.assertNotIn("Traceback", error)


if __name__ == "__main__":
    unittest.main()
