"""Immutable full-run results plus append-only checkpoints, keyed by entrant/round."""

import json

from .retrieval.archive import ArchiveStore
from .retrieval.errors import ArchiveConflict, ArchiveCorruption
from .retrieval.utils import canonical_json, digest, iso, utcnow


class RunArchive(ArchiveStore):
    def __init__(self, path):
        super().__init__(path)
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS forecast_runs(
                    entrant TEXT NOT NULL, round_id TEXT NOT NULL,
                    input_hash TEXT NOT NULL, config_hash TEXT NOT NULL,
                    payload TEXT, payload_hash TEXT,
                    PRIMARY KEY(entrant,round_id));
                CREATE TABLE IF NOT EXISTS run_events(
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entrant TEXT NOT NULL, round_id TEXT NOT NULL,
                    stage TEXT NOT NULL, payload TEXT NOT NULL, payload_hash TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS request_bindings(
                    entrant TEXT NOT NULL, request_id TEXT NOT NULL, payload_hash TEXT NOT NULL,
                    PRIMARY KEY(entrant,request_id));
            """
            )

    def bind_request(self, entrant, request_id, envelope_hash):
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO request_bindings VALUES (?,?,?)",
                (entrant, request_id, envelope_hash),
            )
            current = db.execute(
                "SELECT payload_hash FROM request_bindings WHERE entrant=? AND request_id=?",
                (entrant, request_id),
            ).fetchone()[0]
            if current != envelope_hash:
                raise ArchiveConflict("request_id was reused with a different envelope")

    def begin(self, entrant, round_id, input_hash, config_hash):
        with self._db() as db:
            db.execute(
                "INSERT OR IGNORE INTO forecast_runs VALUES (?,?,?,?,NULL,NULL)",
                (entrant, round_id, input_hash, config_hash),
            )
        return self.load(entrant, round_id, input_hash, config_hash)

    def load(self, entrant, round_id, input_hash, config_hash):
        with self._db() as db:
            row = db.execute(
                "SELECT input_hash,config_hash,payload,payload_hash FROM forecast_runs WHERE entrant=? AND round_id=?",
                (entrant, round_id),
            ).fetchone()
        if row is None:
            return None
        if row[:2] != (input_hash, config_hash):
            raise ArchiveConflict(
                "round has different request/configuration; choose another experiment entrant"
            )
        if row[2] is None:
            return None
        try:
            result = json.loads(row[2])
            if digest(result) != row[3]:
                raise ArchiveCorruption("full-run result hash mismatch")
            return result
        except (ValueError, TypeError) as exc:
            raise ArchiveCorruption("invalid run archive JSON") from exc

    def event(self, entrant, round_id, stage, payload):
        value = {"at": iso(utcnow()), "data": payload}
        with self._db() as db:
            db.execute(
                "INSERT INTO run_events(entrant,round_id,stage,payload,payload_hash) VALUES (?,?,?,?,?)",
                (entrant, round_id, stage, canonical_json(value), digest(value)),
            )

    def events(self, entrant, round_id):
        with self._db() as db:
            rows = db.execute(
                "SELECT stage,payload,payload_hash FROM run_events WHERE entrant=? AND round_id=? ORDER BY id",
                (entrant, round_id),
            ).fetchall()
        result = []
        for stage, payload, hash_value in rows:
            value = json.loads(payload)
            if digest(value) != hash_value:
                raise ArchiveCorruption("run event hash mismatch")
            result.append({"stage": stage, **value})
        return result

    def save(self, entrant, round_id, input_hash, config_hash, payload):
        existing = self.load(entrant, round_id, input_hash, config_hash)
        if existing is not None:
            if digest(existing) != digest(payload):
                raise ArchiveConflict("cannot overwrite full-run result")
            return
        with self._db() as db:
            changed = db.execute(
                "UPDATE forecast_runs SET payload=?,payload_hash=? WHERE entrant=? AND round_id=? AND input_hash=? AND config_hash=? AND payload IS NULL",
                (
                    canonical_json(payload),
                    digest(payload),
                    entrant,
                    round_id,
                    input_hash,
                    config_hash,
                ),
            ).rowcount
            if changed != 1:
                raise ArchiveConflict(
                    "run result changed concurrently or run not begun"
                )
