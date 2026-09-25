"""SQLite transactions for caches, immutable bundles and per-round leases."""

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid

from .errors import ArchiveConflict, ArchiveCorruption, WorkInProgress
from .models import ResearchBundle
from .utils import canonical_json, digest


class ArchiveStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS bundles (
                    round_id TEXT NOT NULL, entrant TEXT NOT NULL,
                    request_hash TEXT NOT NULL, pipeline_hash TEXT NOT NULL,
                    payload TEXT NOT NULL, PRIMARY KEY(round_id, entrant)
                );
                CREATE TABLE IF NOT EXISTS query_cache (
                    key TEXT PRIMARY KEY, created REAL NOT NULL,
                    payload TEXT NOT NULL, hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS leases (
                    round_id TEXT NOT NULL, entrant TEXT NOT NULL,
                    owner TEXT NOT NULL, expires REAL NOT NULL,
                    PRIMARY KEY(round_id, entrant)
                );
            """
            )

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(str(self.path), timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def load_bundle(self, round_id, entrant, request_hash, pipeline_hash):
        with self._db() as db:
            row = db.execute(
                "SELECT request_hash,pipeline_hash,payload FROM bundles WHERE round_id=? AND entrant=?",
                (round_id, entrant),
            ).fetchone()
        if row is None:
            return None
        if row[:2] != (request_hash, pipeline_hash):
            raise ArchiveConflict(
                "该 round/entrant 已冻结其他输入或版本；请使用新的实验 entrant ID"
            )
        try:
            result = ResearchBundle.from_dict(json.loads(row[2]))
            if (
                result.round_id,
                result.entrant,
                result.request_hash,
                result.pipeline_hash,
            ) != (round_id, entrant, request_hash, pipeline_hash):
                raise ArchiveCorruption("归档索引与内容不一致")
            return result
        except (ValueError, TypeError) as exc:
            raise ArchiveCorruption("归档 JSON 损坏") from exc

    def save_bundle(self, bundle):
        bundle.verify()
        try:
            with self._db() as db:
                db.execute(
                    "INSERT INTO bundles VALUES (?,?,?,?,?)",
                    (
                        bundle.round_id,
                        bundle.entrant,
                        bundle.request_hash,
                        bundle.pipeline_hash,
                        canonical_json(bundle.to_dict()),
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self.load_bundle(
                bundle.round_id,
                bundle.entrant,
                bundle.request_hash,
                bundle.pipeline_hash,
            )
            if existing.bundle_hash != bundle.bundle_hash:
                raise ArchiveConflict("不允许覆盖已经冻结的 bundle") from None

    def cache_get(self, key, ttl):
        with self._db() as db:
            row = db.execute(
                "SELECT created,payload,hash FROM query_cache WHERE key=?", (key,)
            ).fetchone()
        if row is None or not 0 <= time.time() - row[0] <= ttl:
            return None
        try:
            value = json.loads(row[1])
            if digest(value) != row[2] or not isinstance(value, list):
                raise ArchiveCorruption("查询缓存校验失败")
            return value
        except (ValueError, TypeError) as exc:
            raise ArchiveCorruption("查询缓存损坏") from exc

    def cache_put(self, key, results):
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO query_cache VALUES (?,?,?,?)",
                (
                    key,
                    time.time(),
                    canonical_json(results),
                    digest(results),
                ),
            )

    @contextmanager
    def claim(self, round_id, entrant, seconds):
        owner = uuid.uuid4().hex
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires < ?", (time.time(),))
            try:
                db.execute(
                    "INSERT INTO leases VALUES (?,?,?,?)",
                    (
                        round_id,
                        entrant,
                        owner,
                        time.time() + seconds,
                    ),
                )
            except sqlite3.IntegrityError:
                raise WorkInProgress(
                    "同一 round/entrant 正在检索，请稍后重试"
                ) from None
        try:
            yield
        finally:
            with self._db() as db:
                db.execute(
                    "DELETE FROM leases WHERE round_id=? AND entrant=? AND owner=?",
                    (round_id, entrant, owner),
                )
