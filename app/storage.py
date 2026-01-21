from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import JobStatus


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class Paths:
    base: Path

    @property
    def uploads(self) -> Path:
        return self.base / "uploads"

    @property
    def results(self) -> Path:
        return self.base / "results"

    @property
    def db(self) -> Path:
        return self.base / "jobs.sqlite3"


async def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
              job_id TEXT PRIMARY KEY,
              status TEXT NOT NULL,
              stage TEXT,
              progress INTEGER NOT NULL DEFAULT 0,
              error TEXT,
              debug_enabled INTEGER NOT NULL DEFAULT 0,
              debug_paragraph_enabled INTEGER NOT NULL DEFAULT 0,
              debug_window_enabled INTEGER NOT NULL DEFAULT 0,
              filename TEXT,
              upload_path TEXT,
              result_path TEXT,
              page_count INTEGER,
              paragraph_count INTEGER,
              created_at TEXT NOT NULL,
              started_at TEXT,
              updated_at TEXT NOT NULL
            )
            """
        )
        # Best-effort migration for older DBs (SQLite supports ADD COLUMN).
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN debug_enabled INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN debug_paragraph_enabled INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        try:
            await db.execute("ALTER TABLE jobs ADD COLUMN debug_window_enabled INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass
        await db.execute("CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at)")
        await db.commit()


async def create_job(
    db_path: Path,
    job_id: str,
    filename: str,
    upload_path: Path,
    *,
    debug_paragraph_enabled: bool = False,
    debug_window_enabled: bool = False,
) -> None:
    now = utcnow().isoformat()
    debug_enabled = bool(debug_paragraph_enabled or debug_window_enabled)
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT INTO jobs(
              job_id, status, stage, progress, error,
              debug_enabled, debug_paragraph_enabled, debug_window_enabled,
              filename, upload_path, created_at, updated_at
            )
            VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                JobStatus.queued.value,
                "upload",
                0,
                None,
                int(debug_enabled),
                int(bool(debug_paragraph_enabled)),
                int(bool(debug_window_enabled)),
                filename,
                str(upload_path),
                now,
                now,
            ),
        )
        await db.commit()


async def update_job(
    db_path: Path,
    job_id: str,
    *,
    status: Optional[JobStatus] = None,
    stage: Optional[str] = None,
    progress: Optional[int] = None,
    error: Optional[str] = None,
    started_at: Optional[datetime] = None,
    page_count: Optional[int] = None,
    paragraph_count: Optional[int] = None,
    result_path: Optional[Path] = None,
) -> None:
    fields = []
    values = []
    if status is not None:
        fields.append("status=?")
        values.append(status.value)
    if stage is not None:
        fields.append("stage=?")
        values.append(stage)
    if progress is not None:
        fields.append("progress=?")
        values.append(int(progress))
    if error is not None:
        fields.append("error=?")
        values.append(error)
    if started_at is not None:
        fields.append("started_at=?")
        values.append(started_at.isoformat())
    if page_count is not None:
        fields.append("page_count=?")
        values.append(int(page_count))
    if paragraph_count is not None:
        fields.append("paragraph_count=?")
        values.append(int(paragraph_count))
    if result_path is not None:
        fields.append("result_path=?")
        values.append(str(result_path))
    fields.append("updated_at=?")
    values.append(utcnow().isoformat())

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            f"UPDATE jobs SET {', '.join(fields)} WHERE job_id=?",
            (*values, job_id),
        )
        await db.commit()


async def get_job(db_path: Path, job_id: str) -> Optional[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,))
        row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)


async def list_expired_jobs(db_path: Path, older_than: datetime) -> list[dict]:
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM jobs WHERE updated_at < ?", (older_than.isoformat(),))
        rows = await cur.fetchall()
        return [dict(r) for r in rows]


async def delete_job(db_path: Path, job_id: str) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
        await db.commit()


def ensure_dirs(paths: Paths) -> None:
    paths.base.mkdir(parents=True, exist_ok=True)
    paths.uploads.mkdir(parents=True, exist_ok=True)
    paths.results.mkdir(parents=True, exist_ok=True)


def job_upload_path(paths: Paths, job_id: str, filename: str) -> Path:
    safe = "".join(ch for ch in filename if ch.isalnum() or ch in ("-", "_", ".", " ")).strip() or "upload.pdf"
    return paths.uploads / f"{job_id}__{safe}"


def job_result_path(paths: Paths, job_id: str) -> Path:
    return paths.results / f"{job_id}.json"


def cleanup_files_and_db(paths: Paths, ttl_hours: int) -> None:
    """Best-effort cleanup run at startup (demo)."""
    cutoff = utcnow() - timedelta(hours=ttl_hours)
    cutoff_ts = cutoff.timestamp()
    # Remove stale files
    for folder in (paths.uploads, paths.results):
        if not folder.exists():
            continue
        for p in folder.iterdir():
            try:
                if p.is_file() and p.stat().st_mtime < cutoff_ts:
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    # Remove stale DB rows (sync cleanup is OK for demo)
    if paths.db.exists():
        con = sqlite3.connect(paths.db)
        try:
            con.execute("DELETE FROM jobs WHERE updated_at < ?", (cutoff.isoformat(),))
            con.commit()
        finally:
            con.close()

