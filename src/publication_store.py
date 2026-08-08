"""Durable approval, OAuth connection metadata, and publication records."""

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ApprovalRecord:
    id: str
    clip_path: str
    clip_sha256: str
    approved_by: str
    note: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AccountRecord:
    id: str
    provider: str
    external_account_id: str
    display_name: str
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublicationRecord:
    id: str
    provider: str
    platform: str
    mode: str
    state: str
    clip_path: str
    clip_sha256: str
    approval_id: str
    account_id: Optional[str]
    title: str
    caption: str
    hashtags: List[str]
    privacy: str
    scheduled_at: Optional[str]
    external_id: Optional[str]
    external_url: Optional[str]
    provider_payload: Optional[Dict[str, Any]]
    error: Optional[str]
    idempotency_key: str
    created_at: str
    updated_at: str
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PublicationStore:
    """SQLite store containing no plaintext OAuth tokens."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = Path(
            database_path or os.getenv('PUBLISH_DATABASE_PATH', './.opuslabs/publishing.sqlite3')
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat(timespec='milliseconds')

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA busy_timeout = 30000')
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        with self._connection() as connection:
            connection.execute('PRAGMA journal_mode = WAL')
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS publish_approvals (
                    id TEXT PRIMARY KEY,
                    clip_path TEXT NOT NULL,
                    clip_sha256 TEXT NOT NULL,
                    approved_by TEXT NOT NULL,
                    note TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS approvals_clip_idx
                    ON publish_approvals(clip_sha256, created_at);

                CREATE TABLE IF NOT EXISTS publish_accounts (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    external_account_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(provider, external_account_id)
                );

                CREATE TABLE IF NOT EXISTS oauth_sessions (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    state_hash TEXT NOT NULL,
                    redirect_uri TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS publications (
                    id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    mode TEXT NOT NULL CHECK(mode IN ('draft','publish','schedule')),
                    state TEXT NOT NULL CHECK(state IN ('submitting','drafted','published','scheduled','failed')),
                    clip_path TEXT NOT NULL,
                    clip_sha256 TEXT NOT NULL,
                    approval_id TEXT NOT NULL REFERENCES publish_approvals(id) ON DELETE RESTRICT,
                    account_id TEXT REFERENCES publish_accounts(id) ON DELETE SET NULL,
                    title TEXT NOT NULL,
                    caption TEXT NOT NULL,
                    hashtags_json TEXT NOT NULL,
                    privacy TEXT NOT NULL,
                    scheduled_at TEXT,
                    external_id TEXT,
                    external_url TEXT,
                    provider_payload_json TEXT,
                    error TEXT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    completed_at TEXT
                );
                CREATE INDEX IF NOT EXISTS publications_state_idx
                    ON publications(state, created_at);
                """
            )

    @staticmethod
    def fingerprint_file(path: str) -> str:
        digest = hashlib.sha256()
        with Path(path).open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def create_approval(self, clip_path: str, approved_by: str, note: str = '') -> ApprovalRecord:
        resolved_clip = Path(clip_path).expanduser().resolve()
        if not resolved_clip.is_file() or resolved_clip.stat().st_size <= 0:
            raise ValueError(f'Approval requires a non-empty clip file: {resolved_clip}')
        approver = re.sub(r'\s+', ' ', str(approved_by or '').strip())
        if not approver or len(approver) > 160:
            raise ValueError('approved_by must contain 1 to 160 characters.')
        approval_id = uuid.uuid4().hex
        clip_hash = self.fingerprint_file(str(resolved_clip))
        timestamp = self._now()
        normalized_note = str(note or '').strip()[:1000]
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO publish_approvals(
                    id, clip_path, clip_sha256, approved_by, note, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    str(resolved_clip),
                    clip_hash,
                    approver,
                    normalized_note,
                    timestamp,
                ),
            )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> Optional[ApprovalRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM publish_approvals WHERE id = ?',
                (approval_id,),
            ).fetchone()
        return self._approval_from_row(row) if row else None

    def upsert_account(
        self,
        provider: str,
        external_account_id: str,
        display_name: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AccountRecord:
        provider_name = str(provider or '').strip().lower()
        external_id = str(external_account_id or '').strip()
        if not provider_name or not external_id:
            raise ValueError('Provider and external account id are required.')
        unsafe_metadata_keys = [
            str(key)
            for key in (metadata or {})
            if any(secret_name in str(key).lower() for secret_name in ('token', 'secret', 'password'))
        ]
        if unsafe_metadata_keys:
            raise ValueError(
                'Account metadata cannot contain credential fields: '
                + ', '.join(sorted(unsafe_metadata_keys))
            )
        timestamp = self._now()
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            existing = connection.execute(
                'SELECT id, created_at FROM publish_accounts WHERE provider = ? AND external_account_id = ?',
                (provider_name, external_id),
            ).fetchone()
            if existing:
                account_id = existing['id']
                connection.execute(
                    """
                    UPDATE publish_accounts SET display_name = ?, metadata_json = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (str(display_name or external_id)[:200], metadata_json, timestamp, account_id),
                )
            else:
                account_id = uuid.uuid4().hex
                connection.execute(
                    """
                    INSERT INTO publish_accounts(
                        id, provider, external_account_id, display_name,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        account_id,
                        provider_name,
                        external_id,
                        str(display_name or external_id)[:200],
                        metadata_json,
                        timestamp,
                        timestamp,
                    ),
                )
        return self.get_account(account_id)

    def get_account(self, account_id: str) -> Optional[AccountRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM publish_accounts WHERE id = ?',
                (account_id,),
            ).fetchone()
        return self._account_from_row(row) if row else None

    def list_accounts(self, provider: Optional[str] = None) -> List[AccountRecord]:
        query = 'SELECT * FROM publish_accounts'
        parameters: List[Any] = []
        if provider:
            query += ' WHERE provider = ?'
            parameters.append(provider.strip().lower())
        query += ' ORDER BY provider, display_name, created_at'
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._account_from_row(row) for row in rows]

    def delete_account(self, account_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                'DELETE FROM publish_accounts WHERE id = ?',
                (account_id,),
            )
        return cursor.rowcount == 1

    def create_oauth_session(
        self,
        provider: str,
        state_hash: str,
        redirect_uri: str,
        expires_at: str,
    ) -> str:
        session_id = uuid.uuid4().hex
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO oauth_sessions(
                    id, provider, state_hash, redirect_uri, expires_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    provider.strip().lower(),
                    state_hash,
                    redirect_uri,
                    expires_at,
                    self._now(),
                ),
            )
        return session_id

    def get_oauth_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM oauth_sessions WHERE id = ?',
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def delete_oauth_session(self, session_id: str) -> None:
        with self._connection() as connection:
            connection.execute('DELETE FROM oauth_sessions WHERE id = ?', (session_id,))

    def find_by_idempotency_key(self, key: str) -> Optional[PublicationRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM publications WHERE idempotency_key = ?',
                (key,),
            ).fetchone()
        return self._publication_from_row(row) if row else None

    def create_publication(self, values: Dict[str, Any]) -> PublicationRecord:
        publication_id = uuid.uuid4().hex
        timestamp = self._now()
        with self._connection() as connection:
            connection.execute(
                """
                INSERT INTO publications(
                    id, provider, platform, mode, state, clip_path, clip_sha256,
                    approval_id, account_id, title, caption, hashtags_json,
                    privacy, scheduled_at, idempotency_key, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'submitting', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    publication_id,
                    values['provider'],
                    values['platform'],
                    values['mode'],
                    values['clip_path'],
                    values['clip_sha256'],
                    values['approval_id'],
                    values.get('account_id'),
                    values['title'],
                    values.get('caption', ''),
                    json.dumps(values.get('hashtags', []), ensure_ascii=False),
                    values['privacy'],
                    values.get('scheduled_at'),
                    values['idempotency_key'],
                    timestamp,
                    timestamp,
                ),
            )
        return self.get_publication(publication_id)

    def finish_publication(
        self,
        publication_id: str,
        state: str,
        external_id: Optional[str],
        external_url: Optional[str],
        provider_payload: Optional[Dict[str, Any]],
    ) -> PublicationRecord:
        if state not in {'drafted', 'published', 'scheduled'}:
            raise ValueError(f'Invalid successful publication state: {state}')
        timestamp = self._now()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE publications SET state = ?, external_id = ?, external_url = ?,
                    provider_payload_json = ?, error = NULL, updated_at = ?, completed_at = ?
                WHERE id = ?
                """,
                (
                    state,
                    external_id,
                    external_url,
                    json.dumps(provider_payload or {}, ensure_ascii=False),
                    timestamp,
                    timestamp,
                    publication_id,
                ),
            )
        return self.get_publication(publication_id)

    def fail_publication(self, publication_id: str, error: str) -> PublicationRecord:
        timestamp = self._now()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE publications SET state = 'failed', error = ?,
                    updated_at = ?, completed_at = ? WHERE id = ?
                """,
                (str(error or 'Unknown publishing error')[:4000], timestamp, timestamp, publication_id),
            )
        return self.get_publication(publication_id)

    def get_publication(self, publication_id: str) -> Optional[PublicationRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM publications WHERE id = ?',
                (publication_id,),
            ).fetchone()
        return self._publication_from_row(row) if row else None

    def list_publications(self, state: Optional[str] = None) -> List[PublicationRecord]:
        query = 'SELECT * FROM publications'
        parameters: List[Any] = []
        if state:
            query += ' WHERE state = ?'
            parameters.append(state)
        query += ' ORDER BY created_at DESC, id DESC'
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._publication_from_row(row) for row in rows]

    @staticmethod
    def _approval_from_row(row: sqlite3.Row) -> ApprovalRecord:
        return ApprovalRecord(
            id=row['id'],
            clip_path=row['clip_path'],
            clip_sha256=row['clip_sha256'],
            approved_by=row['approved_by'],
            note=row['note'],
            created_at=row['created_at'],
        )

    @staticmethod
    def _account_from_row(row: sqlite3.Row) -> AccountRecord:
        return AccountRecord(
            id=row['id'],
            provider=row['provider'],
            external_account_id=row['external_account_id'],
            display_name=row['display_name'],
            metadata=json.loads(row['metadata_json']),
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    @staticmethod
    def _publication_from_row(row: sqlite3.Row) -> PublicationRecord:
        return PublicationRecord(
            id=row['id'],
            provider=row['provider'],
            platform=row['platform'],
            mode=row['mode'],
            state=row['state'],
            clip_path=row['clip_path'],
            clip_sha256=row['clip_sha256'],
            approval_id=row['approval_id'],
            account_id=row['account_id'],
            title=row['title'],
            caption=row['caption'],
            hashtags=json.loads(row['hashtags_json']),
            privacy=row['privacy'],
            scheduled_at=row['scheduled_at'],
            external_id=row['external_id'],
            external_url=row['external_url'],
            provider_payload=(
                json.loads(row['provider_payload_json'])
                if row['provider_payload_json']
                else None
            ),
            error=row['error'],
            idempotency_key=row['idempotency_key'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            completed_at=row['completed_at'],
        )
