"""Durable projects, isolated job workspaces, and an atomic processing queue."""

import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional


JOB_STATES = {'queued', 'running', 'completed', 'failed', 'cancelled'}
_CHECKPOINT_NAME = re.compile(r'^[a-z0-9][a-z0-9_-]{0,63}$')


class JobCancelled(RuntimeError):
    """Raised when a running job reaches a cooperative cancellation point."""


@dataclass(frozen=True)
class ProjectRecord:
    id: str
    name: str
    created_at: str
    updated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceRecord:
    id: str
    project_id: str
    source_path: str
    position: int
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class JobRecord:
    id: str
    project_id: str
    source_id: str
    source_path: str
    request_text: str
    preferences: Dict[str, Any]
    state: str
    progress: int
    stage: str
    attempts: int
    max_attempts: int
    cancel_requested: bool
    workspace_path: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    created_at: str
    updated_at: str
    started_at: Optional[str]
    completed_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class JobManager:
    """SQLite-backed project and job store safe for multiple worker processes."""

    def __init__(
        self,
        database_path: Optional[str] = None,
        workspace_root: Optional[str] = None,
    ):
        self.database_path = Path(
            database_path or os.getenv('JOB_DATABASE_PATH', './.opuslabs/jobs.sqlite3')
        ).resolve()
        self.workspace_root = Path(
            workspace_root or os.getenv('JOB_WORKSPACE_ROOT', './.opuslabs/workspaces')
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
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
        """Commit or roll back a short-lived connection, then always close it."""
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
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS project_sources (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_path TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(project_id, source_path)
                );

                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_id TEXT NOT NULL REFERENCES project_sources(id) ON DELETE RESTRICT,
                    request_text TEXT NOT NULL,
                    preferences_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('queued','running','completed','failed','cancelled')),
                    progress INTEGER NOT NULL DEFAULT 0 CHECK(progress BETWEEN 0 AND 100),
                    stage TEXT NOT NULL DEFAULT 'Queued',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL DEFAULT 3,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    workspace_path TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT
                );

                CREATE INDEX IF NOT EXISTS jobs_queue_idx
                    ON jobs(state, cancel_requested, created_at);
                CREATE INDEX IF NOT EXISTS jobs_project_idx
                    ON jobs(project_id, created_at);

                CREATE TABLE IF NOT EXISTS job_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    @staticmethod
    def _normalize_source(source_path: str) -> str:
        source = str(source_path or '').strip().strip('"\'')
        if not source:
            raise ValueError('A project source cannot be empty.')
        if source.startswith(('http://', 'https://')):
            return source
        return str(Path(source).expanduser().resolve())

    def _project_exists(self, connection: sqlite3.Connection, project_id: str) -> bool:
        return connection.execute(
            'SELECT 1 FROM projects WHERE id = ?',
            (project_id,),
        ).fetchone() is not None

    def _workspace_path(self, project_id: str, job_id: str) -> Path:
        if not re.fullmatch(r'[0-9a-f]{32}', project_id) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ValueError('Invalid project or job identifier.')
        workspace = (self.workspace_root / project_id / job_id).resolve()
        if self.workspace_root not in workspace.parents:
            raise ValueError('Job workspace escaped the configured workspace root.')
        return workspace

    @staticmethod
    def _prepare_workspace(workspace: Path) -> None:
        for directory_name in ('input', 'temp', 'output', 'checkpoints'):
            (workspace / directory_name).mkdir(parents=True, exist_ok=True)

    def create_project(self, name: str, sources: Optional[List[str]] = None) -> ProjectRecord:
        normalized_name = re.sub(r'\s+', ' ', str(name or '').strip())
        if not normalized_name or len(normalized_name) > 120:
            raise ValueError('Project names must contain 1 to 120 characters.')
        project_id = uuid.uuid4().hex
        timestamp = self._now()
        with self._connection() as connection:
            connection.execute(
                'INSERT INTO projects(id, name, created_at, updated_at) VALUES (?, ?, ?, ?)',
                (project_id, normalized_name, timestamp, timestamp),
            )
        project = ProjectRecord(project_id, normalized_name, timestamp, timestamp)
        for source in sources or []:
            self.add_source(project_id, source)
        return project

    def get_project(self, project_id: str) -> Optional[ProjectRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM projects WHERE id = ?',
                (project_id,),
            ).fetchone()
        return self._project_from_row(row) if row else None

    def list_projects(self) -> List[ProjectRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM projects ORDER BY updated_at DESC, created_at DESC'
            ).fetchall()
        return [self._project_from_row(row) for row in rows]

    def add_source(self, project_id: str, source_path: str) -> SourceRecord:
        normalized_source = self._normalize_source(source_path)
        source_id = uuid.uuid4().hex
        timestamp = self._now()
        with self._connection() as connection:
            if not self._project_exists(connection, project_id):
                raise ValueError(f'Project not found: {project_id}')
            position = connection.execute(
                'SELECT COALESCE(MAX(position), -1) + 1 FROM project_sources WHERE project_id = ?',
                (project_id,),
            ).fetchone()[0]
            try:
                connection.execute(
                    """
                    INSERT INTO project_sources(id, project_id, source_path, position, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (source_id, project_id, normalized_source, position, timestamp),
                )
            except sqlite3.IntegrityError as error:
                raise ValueError(f'Source is already part of this project: {normalized_source}') from error
            connection.execute(
                'UPDATE projects SET updated_at = ? WHERE id = ?',
                (timestamp, project_id),
            )
        return SourceRecord(source_id, project_id, normalized_source, position, timestamp)

    def list_sources(self, project_id: str) -> List[SourceRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM project_sources WHERE project_id = ? ORDER BY position, created_at',
                (project_id,),
            ).fetchall()
        return [self._source_from_row(row) for row in rows]

    def enqueue_project(
        self,
        project_id: str,
        request_text: str,
        preferences: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> List[JobRecord]:
        if not str(request_text or '').strip():
            raise ValueError('A queued job requires a processing request.')
        if not 1 <= int(max_attempts) <= 10:
            raise ValueError('max_attempts must be between 1 and 10.')
        sources = self.list_sources(project_id)
        if not sources:
            raise ValueError('Add at least one source before queueing a project.')

        jobs: List[JobRecord] = []
        for source in sources:
            jobs.append(
                self.enqueue_source(
                    project_id,
                    source.id,
                    request_text,
                    preferences or {},
                    max_attempts,
                )
            )
        return jobs

    def enqueue_source(
        self,
        project_id: str,
        source_id: str,
        request_text: str,
        preferences: Dict[str, Any],
        max_attempts: int = 3,
    ) -> JobRecord:
        if not 1 <= int(max_attempts) <= 10:
            raise ValueError('max_attempts must be between 1 and 10.')
        job_id = uuid.uuid4().hex
        workspace = self._workspace_path(project_id, job_id)
        timestamp = self._now()
        preferences_json = json.dumps(preferences, ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            source = connection.execute(
                'SELECT * FROM project_sources WHERE id = ? AND project_id = ?',
                (source_id, project_id),
            ).fetchone()
            if source is None:
                raise ValueError(f'Project source not found: {source_id}')
            self._prepare_workspace(workspace)
            connection.execute(
                """
                INSERT INTO jobs(
                    id, project_id, source_id, request_text, preferences_json,
                    state, progress, stage, attempts, max_attempts,
                    cancel_requested, workspace_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', 0, 'Queued', 0, ?, 0, ?, ?, ?)
                """,
                (
                    job_id,
                    project_id,
                    source_id,
                    request_text.strip(),
                    preferences_json,
                    int(max_attempts),
                    str(workspace),
                    timestamp,
                    timestamp,
                ),
            )
            self._add_event(connection, job_id, 'queued', 'Job added to the processing queue.')
            connection.execute(
                'UPDATE projects SET updated_at = ? WHERE id = ?',
                (timestamp, project_id),
            )
        return self.get_job(job_id)

    def claim_next_job(self, project_id: Optional[str] = None) -> Optional[JobRecord]:
        connection = self._connect()
        try:
            connection.execute('BEGIN IMMEDIATE')
            query = (
                """
                SELECT id FROM jobs
                WHERE state = 'queued' AND cancel_requested = 0 AND attempts < max_attempts
                """
            )
            parameters: List[Any] = []
            if project_id:
                query += ' AND project_id = ?'
                parameters.append(project_id)
            query += ' ORDER BY created_at, id LIMIT 1'
            row = connection.execute(query, parameters).fetchone()
            if row is None:
                connection.commit()
                return None
            timestamp = self._now()
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = 'running', attempts = attempts + 1,
                    stage = 'Starting', updated_at = ?,
                    started_at = COALESCE(started_at, ?), completed_at = NULL,
                    error = NULL
                WHERE id = ? AND state = 'queued' AND cancel_requested = 0
                """,
                (timestamp, timestamp, row['id']),
            )
            if cursor.rowcount != 1:
                connection.rollback()
                return None
            self._add_event(connection, row['id'], 'claimed', 'A worker claimed the job.')
            connection.commit()
        finally:
            connection.close()
        return self.get_job(row['id'])

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._connection() as connection:
            row = connection.execute(
                """
                SELECT jobs.*, project_sources.source_path
                FROM jobs JOIN project_sources ON project_sources.id = jobs.source_id
                WHERE jobs.id = ?
                """,
                (job_id,),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def list_jobs(
        self,
        project_id: Optional[str] = None,
        state: Optional[str] = None,
    ) -> List[JobRecord]:
        if state and state not in JOB_STATES:
            raise ValueError(f'Unknown job state: {state}')
        query = (
            """
            SELECT jobs.*, project_sources.source_path
            FROM jobs JOIN project_sources ON project_sources.id = jobs.source_id
            WHERE 1 = 1
            """
        )
        parameters: List[Any] = []
        if project_id:
            query += ' AND jobs.project_id = ?'
            parameters.append(project_id)
        if state:
            query += ' AND jobs.state = ?'
            parameters.append(state)
        query += ' ORDER BY jobs.created_at, jobs.id'
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._job_from_row(row) for row in rows]

    def update_progress(self, job_id: str, progress: int, stage: str) -> JobRecord:
        progress = min(99, max(0, int(progress)))
        normalized_stage = re.sub(r'\s+', ' ', str(stage or '').strip())[:160] or 'Processing'
        with self._connection() as connection:
            current = connection.execute(
                'SELECT progress, state, cancel_requested FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
            if current is None:
                raise ValueError(f'Job not found: {job_id}')
            if current['state'] != 'running':
                raise ValueError(f'Only running jobs can report progress; current state is {current["state"]}.')
            if current['cancel_requested']:
                raise JobCancelled(f'Job cancellation requested: {job_id}')
            progress = max(progress, int(current['progress']))
            timestamp = self._now()
            connection.execute(
                'UPDATE jobs SET progress = ?, stage = ?, updated_at = ? WHERE id = ?',
                (progress, normalized_stage, timestamp, job_id),
            )
            self._add_event(connection, job_id, 'progress', f'{progress}% — {normalized_stage}')
        return self.get_job(job_id)

    def check_cancelled(self, job_id: str) -> None:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT cancel_requested, state FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
        if row is None:
            raise ValueError(f'Job not found: {job_id}')
        if row['cancel_requested'] or row['state'] == 'cancelled':
            raise JobCancelled(f'Job cancellation requested: {job_id}')

    def request_cancel(self, job_id: str) -> JobRecord:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT state FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'Job not found: {job_id}')
            if row['state'] in {'completed', 'failed', 'cancelled'}:
                return self.get_job(job_id)
            timestamp = self._now()
            if row['state'] == 'queued':
                connection.execute(
                    """
                    UPDATE jobs SET state = 'cancelled', cancel_requested = 1,
                        stage = 'Cancelled', updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (timestamp, timestamp, job_id),
                )
            else:
                connection.execute(
                    'UPDATE jobs SET cancel_requested = 1, stage = ?, updated_at = ? WHERE id = ?',
                    ('Cancellation requested', timestamp, job_id),
                )
            self._add_event(connection, job_id, 'cancel_requested', 'Cancellation was requested.')
        return self.get_job(job_id)

    def mark_cancelled(self, job_id: str, message: str = 'Job cancelled.') -> JobRecord:
        timestamp = self._now()
        with self._connection() as connection:
            connection.execute(
                """
                UPDATE jobs SET state = 'cancelled', cancel_requested = 1,
                    stage = 'Cancelled', error = ?, updated_at = ?, completed_at = ?
                WHERE id = ? AND state != 'completed'
                """,
                (message, timestamp, timestamp, job_id),
            )
            self._add_event(connection, job_id, 'cancelled', message)
        return self.get_job(job_id)

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> JobRecord:
        timestamp = self._now()
        with self._connection() as connection:
            row = connection.execute(
                'SELECT state, cancel_requested FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'Job not found: {job_id}')
            if row['cancel_requested']:
                raise JobCancelled(f'Job cancellation requested: {job_id}')
            if row['state'] != 'running':
                raise ValueError(f'Only running jobs can complete; current state is {row["state"]}.')
            connection.execute(
                """
                UPDATE jobs SET state = 'completed', progress = 100,
                    stage = 'Completed', result_json = ?, error = NULL,
                    updated_at = ?, completed_at = ? WHERE id = ?
                """,
                (json.dumps(result, ensure_ascii=False), timestamp, timestamp, job_id),
            )
            self._add_event(connection, job_id, 'completed', 'Job completed successfully.')
        return self.get_job(job_id)

    def fail_job(self, job_id: str, error: str) -> JobRecord:
        normalized_error = str(error or 'Unknown processing failure').strip()[:4000]
        timestamp = self._now()
        with self._connection() as connection:
            row = connection.execute(
                'SELECT cancel_requested FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'Job not found: {job_id}')
            if row['cancel_requested']:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'cancelled', stage = 'Cancelled',
                        error = ?, updated_at = ?, completed_at = ? WHERE id = ?
                    """,
                    (normalized_error, timestamp, timestamp, job_id),
                )
                event_type = 'cancelled'
            else:
                connection.execute(
                    """
                    UPDATE jobs SET state = 'failed', stage = 'Failed',
                        error = ?, updated_at = ?, completed_at = ? WHERE id = ?
                    """,
                    (normalized_error, timestamp, timestamp, job_id),
                )
                event_type = 'failed'
            self._add_event(connection, job_id, event_type, normalized_error)
        return self.get_job(job_id)

    def retry_job(self, job_id: str) -> JobRecord:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT state, attempts, max_attempts FROM jobs WHERE id = ?',
                (job_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'Job not found: {job_id}')
            if row['state'] not in {'failed', 'cancelled'}:
                raise ValueError(f'Only failed or cancelled jobs can retry; current state is {row["state"]}.')
            if row['attempts'] >= row['max_attempts']:
                raise ValueError('Job has exhausted its configured retry attempts.')
            timestamp = self._now()
            connection.execute(
                """
                UPDATE jobs SET state = 'queued', cancel_requested = 0,
                    stage = 'Queued for retry', error = NULL, completed_at = NULL,
                    updated_at = ? WHERE id = ?
                """,
                (timestamp, job_id),
            )
            self._add_event(connection, job_id, 'retried', 'Job returned to the queue.')
        return self.get_job(job_id)

    def recover_interrupted_jobs(
        self,
        stale_after_seconds: Optional[int] = None,
        force: bool = False,
    ) -> Dict[str, int]:
        """Recover jobs whose worker heartbeat is stale, or all running jobs when forced."""
        recovered = 0
        failed = 0
        cancelled = 0
        stale_after = int(
            stale_after_seconds
            if stale_after_seconds is not None
            else os.getenv('JOB_STALE_AFTER_SECONDS', '3600')
        )
        if stale_after < 0:
            raise ValueError('stale_after_seconds cannot be negative.')
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=stale_after)).isoformat(
            timespec='milliseconds'
        )
        timestamp = self._now()
        with self._connection() as connection:
            if force:
                rows = connection.execute(
                    "SELECT id, attempts, max_attempts, cancel_requested FROM jobs WHERE state = 'running'"
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT id, attempts, max_attempts, cancel_requested
                    FROM jobs WHERE state = 'running' AND updated_at <= ?
                    """,
                    (cutoff,),
                ).fetchall()
            for row in rows:
                if row['cancel_requested']:
                    state, stage, event_type = 'cancelled', 'Cancelled during recovery', 'cancelled'
                    cancelled += 1
                elif row['attempts'] < row['max_attempts']:
                    state, stage, event_type = 'queued', 'Recovered after interruption', 'recovered'
                    recovered += 1
                else:
                    state, stage, event_type = 'failed', 'Retry limit reached', 'failed'
                    failed += 1
                completed_at = timestamp if state in {'failed', 'cancelled'} else None
                connection.execute(
                    """
                    UPDATE jobs SET state = ?, stage = ?, updated_at = ?, completed_at = ?
                    WHERE id = ?
                    """,
                    (state, stage, timestamp, completed_at, row['id']),
                )
                self._add_event(
                    connection,
                    row['id'],
                    event_type,
                    'Recovered persisted job state after an interrupted worker.',
                )
        return {'requeued': recovered, 'failed': failed, 'cancelled': cancelled}

    def workspace_directories(self, job_id: str) -> Dict[str, str]:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f'Job not found: {job_id}')
        workspace = Path(job.workspace_path).resolve()
        if self.workspace_root not in workspace.parents:
            raise ValueError('Persisted job workspace escaped the configured root.')
        self._prepare_workspace(workspace)
        return {
            'root': str(workspace),
            'input': str(workspace / 'input'),
            'temp': str(workspace / 'temp'),
            'output': str(workspace / 'output'),
            'checkpoints': str(workspace / 'checkpoints'),
        }

    def write_checkpoint(self, job_id: str, name: str, payload: Any) -> str:
        normalized_name = str(name or '').strip().lower()
        if not _CHECKPOINT_NAME.fullmatch(normalized_name):
            raise ValueError('Checkpoint names may contain lowercase letters, numbers, dashes, and underscores.')
        directories = self.workspace_directories(job_id)
        checkpoint_path = Path(directories['checkpoints']) / f'{normalized_name}.json'
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=checkpoint_path.parent,
                prefix=f'.{checkpoint_path.name}.',
                suffix='.tmp',
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                temporary_path = Path(handle.name)
            os.replace(temporary_path, checkpoint_path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        return str(checkpoint_path)

    def read_checkpoint(self, job_id: str, name: str) -> Optional[Any]:
        normalized_name = str(name or '').strip().lower()
        if not _CHECKPOINT_NAME.fullmatch(normalized_name):
            raise ValueError('Invalid checkpoint name.')
        checkpoint_path = Path(self.workspace_directories(job_id)['checkpoints']) / f'{normalized_name}.json'
        if not checkpoint_path.exists():
            return None
        try:
            return json.loads(checkpoint_path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'Could not read checkpoint {normalized_name}: {error}') from error

    def list_events(self, job_id: str) -> List[Dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT event_type, message, created_at FROM job_events WHERE job_id = ? ORDER BY id',
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _add_event(
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        message: str,
    ) -> None:
        connection.execute(
            'INSERT INTO job_events(job_id, event_type, message, created_at) VALUES (?, ?, ?, ?)',
            (job_id, event_type, str(message)[:4000], JobManager._now()),
        )

    @staticmethod
    def _project_from_row(row: sqlite3.Row) -> ProjectRecord:
        return ProjectRecord(
            id=row['id'],
            name=row['name'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    @staticmethod
    def _source_from_row(row: sqlite3.Row) -> SourceRecord:
        return SourceRecord(
            id=row['id'],
            project_id=row['project_id'],
            source_path=row['source_path'],
            position=int(row['position']),
            created_at=row['created_at'],
        )

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row['id'],
            project_id=row['project_id'],
            source_id=row['source_id'],
            source_path=row['source_path'],
            request_text=row['request_text'],
            preferences=json.loads(row['preferences_json']),
            state=row['state'],
            progress=int(row['progress']),
            stage=row['stage'],
            attempts=int(row['attempts']),
            max_attempts=int(row['max_attempts']),
            cancel_requested=bool(row['cancel_requested']),
            workspace_path=row['workspace_path'],
            result=json.loads(row['result_json']) if row['result_json'] else None,
            error=row['error'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            started_at=row['started_at'],
            completed_at=row['completed_at'],
        )
