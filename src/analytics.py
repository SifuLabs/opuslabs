"""Provider-neutral performance imports and evidence-backed clip reranking."""

import csv
import hashlib
import io
import json
import math
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def _identifier(value: Any, field: str, required: bool = True) -> str:
    text = re.sub(r'[^a-z0-9]+', '_', str(value or '').strip().lower()).strip('_')
    if required and not text:
        raise ValueError(f'{field} is required.')
    if len(text) > 80:
        raise ValueError(f'{field} cannot exceed 80 normalized characters.')
    return text


def _performance_score(
    views: int,
    engaged_views: int,
    retention_percent: float,
    shares: int,
    conversions: int,
) -> float:
    """Return a transparent 0-10 directional score for cohort comparisons."""
    if views <= 0:
        return 0.0
    engaged_rate = min(1.0, engaged_views / views)
    retention_rate = min(1.0, retention_percent / 100.0)
    share_signal = min(1.0, (shares / views) / 0.05)
    conversion_signal = min(1.0, (conversions / views) / 0.02)
    return round(
        10.0 * (
            0.35 * retention_rate
            + 0.35 * engaged_rate
            + 0.20 * share_signal
            + 0.10 * conversion_signal
        ),
        4,
    )


@dataclass(frozen=True)
class AnalyticsImportRecord:
    id: str
    source_name: str
    source_sha256: str
    record_count: int
    imported_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnalyticsImportResult:
    import_record: AnalyticsImportRecord
    already_imported: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.import_record.to_dict(),
            'already_imported': self.already_imported,
        }


@dataclass(frozen=True)
class PerformanceObservation:
    id: str
    import_id: str
    source_row: int
    publication_id: Optional[str]
    clip_path: Optional[str]
    clip_sha256: Optional[str]
    provider: str
    platform: str
    style: str
    segment_type: str
    keywords: List[str]
    predicted_engagement_score: Optional[float]
    views: int
    engaged_views: int
    retention_percent: float
    shares: int
    conversions: int
    observed_at: str
    imported_at: str

    @property
    def engaged_view_rate(self) -> float:
        return self.engaged_views / self.views if self.views else 0.0

    @property
    def share_rate(self) -> float:
        return self.shares / self.views if self.views else 0.0

    @property
    def conversion_rate(self) -> float:
        return self.conversions / self.views if self.views else 0.0

    @property
    def actual_performance_score(self) -> float:
        return _performance_score(
            self.views,
            self.engaged_views,
            self.retention_percent,
            self.shares,
            self.conversions,
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.update({
            'engaged_view_rate': round(self.engaged_view_rate, 6),
            'share_rate': round(self.share_rate, 6),
            'conversion_rate': round(self.conversion_rate, 6),
            'actual_performance_score': round(self.actual_performance_score, 2),
            'prediction_delta': (
                round(self.actual_performance_score - self.predicted_engagement_score, 2)
                if self.predicted_engagement_score is not None
                else None
            ),
        })
        return payload


@dataclass(frozen=True)
class _NormalizedObservation:
    source_row: int
    publication_id: Optional[str]
    clip_path: Optional[str]
    clip_sha256: Optional[str]
    provider: str
    platform: str
    style: str
    segment_type: str
    keywords: List[str]
    predicted_engagement_score: Optional[float]
    views: int
    engaged_views: int
    retention_percent: float
    shares: int
    conversions: int
    observed_at: str
    raw_payload: Dict[str, Any]


class AnalyticsStore:
    """WAL-enabled SQLite storage for immutable analytics imports."""

    def __init__(self, database_path: Optional[str] = None):
        self.database_path = Path(
            database_path or os.getenv(
                'ANALYTICS_DATABASE_PATH',
                './.opuslabs/analytics.sqlite3',
            )
        ).resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()

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
                CREATE TABLE IF NOT EXISTS analytics_imports (
                    id TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL UNIQUE,
                    record_count INTEGER NOT NULL CHECK(record_count >= 0),
                    imported_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS performance_observations (
                    id TEXT PRIMARY KEY,
                    import_id TEXT NOT NULL REFERENCES analytics_imports(id) ON DELETE RESTRICT,
                    source_row INTEGER NOT NULL CHECK(source_row > 0),
                    publication_id TEXT,
                    clip_path TEXT,
                    clip_sha256 TEXT,
                    provider TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    style TEXT NOT NULL,
                    segment_type TEXT NOT NULL,
                    keywords_json TEXT NOT NULL,
                    predicted_engagement_score REAL,
                    views INTEGER NOT NULL CHECK(views > 0),
                    engaged_views INTEGER NOT NULL CHECK(engaged_views >= 0),
                    retention_percent REAL NOT NULL CHECK(retention_percent BETWEEN 0 AND 100),
                    shares INTEGER NOT NULL CHECK(shares >= 0),
                    conversions INTEGER NOT NULL CHECK(conversions >= 0),
                    observed_at TEXT NOT NULL,
                    raw_json TEXT NOT NULL,
                    imported_at TEXT NOT NULL,
                    UNIQUE(import_id, source_row),
                    CHECK(publication_id IS NOT NULL OR clip_path IS NOT NULL OR clip_sha256 IS NOT NULL),
                    CHECK(predicted_engagement_score IS NULL OR predicted_engagement_score BETWEEN 0 AND 10),
                    CHECK(engaged_views <= views)
                );
                CREATE INDEX IF NOT EXISTS observations_platform_style_idx
                    ON performance_observations(platform, style, observed_at);
                CREATE INDEX IF NOT EXISTS observations_publication_idx
                    ON performance_observations(publication_id, observed_at);
                CREATE INDEX IF NOT EXISTS observations_clip_idx
                    ON performance_observations(clip_sha256, observed_at);
                """
            )

    def create_import(
        self,
        source_name: str,
        source_sha256: str,
        observations: Sequence[_NormalizedObservation],
    ) -> AnalyticsImportResult:
        with self._connection() as connection:
            existing = connection.execute(
                'SELECT * FROM analytics_imports WHERE source_sha256 = ?',
                (source_sha256,),
            ).fetchone()
            if existing:
                return AnalyticsImportResult(self._import_from_row(existing), True)

            import_id = uuid.uuid4().hex
            imported_at = _now()
            connection.execute(
                """
                INSERT INTO analytics_imports(
                    id, source_name, source_sha256, record_count, imported_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (import_id, source_name, source_sha256, len(observations), imported_at),
            )
            for observation in observations:
                connection.execute(
                    """
                    INSERT INTO performance_observations(
                        id, import_id, source_row, publication_id, clip_path,
                        clip_sha256, provider, platform, style, segment_type,
                        keywords_json, predicted_engagement_score, views,
                        engaged_views, retention_percent, shares, conversions,
                        observed_at, raw_json, imported_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        uuid.uuid4().hex,
                        import_id,
                        observation.source_row,
                        observation.publication_id,
                        observation.clip_path,
                        observation.clip_sha256,
                        observation.provider,
                        observation.platform,
                        observation.style,
                        observation.segment_type,
                        json.dumps(observation.keywords, ensure_ascii=False),
                        observation.predicted_engagement_score,
                        observation.views,
                        observation.engaged_views,
                        observation.retention_percent,
                        observation.shares,
                        observation.conversions,
                        observation.observed_at,
                        json.dumps(observation.raw_payload, ensure_ascii=False, sort_keys=True),
                        imported_at,
                    ),
                )
            row = connection.execute(
                'SELECT * FROM analytics_imports WHERE id = ?',
                (import_id,),
            ).fetchone()
        return AnalyticsImportResult(self._import_from_row(row), False)

    def list_imports(self) -> List[AnalyticsImportRecord]:
        with self._connection() as connection:
            rows = connection.execute(
                'SELECT * FROM analytics_imports ORDER BY imported_at DESC, id DESC'
            ).fetchall()
        return [self._import_from_row(row) for row in rows]

    def find_import_by_sha256(
        self,
        source_sha256: str,
    ) -> Optional[AnalyticsImportRecord]:
        with self._connection() as connection:
            row = connection.execute(
                'SELECT * FROM analytics_imports WHERE source_sha256 = ?',
                (source_sha256,),
            ).fetchone()
        return self._import_from_row(row) if row else None

    def list_observations(
        self,
        platform: Optional[str] = None,
        style: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[PerformanceObservation]:
        clauses: List[str] = []
        parameters: List[Any] = []
        if platform:
            clauses.append('platform = ?')
            parameters.append(_identifier(platform, 'platform'))
        if style:
            clauses.append('style = ?')
            parameters.append(_identifier(style, 'style'))
        query = 'SELECT * FROM performance_observations'
        if clauses:
            query += ' WHERE ' + ' AND '.join(clauses)
        query += ' ORDER BY observed_at DESC, source_row DESC'
        if limit is not None:
            if limit < 1:
                raise ValueError('limit must be at least 1.')
            query += ' LIMIT ?'
            parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._observation_from_row(row) for row in rows]

    def observation_count(self, platform: Optional[str] = None) -> int:
        if platform:
            with self._connection() as connection:
                row = connection.execute(
                    'SELECT COUNT(*) AS count FROM performance_observations WHERE platform = ?',
                    (_identifier(platform, 'platform'),),
                ).fetchone()
        else:
            with self._connection() as connection:
                row = connection.execute(
                    'SELECT COUNT(*) AS count FROM performance_observations'
                ).fetchone()
        return int(row['count'])

    @staticmethod
    def _import_from_row(row: sqlite3.Row) -> AnalyticsImportRecord:
        return AnalyticsImportRecord(
            id=row['id'],
            source_name=row['source_name'],
            source_sha256=row['source_sha256'],
            record_count=row['record_count'],
            imported_at=row['imported_at'],
        )

    @staticmethod
    def _observation_from_row(row: sqlite3.Row) -> PerformanceObservation:
        return PerformanceObservation(
            id=row['id'],
            import_id=row['import_id'],
            source_row=row['source_row'],
            publication_id=row['publication_id'],
            clip_path=row['clip_path'],
            clip_sha256=row['clip_sha256'],
            provider=row['provider'],
            platform=row['platform'],
            style=row['style'],
            segment_type=row['segment_type'],
            keywords=json.loads(row['keywords_json']),
            predicted_engagement_score=row['predicted_engagement_score'],
            views=row['views'],
            engaged_views=row['engaged_views'],
            retention_percent=row['retention_percent'],
            shares=row['shares'],
            conversions=row['conversions'],
            observed_at=row['observed_at'],
            imported_at=row['imported_at'],
        )


class AnalyticsService:
    """Normalize exports, report observed performance, and rerank candidates."""

    FIELD_ALIASES = {
        'predicted_engagement_score': (
            'predicted_engagement_score', 'predicted_engagement', 'engagement_score'
        ),
        'views': ('views', 'view_count'),
        'engaged_views': ('engaged_views', 'engaged_view_count'),
        'retention_percent': (
            'retention_percent', 'average_retention_percent', 'average_percentage_viewed'
        ),
        'retention_ratio': ('retention', 'average_retention'),
        'shares': ('shares', 'share_count'),
        'conversions': ('conversions', 'conversion_count'),
    }

    def __init__(self, store: AnalyticsStore, publication_store: Optional[Any] = None):
        self.store = store
        self.publication_store = publication_store

    @classmethod
    def create_default(cls, publication_store: Optional[Any] = None) -> 'AnalyticsService':
        return cls(AnalyticsStore(), publication_store)

    @staticmethod
    def import_schema() -> Dict[str, Any]:
        return {
            'formats': ['json', 'csv'],
            'json_shape': {'records': ['one or more observation objects']},
            'required_metrics': [
                'views', 'engaged_views', 'retention_percent', 'shares', 'conversions'
            ],
            'required_context': ['platform', 'style'],
            'linking': [
                'publication_id', 'clip_sha256', 'clip_path',
                'or manifest_path plus clip_number',
            ],
            'optional': [
                'provider', 'predicted_engagement_score', 'segment_type',
                'keywords', 'observed_at', 'manifest_path', 'clip_number',
            ],
            'retention': (
                'retention_percent fields use 0-100; retention fields accept '
                'a 0-1 fraction or 0-100 percentage.'
            ),
        }

    def import_file(self, source_path: str) -> AnalyticsImportResult:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f'Analytics import file not found: {path}')
        raw_bytes = path.read_bytes()
        source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        existing = self.store.find_import_by_sha256(source_sha256)
        if existing:
            return AnalyticsImportResult(existing, True)
        suffix = path.suffix.lower()
        try:
            text = raw_bytes.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Analytics imports must use UTF-8 encoding.') from error

        if suffix == '.json':
            rows = self._read_json(text)
        elif suffix == '.csv':
            rows = list(csv.DictReader(io.StringIO(text)))
        else:
            raise ValueError('Analytics imports must be .json or .csv files.')
        if not rows:
            raise ValueError('Analytics import contains no observations.')

        manifest_cache: Dict[str, Dict[str, Any]] = {}
        normalized: List[_NormalizedObservation] = []
        for index, raw_row in enumerate(rows, start=1):
            if not isinstance(raw_row, dict):
                raise ValueError(f'Analytics row {index} must be an object.')
            try:
                enriched = self._enrich_from_manifest(raw_row, path.parent, manifest_cache)
                normalized.append(self._normalize_row(enriched, index, path.parent))
            except ValueError as error:
                raise ValueError(f'Analytics row {index}: {error}') from error
        return self.store.create_import(str(path), source_sha256, normalized)

    @staticmethod
    def _read_json(text: str) -> List[Dict[str, Any]]:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f'Invalid analytics JSON: {error.msg}.') from error
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get('records'), list):
            return payload['records']
        if isinstance(payload, dict) and 'views' in payload:
            return [payload]
        raise ValueError('Analytics JSON must be an array or an object with a records array.')

    def _enrich_from_manifest(
        self,
        raw_row: Dict[str, Any],
        source_directory: Path,
        cache: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        row = {str(key).strip().lower(): value for key, value in raw_row.items()}
        manifest_value = str(row.get('manifest_path') or '').strip()
        if not manifest_value:
            return row
        manifest_path = Path(manifest_value).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = source_directory / manifest_path
        manifest_path = manifest_path.resolve()
        cache_key = str(manifest_path)
        if cache_key not in cache:
            if not manifest_path.is_file():
                raise ValueError(f'manifest file not found: {manifest_path}')
            try:
                cache[cache_key] = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise ValueError(f'invalid clip manifest: {manifest_path}') from error
        manifest = cache[cache_key]
        clips = manifest.get('clips', []) if isinstance(manifest, dict) else []
        requested_number = self._optional_int(row.get('clip_number'), 'clip_number')
        requested_path = str(row.get('clip_path') or '').strip()
        matched = None
        for clip in clips:
            if requested_number is not None and clip.get('clip_number') == requested_number:
                matched = clip
                break
            if requested_path and str(clip.get('output_path') or '') == requested_path:
                matched = clip
                break
        if matched is None:
            if len(clips) == 1 and requested_number is None and not requested_path:
                matched = clips[0]
            else:
                raise ValueError('manifest entry was not found; provide a valid clip_number.')
        defaults = {
            'clip_path': matched.get('optimized_path') or matched.get('output_path'),
            'platform': matched.get('platform') or manifest.get('platform'),
            'style': matched.get('style') or manifest.get('style'),
            'predicted_engagement_score': matched.get('engagement_score'),
            'segment_type': matched.get('segment_type'),
            'keywords': matched.get('keywords'),
        }
        for key, value in defaults.items():
            if row.get(key) in (None, '') and value not in (None, ''):
                row[key] = value
        return row

    def _normalize_row(
        self,
        raw_row: Dict[str, Any],
        row_number: int,
        source_directory: Path,
    ) -> _NormalizedObservation:
        row = {str(key).strip().lower(): value for key, value in raw_row.items()}
        publication_id = self._text(row.get('publication_id'))
        publication = None
        if publication_id:
            if not self.publication_store:
                raise ValueError('publication_id requires the publication store.')
            publication = self.publication_store.get_publication(publication_id)
            if not publication:
                raise ValueError(f'publication not found: {publication_id}')

        provider = self._text(row.get('provider')) or (
            publication.provider if publication else 'manual'
        )
        platform = self._text(row.get('platform')) or (
            publication.platform if publication else None
        )
        if publication:
            self._assert_link_match(row, publication)

        supplied_clip_path = self._text(row.get('clip_path'))
        clip_path = supplied_clip_path or (
            publication.clip_path if publication else None
        )
        if clip_path:
            candidate_path = Path(clip_path).expanduser()
            if not candidate_path.is_absolute():
                candidate_path = source_directory / candidate_path
            clip_path = str(candidate_path.resolve())
        if publication and supplied_clip_path:
            expected_path = str(Path(publication.clip_path).expanduser().resolve())
            if os.path.normcase(clip_path) != os.path.normcase(expected_path):
                raise ValueError(
                    f'clip_path does not match publication {publication.id}.'
                )
        clip_sha256 = self._text(row.get('clip_sha256')) or (
            publication.clip_sha256 if publication else None
        )
        if clip_sha256:
            clip_sha256 = clip_sha256.lower()
            if not re.fullmatch(r'[0-9a-f]{64}', clip_sha256):
                raise ValueError('clip_sha256 must contain exactly 64 hexadecimal characters.')
        elif clip_path and Path(clip_path).is_file():
            clip_sha256 = self._fingerprint(Path(clip_path))
        if not (publication_id or clip_path or clip_sha256):
            raise ValueError(
                'link the result with publication_id, clip_sha256, clip_path, or a manifest entry.'
            )

        style = _identifier(row.get('style'), 'style')
        segment_type = _identifier(row.get('segment_type') or 'unknown', 'segment_type')
        predicted = self._optional_float(
            self._aliased(row, 'predicted_engagement_score'),
            'predicted_engagement_score',
        )
        if predicted is not None and not 0 <= predicted <= 10:
            raise ValueError('predicted_engagement_score must be between 0 and 10.')

        views = self._required_int(self._aliased(row, 'views'), 'views')
        engaged_views = self._required_int(
            self._aliased(row, 'engaged_views'), 'engaged_views'
        )
        shares = self._required_int(self._aliased(row, 'shares'), 'shares')
        conversions = self._required_int(
            self._aliased(row, 'conversions'), 'conversions'
        )
        if views <= 0:
            raise ValueError('views must be greater than zero.')
        for name, value in (
            ('engaged_views', engaged_views), ('shares', shares), ('conversions', conversions)
        ):
            if value < 0:
                raise ValueError(f'{name} cannot be negative.')
        if engaged_views > views:
            raise ValueError('engaged_views cannot exceed views.')

        retention_value = self._aliased(row, 'retention_percent')
        retention_is_ratio = False
        if retention_value in (None, ''):
            retention_value = self._aliased(row, 'retention_ratio')
            retention_is_ratio = True
        retention = self._required_float(retention_value, 'retention_percent')
        if retention_is_ratio and 0 <= retention <= 1:
            retention *= 100
        if not 0 <= retention <= 100:
            raise ValueError('retention must be between 0 and 100 percent.')

        return _NormalizedObservation(
            source_row=row_number,
            publication_id=publication_id,
            clip_path=clip_path,
            clip_sha256=clip_sha256,
            provider=_identifier(provider, 'provider'),
            platform=_identifier(platform, 'platform'),
            style=style,
            segment_type=segment_type,
            keywords=self._keywords(row.get('keywords')),
            predicted_engagement_score=predicted,
            views=views,
            engaged_views=engaged_views,
            retention_percent=round(retention, 4),
            shares=shares,
            conversions=conversions,
            observed_at=self._timestamp(row.get('observed_at')),
            raw_payload=self._safe_raw_payload(raw_row),
        )

    @staticmethod
    def _safe_raw_payload(row: Dict[str, Any]) -> Dict[str, Any]:
        """Keep import audit context without copying accidental credentials."""
        sensitive_fragments = ('token', 'secret', 'password', 'authorization', 'cookie')
        return {
            str(key): value
            for key, value in row.items()
            if not any(fragment in str(key).lower() for fragment in sensitive_fragments)
        }

    @staticmethod
    def _assert_link_match(row: Dict[str, Any], publication: Any) -> None:
        for field in ('provider', 'platform', 'clip_sha256'):
            supplied = str(row.get(field) or '').strip().lower()
            expected = str(getattr(publication, field) or '').strip().lower()
            if field in {'provider', 'platform'}:
                supplied = _identifier(supplied, field, required=False)
                expected = _identifier(expected, field, required=False)
            if supplied and supplied != expected:
                raise ValueError(
                    f'{field} does not match publication {publication.id}: '
                    f'{supplied!r} != {expected!r}.'
                )

    @classmethod
    def _aliased(cls, row: Dict[str, Any], field: str) -> Any:
        for alias in cls.FIELD_ALIASES[field]:
            if alias in row and row[alias] not in (None, ''):
                return row[alias]
        return None

    @staticmethod
    def _text(value: Any) -> Optional[str]:
        text = str(value or '').strip()
        return text or None

    @staticmethod
    def _required_float(value: Any, field: str) -> float:
        if value in (None, ''):
            raise ValueError(f'{field} is required.')
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'{field} must be numeric.') from error
        if not math.isfinite(result):
            raise ValueError(f'{field} must be finite.')
        return result

    @classmethod
    def _optional_float(cls, value: Any, field: str) -> Optional[float]:
        if value in (None, ''):
            return None
        return cls._required_float(value, field)

    @classmethod
    def _required_int(cls, value: Any, field: str) -> int:
        result = cls._required_float(value, field)
        if not result.is_integer():
            raise ValueError(f'{field} must be a whole number.')
        return int(result)

    @classmethod
    def _optional_int(cls, value: Any, field: str) -> Optional[int]:
        if value in (None, ''):
            return None
        return cls._required_int(value, field)

    @staticmethod
    def _keywords(value: Any) -> List[str]:
        if value in (None, ''):
            return []
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.startswith('['):
                try:
                    value = json.loads(stripped)
                except json.JSONDecodeError:
                    value = re.split(r'[,;|]', stripped)
            else:
                value = re.split(r'[,;|]', stripped)
        if not isinstance(value, (list, tuple)):
            raise ValueError('keywords must be a list or comma-separated text.')
        normalized: List[str] = []
        for item in value:
            keyword = _identifier(item, 'keyword', required=False)
            if keyword and keyword not in normalized:
                normalized.append(keyword)
        return normalized[:30]

    @staticmethod
    def _timestamp(value: Any) -> str:
        if value in (None, ''):
            return _now()
        text = str(value).strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError as error:
            raise ValueError('observed_at must be an ISO-8601 timestamp.') from error
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat(timespec='milliseconds')

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()

    def build_report(
        self,
        platform: Optional[str] = None,
        style: Optional[str] = None,
    ) -> Dict[str, Any]:
        observations = self.store.list_observations(platform=platform, style=style)
        summary = self._aggregate(observations)
        platforms = self._group(observations, 'platform')
        styles = self._group(observations, 'style')
        prediction = self._prediction_comparison(observations)
        return {
            'filters': {
                'platform': _identifier(platform, 'platform') if platform else None,
                'style': _identifier(style, 'style') if style else None,
            },
            'summary': summary,
            'prediction_comparison': prediction,
            'by_platform': platforms,
            'by_style': styles,
            'recommendations': self._recommendations(observations, prediction),
            'methodology': {
                'actual_performance_score': (
                    '0-10 directional score: 35% retention, 35% engaged-view rate, '
                    '20% share signal (capped at 5% of views), and 10% conversion '
                    'signal (capped at 2% of views).'
                ),
                'guidance': 'Compare cohorts; do not treat the score as a revenue forecast.',
            },
        }

    @staticmethod
    def _aggregate(observations: Sequence[PerformanceObservation]) -> Dict[str, Any]:
        count = len(observations)
        views = sum(item.views for item in observations)
        engaged = sum(item.engaged_views for item in observations)
        shares = sum(item.shares for item in observations)
        conversions = sum(item.conversions for item in observations)
        retention = (
            sum(item.retention_percent * item.views for item in observations) / views
            if views else 0.0
        )
        return {
            'observation_count': count,
            'total_views': views,
            'total_engaged_views': engaged,
            'engaged_view_rate': round(engaged / views, 6) if views else 0.0,
            'weighted_retention_percent': round(retention, 2),
            'total_shares': shares,
            'share_rate': round(shares / views, 6) if views else 0.0,
            'total_conversions': conversions,
            'conversion_rate': round(conversions / views, 6) if views else 0.0,
            'actual_performance_score': round(
                _performance_score(views, engaged, retention, shares, conversions), 2
            ),
        }

    @classmethod
    def _group(
        cls,
        observations: Sequence[PerformanceObservation],
        attribute: str,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[str, List[PerformanceObservation]] = {}
        for observation in observations:
            grouped.setdefault(getattr(observation, attribute), []).append(observation)
        results = []
        for name, items in grouped.items():
            results.append({attribute: name, **cls._aggregate(items)})
        return sorted(
            results,
            key=lambda item: (-item['actual_performance_score'], -item['total_views'], item[attribute]),
        )

    @staticmethod
    def _prediction_comparison(
        observations: Sequence[PerformanceObservation],
    ) -> Dict[str, Any]:
        matched = [
            item for item in observations if item.predicted_engagement_score is not None
        ]
        if not matched:
            return {
                'matched_observation_count': 0,
                'average_predicted_score': None,
                'average_actual_score': None,
                'mean_prediction_delta': None,
                'mean_absolute_error': None,
                'overpredicted_count': 0,
                'underpredicted_count': 0,
            }
        deltas = [
            item.actual_performance_score - item.predicted_engagement_score
            for item in matched
        ]
        return {
            'matched_observation_count': len(matched),
            'average_predicted_score': round(
                sum(item.predicted_engagement_score for item in matched) / len(matched), 2
            ),
            'average_actual_score': round(
                sum(item.actual_performance_score for item in matched) / len(matched), 2
            ),
            'mean_prediction_delta': round(sum(deltas) / len(deltas), 2),
            'mean_absolute_error': round(sum(abs(delta) for delta in deltas) / len(deltas), 2),
            'overpredicted_count': sum(delta < 0 for delta in deltas),
            'underpredicted_count': sum(delta > 0 for delta in deltas),
        }

    @classmethod
    def _recommendations(
        cls,
        observations: Sequence[PerformanceObservation],
        prediction: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        if not observations:
            return [{
                'type': 'insufficient_data',
                'message': 'No observed project performance has been imported yet.',
                'evidence': {'observation_count': 0, 'total_views': 0},
            }]
        recommendations: List[Dict[str, Any]] = []
        platform_groups: Dict[str, List[PerformanceObservation]] = {}
        for item in observations:
            platform_groups.setdefault(item.platform, []).append(item)
        for platform, platform_items in sorted(platform_groups.items()):
            baseline = cls._aggregate(platform_items)
            styles: Dict[str, List[PerformanceObservation]] = {}
            for item in platform_items:
                styles.setdefault(item.style, []).append(item)
            best_style, best_items = max(
                styles.items(),
                key=lambda pair: (
                    cls._aggregate(pair[1])['actual_performance_score'],
                    cls._aggregate(pair[1])['total_views'],
                ),
            )
            evidence = cls._aggregate(best_items)
            uplift = round(
                evidence['actual_performance_score']
                - baseline['actual_performance_score'],
                2,
            )
            confidence = 'directional' if len(best_items) < 3 else 'established'
            recommendations.append({
                'type': 'style_on_platform',
                'platform': platform,
                'style': best_style,
                'confidence': confidence,
                'score_uplift_vs_platform': uplift,
                'message': (
                    f'Observed {best_style} on {platform}: '
                    f'{evidence["total_views"]:,} views across '
                    f'{evidence["observation_count"]} result(s), '
                    f'{evidence["weighted_retention_percent"]:.2f}% retention, '
                    f'{evidence["engaged_view_rate"] * 100:.2f}% engaged views, '
                    f'{evidence["share_rate"] * 100:.2f}% shares, and '
                    f'{evidence["conversion_rate"] * 100:.2f}% conversions. '
                    f'Its {evidence["actual_performance_score"]:.2f}/10 observed score '
                    f'is {uplift:+.2f} versus this platform baseline.'
                ),
                'evidence': evidence,
            })
        if prediction['matched_observation_count']:
            recommendations.append({
                'type': 'prediction_calibration',
                'message': (
                    f'Across {prediction["matched_observation_count"]} matched result(s), '
                    f'predictions averaged {prediction["average_predicted_score"]:.2f}/10 '
                    f'and observed outcomes averaged {prediction["average_actual_score"]:.2f}/10 '
                    f'(mean delta {prediction["mean_prediction_delta"]:+.2f}).'
                ),
                'evidence': prediction,
            })
        return recommendations

    def candidate_pool_size(self, requested_count: int, platform: str = 'general') -> int:
        requested = max(1, int(requested_count))
        if self.store.observation_count(platform) or self.store.observation_count():
            return min(10, max(requested + 2, requested * 2))
        return requested

    def rerank_segments(
        self,
        segments: Sequence[Any],
        platform: str = 'general',
        requested_style: str = 'engaging',
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Order candidates using shrunken feature outcomes; keep predictions intact."""
        segment_list = list(segments)
        normalized_platform = _identifier(platform or 'general', 'platform')
        history = self.store.list_observations(platform=normalized_platform)
        history_scope = normalized_platform
        if not history:
            history = self.store.list_observations()
            history_scope = 'all_platforms'
        if not history or len(segment_list) < 2:
            return segment_list, {
                'applied': False,
                'history_scope': history_scope,
                'observation_count': len(history),
                'candidates': [],
            }

        baseline = sum(item.actual_performance_score for item in history) / len(history)
        requested_style_id = _identifier(
            requested_style or 'engaging', 'requested_style'
        )
        scored = []
        for original_index, segment in enumerate(segment_list):
            base = float(getattr(segment, 'engagement_score', 0.0) or 0.0)
            segment_type = _identifier(
                getattr(segment, 'segment_type', None) or 'unknown',
                'segment_type',
            )
            emotion = _identifier(
                getattr(segment, 'emotion', None) or requested_style_id,
                'style',
            )
            style = emotion if requested_style_id in {'engaging', 'viral'} else requested_style_id
            keywords = {
                _identifier(item, 'keyword', required=False)
                for item in (getattr(segment, 'keywords', None) or [])
            }
            keywords.discard('')
            evidence: List[Dict[str, Any]] = []
            adjustment = 0.0
            feature_specs = [
                (
                    'style', style, 0.45,
                    [item for item in history if item.style == style],
                ),
                (
                    'segment_type', segment_type, 0.35,
                    [item for item in history if item.segment_type == segment_type],
                ),
                (
                    'keywords', ','.join(sorted(keywords)), 0.20,
                    [item for item in history if keywords.intersection(item.keywords)],
                ),
            ]
            for feature, value, weight, matched in feature_specs:
                if not value or not matched:
                    continue
                feature_score = sum(
                    item.actual_performance_score for item in matched
                ) / len(matched)
                confidence = len(matched) / (len(matched) + 2.0)
                contribution = (feature_score - baseline) * confidence * weight
                adjustment += contribution
                evidence.append({
                    'feature': feature,
                    'value': value,
                    'observation_count': len(matched),
                    'observed_score': round(feature_score, 2),
                    'baseline_score': round(baseline, 2),
                    'contribution': round(contribution, 3),
                })
            final_score = min(10.0, max(0.0, base + adjustment))
            scored.append({
                'segment': segment,
                'original_index': original_index,
                'base_score': base,
                'historical_adjustment': adjustment,
                'final_score': final_score,
                'evidence': evidence,
            })

        scored.sort(key=lambda item: (-item['final_score'], item['original_index']))
        candidates = []
        for new_index, item in enumerate(scored):
            segment = item['segment']
            candidates.append({
                'title': getattr(segment, 'suggested_title', ''),
                'start_time': getattr(segment, 'start_time', None),
                'original_rank': item['original_index'] + 1,
                'reranked_position': new_index + 1,
                'predicted_engagement_score': round(item['base_score'], 2),
                'historical_adjustment': round(item['historical_adjustment'], 3),
                'ranking_score': round(item['final_score'], 3),
                'evidence': item['evidence'],
            })
        return [item['segment'] for item in scored], {
            'applied': any(item['evidence'] for item in scored),
            'history_scope': history_scope,
            'observation_count': len(history),
            'baseline_observed_score': round(baseline, 2),
            'candidates': candidates,
        }
