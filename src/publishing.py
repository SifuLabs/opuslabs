"""Provider-neutral, approval-gated publishing workflow."""

import hashlib
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .credential_vault import TokenVault, create_default_token_vault
from .publication_store import AccountRecord, PublicationRecord, PublicationStore


PUBLISH_MODES = {'draft', 'publish', 'schedule'}
PRIVACY_LEVELS = {'private', 'unlisted', 'public'}
SUPPORTED_MEDIA_SUFFIXES = {'.mp4', '.mov', '.mkv', '.webm'}


class PublishingValidationError(ValueError):
    """Raised when a request cannot safely be sent to a provider."""


@dataclass(frozen=True)
class ProviderCapabilities:
    provider: str
    display_name: str
    modes: Tuple[str, ...]
    platforms: Tuple[str, ...]
    privacy_levels: Tuple[str, ...]
    account_required: bool
    max_title_characters: int = 100
    max_caption_characters: int = 5000
    max_hashtags: int = 30

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PublishRequest:
    provider: str
    platform: str
    mode: str
    clip_path: str
    approval_id: str
    title: str
    caption: str = ''
    hashtags: Sequence[str] = field(default_factory=tuple)
    privacy: str = 'private'
    account_id: Optional[str] = None
    scheduled_at: Optional[str] = None
    idempotency_key: Optional[str] = None


@dataclass(frozen=True)
class ProviderContext:
    publication_id: str
    account: Optional[AccountRecord]
    token_bundle: Optional[Dict[str, Any]]


@dataclass(frozen=True)
class ProviderResult:
    state: str
    external_id: Optional[str] = None
    external_url: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)


class PublishingProvider(ABC):
    """Contract implemented by every external or local publishing provider."""

    @property
    @abstractmethod
    def capabilities(self) -> ProviderCapabilities:
        raise NotImplementedError

    def validate_request(self, request: PublishRequest) -> None:
        """Provider-owned hook for dynamic platform and account requirements."""

    @abstractmethod
    def create_draft(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    def publish_now(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        raise NotImplementedError

    @abstractmethod
    def schedule(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        raise NotImplementedError


class ProviderRegistry:
    def __init__(self):
        self._providers: Dict[str, PublishingProvider] = {}

    def register(self, provider: PublishingProvider) -> None:
        name = provider.capabilities.provider.strip().lower()
        if not name:
            raise ValueError('Publishing providers require a stable name.')
        self._providers[name] = provider

    def get(self, name: str) -> PublishingProvider:
        provider = self._providers.get(str(name or '').strip().lower())
        if not provider:
            available = ', '.join(sorted(self._providers)) or 'none'
            raise PublishingValidationError(
                f'Publishing provider "{name}" is unavailable. Available providers: {available}.'
            )
        return provider

    def list_capabilities(self) -> List[ProviderCapabilities]:
        return [
            self._providers[name].capabilities
            for name in sorted(self._providers)
        ]


class LocalDraftProvider(PublishingProvider):
    """Create a local draft package without any external network side effect."""

    def __init__(self, draft_directory: Optional[str] = None):
        self.draft_directory = Path(
            draft_directory or os.getenv('PUBLISH_DRAFT_DIR', './.opuslabs/publishing/drafts')
        ).resolve()
        self.draft_directory.mkdir(parents=True, exist_ok=True)
        self._capabilities = ProviderCapabilities(
            provider='local',
            display_name='Local draft package',
            modes=('draft',),
            platforms=('general', 'youtube_shorts', 'tiktok', 'instagram', 'linkedin'),
            privacy_levels=('private', 'unlisted', 'public'),
            account_required=False,
        )

    @property
    def capabilities(self) -> ProviderCapabilities:
        return self._capabilities

    def create_draft(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        draft_path = self.draft_directory / f'{context.publication_id}.json'
        payload = {
            'schema_version': 1,
            'publication_id': context.publication_id,
            'provider': 'local',
            'external_upload_performed': False,
            'platform': request.platform,
            'clip_path': str(Path(request.clip_path).resolve()),
            'title': request.title,
            'caption': request.caption,
            'hashtags': list(request.hashtags),
            'privacy': request.privacy,
            'approval_id': request.approval_id,
            'created_at': datetime.now(timezone.utc).isoformat(timespec='milliseconds'),
        }
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=draft_path.parent,
                prefix=f'.{draft_path.name}.',
                suffix='.tmp',
                delete=False,
            ) as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write('\n')
                temporary_path = Path(handle.name)
            os.replace(temporary_path, draft_path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        return ProviderResult(
            state='drafted',
            external_id=context.publication_id,
            payload={
                'draft_path': str(draft_path),
                'external_upload_performed': False,
            },
        )

    def publish_now(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        raise PublishingValidationError('The local provider cannot publish externally.')

    def schedule(self, request: PublishRequest, context: ProviderContext) -> ProviderResult:
        raise PublishingValidationError('The local provider cannot schedule external publishing.')


class PublishingService:
    """Validate approval evidence, enforce capabilities, and record each attempt."""

    def __init__(
        self,
        store: Optional[PublicationStore] = None,
        registry: Optional[ProviderRegistry] = None,
        token_vault: Optional[TokenVault] = None,
    ):
        self.store = store or PublicationStore()
        self.registry = registry or ProviderRegistry()
        if not self.registry.list_capabilities():
            self.registry.register(LocalDraftProvider())
        self.token_vault = token_vault or create_default_token_vault()

    @staticmethod
    def _normalize_hashtags(hashtags: Sequence[str], maximum: int) -> List[str]:
        normalized: List[str] = []
        for value in hashtags or []:
            hashtag = re.sub(r'[^\w-]', '', str(value).strip().lstrip('#'), flags=re.UNICODE)
            if not hashtag:
                continue
            rendered = f'#{hashtag}'
            if rendered not in normalized:
                normalized.append(rendered)
            if len(normalized) >= maximum:
                break
        return normalized

    @staticmethod
    def _parse_scheduled_at(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            scheduled = datetime.fromisoformat(value.replace('Z', '+00:00'))
        except ValueError as error:
            raise PublishingValidationError('scheduled_at must be an ISO-8601 timestamp.') from error
        if scheduled.tzinfo is None:
            raise PublishingValidationError('scheduled_at must include a timezone offset.')
        return scheduled.astimezone(timezone.utc)

    def _validated_request(
        self,
        request: PublishRequest,
    ) -> Tuple[PublishRequest, PublishingProvider, str, Optional[AccountRecord], Optional[Dict[str, Any]]]:
        provider = self.registry.get(request.provider)
        capabilities = provider.capabilities
        mode = str(request.mode or '').strip().lower()
        platform = str(request.platform or '').strip().lower()
        privacy = str(request.privacy or '').strip().lower()
        if mode not in PUBLISH_MODES or mode not in capabilities.modes:
            raise PublishingValidationError(
                f'{capabilities.display_name} does not support mode "{mode}".'
            )
        if platform not in capabilities.platforms:
            raise PublishingValidationError(
                f'{capabilities.display_name} does not support platform "{platform}".'
            )
        if privacy not in PRIVACY_LEVELS or privacy not in capabilities.privacy_levels:
            raise PublishingValidationError(
                f'{capabilities.display_name} does not support privacy "{privacy}".'
            )

        clip_path = Path(request.clip_path).expanduser().resolve()
        if clip_path.suffix.lower() not in SUPPORTED_MEDIA_SUFFIXES:
            raise PublishingValidationError(f'Unsupported publish media format: {clip_path.suffix}')
        if not clip_path.is_file() or clip_path.stat().st_size <= 0:
            raise PublishingValidationError(f'Publish clip is missing or empty: {clip_path}')

        approval = self.store.get_approval(request.approval_id)
        if not approval:
            raise PublishingValidationError('An existing approval record is required before publishing.')
        clip_hash = self.store.fingerprint_file(str(clip_path))
        if approval.clip_sha256 != clip_hash or Path(approval.clip_path).resolve() != clip_path:
            raise PublishingValidationError(
                'The approved clip has changed or does not match this publishing request.'
            )

        title = re.sub(r'\s+', ' ', str(request.title or '').strip())
        caption = str(request.caption or '').strip()
        if not title:
            raise PublishingValidationError('A publish title is required.')
        if len(title) > capabilities.max_title_characters:
            raise PublishingValidationError(
                f'Title exceeds the provider limit of {capabilities.max_title_characters} characters.'
            )
        if len(caption) > capabilities.max_caption_characters:
            raise PublishingValidationError(
                f'Caption exceeds the provider limit of {capabilities.max_caption_characters} characters.'
            )
        hashtags = self._normalize_hashtags(request.hashtags, capabilities.max_hashtags)

        scheduled = self._parse_scheduled_at(request.scheduled_at)
        if mode == 'schedule':
            if not scheduled or scheduled <= datetime.now(timezone.utc):
                raise PublishingValidationError('Scheduled publishing requires a future timestamp.')
        elif scheduled:
            raise PublishingValidationError('scheduled_at is only valid in schedule mode.')

        account = None
        token_bundle = None
        if request.account_id:
            account = self.store.get_account(request.account_id)
            if not account or account.provider != capabilities.provider:
                raise PublishingValidationError('Publishing account does not match the selected provider.')
            token_bundle = self.token_vault.retrieve(capabilities.provider, account.id)
            if not token_bundle:
                raise PublishingValidationError('Publishing account credentials are unavailable.')
        elif capabilities.account_required:
            raise PublishingValidationError('This provider requires a connected account.')

        normalized = PublishRequest(
            provider=capabilities.provider,
            platform=platform,
            mode=mode,
            clip_path=str(clip_path),
            approval_id=approval.id,
            title=title,
            caption=caption,
            hashtags=hashtags,
            privacy=privacy,
            account_id=account.id if account else None,
            scheduled_at=scheduled.isoformat() if scheduled else None,
            idempotency_key=request.idempotency_key,
        )
        provider.validate_request(normalized)
        return normalized, provider, clip_hash, account, token_bundle

    @staticmethod
    def _idempotency_key(request: PublishRequest, clip_hash: str) -> str:
        if request.idempotency_key:
            return str(request.idempotency_key).strip()[:200]
        payload = json.dumps(
            {
                'provider': request.provider,
                'platform': request.platform,
                'mode': request.mode,
                'clip': clip_hash,
                'approval': request.approval_id,
                'account': request.account_id,
                'title': request.title,
                'caption': request.caption,
                'hashtags': list(request.hashtags),
                'privacy': request.privacy,
                'scheduled_at': request.scheduled_at,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

    def submit(self, request: PublishRequest) -> PublicationRecord:
        normalized, provider, clip_hash, account, token_bundle = self._validated_request(request)
        idempotency_key = self._idempotency_key(normalized, clip_hash)
        existing = self.store.find_by_idempotency_key(idempotency_key)
        if existing:
            return existing
        publication = self.store.create_publication({
            'provider': normalized.provider,
            'platform': normalized.platform,
            'mode': normalized.mode,
            'clip_path': normalized.clip_path,
            'clip_sha256': clip_hash,
            'approval_id': normalized.approval_id,
            'account_id': normalized.account_id,
            'title': normalized.title,
            'caption': normalized.caption,
            'hashtags': list(normalized.hashtags),
            'privacy': normalized.privacy,
            'scheduled_at': normalized.scheduled_at,
            'idempotency_key': idempotency_key,
        })
        context = ProviderContext(publication.id, account, token_bundle)
        try:
            if normalized.mode == 'draft':
                result = provider.create_draft(normalized, context)
            elif normalized.mode == 'publish':
                result = provider.publish_now(normalized, context)
            else:
                result = provider.schedule(normalized, context)
            return self.store.finish_publication(
                publication.id,
                result.state,
                result.external_id,
                result.external_url,
                result.payload,
            )
        except Exception as error:
            return self.store.fail_publication(publication.id, str(error))


def create_default_publishing_service() -> PublishingService:
    registry = ProviderRegistry()
    registry.register(LocalDraftProvider())
    return PublishingService(registry=registry)

