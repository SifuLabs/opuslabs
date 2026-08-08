"""OAuth connection orchestration with PKCE and OS-backed token storage."""

import base64
import hashlib
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from .credential_vault import TokenVault, create_default_token_vault
from .publication_store import AccountRecord, PublicationStore


@dataclass(frozen=True)
class OAuthExchange:
    external_account_id: str
    display_name: str
    token_bundle: Dict[str, Any] = field(repr=False)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OAuthStart:
    session_id: str
    state: str = field(repr=False)
    authorization_url: str
    expires_at: str


class OAuthAdapter(ABC):
    @property
    @abstractmethod
    def provider(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def build_authorization_url(
        self,
        redirect_uri: str,
        state: str,
        code_challenge: str,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def exchange_code(
        self,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> OAuthExchange:
        raise NotImplementedError


class OAuthAdapterRegistry:
    def __init__(self):
        self._adapters: Dict[str, OAuthAdapter] = {}

    def register(self, adapter: OAuthAdapter) -> None:
        provider = adapter.provider.strip().lower()
        if not provider:
            raise ValueError('OAuth adapters require a stable provider name.')
        self._adapters[provider] = adapter

    def get(self, provider: str) -> OAuthAdapter:
        adapter = self._adapters.get(str(provider or '').strip().lower())
        if not adapter:
            available = ', '.join(sorted(self._adapters)) or 'none'
            raise ValueError(f'OAuth provider is unavailable. Available adapters: {available}.')
        return adapter

    def list_providers(self) -> List[str]:
        return sorted(self._adapters)


class OAuthConnectionService:
    """Run state-checked OAuth callbacks without persisting plaintext tokens."""

    def __init__(
        self,
        store: Optional[PublicationStore] = None,
        token_vault: Optional[TokenVault] = None,
        adapters: Optional[OAuthAdapterRegistry] = None,
    ):
        self.store = store or PublicationStore()
        self.token_vault = token_vault or create_default_token_vault()
        self.adapters = adapters or OAuthAdapterRegistry()

    @staticmethod
    def _validate_redirect_uri(redirect_uri: str) -> str:
        parsed = urlparse(str(redirect_uri or '').strip())
        is_local = parsed.hostname in {'localhost', '127.0.0.1', '::1'}
        if not parsed.scheme or not parsed.netloc:
            raise ValueError('OAuth redirect URI must be absolute.')
        if parsed.scheme != 'https' and not (parsed.scheme == 'http' and is_local):
            raise ValueError('OAuth redirect URI must use HTTPS, except for localhost development.')
        return parsed.geturl()

    def begin_connection(
        self,
        provider: str,
        redirect_uri: str,
        lifetime_minutes: int = 10,
    ) -> OAuthStart:
        if not self.token_vault.available:
            raise RuntimeError('A secure token vault is required before starting OAuth.')
        if not 1 <= int(lifetime_minutes) <= 30:
            raise ValueError('OAuth session lifetime must be between 1 and 30 minutes.')
        adapter = self.adapters.get(provider)
        redirect = self._validate_redirect_uri(redirect_uri)
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(
            hashlib.sha256(verifier.encode('ascii')).digest()
        ).rstrip(b'=').decode('ascii')
        expires = datetime.now(timezone.utc) + timedelta(minutes=int(lifetime_minutes))
        state_hash = hashlib.sha256(state.encode('utf-8')).hexdigest()
        session_id = self.store.create_oauth_session(
            adapter.provider,
            state_hash,
            redirect,
            expires.isoformat(timespec='milliseconds'),
        )
        try:
            self.token_vault.store(
                'oauth-session',
                session_id,
                {'state': state, 'code_verifier': verifier},
            )
            authorization_url = adapter.build_authorization_url(
                redirect,
                state,
                challenge,
            )
        except Exception:
            self.store.delete_oauth_session(session_id)
            self.token_vault.delete('oauth-session', session_id)
            raise
        return OAuthStart(
            session_id=session_id,
            state=state,
            authorization_url=authorization_url,
            expires_at=expires.isoformat(timespec='milliseconds'),
        )

    def complete_connection(
        self,
        session_id: str,
        state: str,
        code: str,
    ) -> AccountRecord:
        session = self.store.get_oauth_session(session_id)
        if not session:
            raise ValueError('OAuth session was not found or was already consumed.')
        expires = datetime.fromisoformat(session['expires_at'])
        if expires <= datetime.now(timezone.utc):
            self.store.delete_oauth_session(session_id)
            self.token_vault.delete('oauth-session', session_id)
            raise ValueError('OAuth session has expired.')
        secret_session = self.token_vault.retrieve('oauth-session', session_id)
        if not secret_session:
            raise ValueError('OAuth session secrets are unavailable.')
        expected_state = secret_session.get('state', '')
        provided_hash = hashlib.sha256(str(state).encode('utf-8')).hexdigest()
        if (
            not secrets.compare_digest(str(state), str(expected_state))
            or not secrets.compare_digest(provided_hash, session['state_hash'])
        ):
            raise ValueError('OAuth state validation failed.')
        if not str(code or '').strip():
            raise ValueError('OAuth authorization code is required.')

        adapter = self.adapters.get(session['provider'])
        try:
            exchange = adapter.exchange_code(
                str(code).strip(),
                session['redirect_uri'],
                secret_session['code_verifier'],
            )
            safe_metadata = {
                key: value
                for key, value in exchange.metadata.items()
                if not any(secret_name in key.lower() for secret_name in ('token', 'secret', 'password'))
            }
            account = self.store.upsert_account(
                adapter.provider,
                exchange.external_account_id,
                exchange.display_name,
                safe_metadata,
            )
            self.token_vault.store(adapter.provider, account.id, exchange.token_bundle)
            return account
        finally:
            self.store.delete_oauth_session(session_id)
            self.token_vault.delete('oauth-session', session_id)

    def disconnect(self, account_id: str) -> bool:
        account = self.store.get_account(account_id)
        if not account:
            return False
        self.token_vault.delete(account.provider, account.id)
        return self.store.delete_account(account.id)

