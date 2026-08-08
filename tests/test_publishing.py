import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.credential_vault import InMemoryTokenVault, WindowsDpapiTokenVault
from src.oauth_connections import (
    OAuthAdapter,
    OAuthAdapterRegistry,
    OAuthConnectionService,
    OAuthExchange,
)
from src.publication_store import PublicationStore
from src.publishing import (
    LocalDraftProvider,
    ProviderCapabilities,
    ProviderContext,
    ProviderRegistry,
    ProviderResult,
    PublishRequest,
    PublishingProvider,
    PublishingService,
    PublishingValidationError,
)


class FakePublishingProvider(PublishingProvider):
    def __init__(self):
        self.calls = []
        self._capabilities = ProviderCapabilities(
            provider='fake',
            display_name='Fake live provider',
            modes=('draft', 'publish', 'schedule'),
            platforms=('youtube_shorts',),
            privacy_levels=('private', 'unlisted', 'public'),
            account_required=True,
            max_title_characters=40,
            max_caption_characters=200,
            max_hashtags=3,
        )

    @property
    def capabilities(self):
        return self._capabilities

    def validate_request(self, request):
        if 'forbidden' in request.caption.lower():
            raise PublishingValidationError('Provider rejected forbidden caption text.')

    def _result(self, mode, request, context, state):
        self.calls.append((mode, request, context))
        if context.token_bundle.get('access_token') != 'secure-token':
            raise RuntimeError('Missing account token')
        return ProviderResult(
            state=state,
            external_id=f'{mode}-123',
            external_url=f'https://provider.invalid/{mode}-123',
            payload={'mode': mode},
        )

    def create_draft(self, request, context):
        return self._result('draft', request, context, 'drafted')

    def publish_now(self, request, context):
        return self._result('publish', request, context, 'published')

    def schedule(self, request, context):
        return self._result('schedule', request, context, 'scheduled')


class FakeOAuthAdapter(OAuthAdapter):
    def __init__(self):
        self.authorization = None
        self.exchange = None

    @property
    def provider(self):
        return 'fake'

    def build_authorization_url(self, redirect_uri, state, code_challenge):
        self.authorization = (redirect_uri, state, code_challenge)
        return f'https://provider.invalid/oauth?state={state}&challenge={code_challenge}'

    def exchange_code(self, code, redirect_uri, code_verifier):
        self.exchange = (code, redirect_uri, code_verifier)
        return OAuthExchange(
            external_account_id='channel-123',
            display_name='Test Channel',
            token_bundle={'access_token': 'secure-token', 'refresh_token': 'refresh-token'},
            metadata={'channel_type': 'creator', 'access_token_hint': 'must-not-persist'},
        )


class PublishingTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.store = PublicationStore(str(self.root / 'publishing.sqlite3'))
        self.vault = InMemoryTokenVault()
        self.clip = self.root / 'clip.mp4'
        self.clip.write_bytes(b'video-data-v1')

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_local_draft_requires_exact_approval_and_is_idempotent(self):
        registry = ProviderRegistry()
        registry.register(LocalDraftProvider(str(self.root / 'drafts')))
        service = PublishingService(self.store, registry, self.vault)
        approval = self.store.create_approval(str(self.clip), 'reviewer@example.com')
        request = PublishRequest(
            provider='local',
            platform='youtube_shorts',
            mode='draft',
            clip_path=str(self.clip),
            approval_id=approval.id,
            title='A useful short',
            caption='Watch this useful moment.',
            hashtags=['Shorts', '#Useful', '#Useful'],
            privacy='private',
        )

        first = service.submit(request)
        second = service.submit(request)

        self.assertEqual(first.state, 'drafted')
        self.assertEqual(first.id, second.id)
        self.assertEqual(first.hashtags, ['#Shorts', '#Useful'])
        draft_path = Path(first.provider_payload['draft_path'])
        payload = json.loads(draft_path.read_text(encoding='utf-8'))
        self.assertFalse(payload['external_upload_performed'])
        self.assertEqual(payload['approval_id'], approval.id)
        self.assertEqual(len(self.store.list_publications()), 1)

        self.clip.write_bytes(b'changed-after-approval')
        with self.assertRaisesRegex(PublishingValidationError, 'changed'):
            service.submit(request)

    def test_local_provider_refuses_unavailable_live_modes(self):
        registry = ProviderRegistry()
        registry.register(LocalDraftProvider(str(self.root / 'drafts')))
        service = PublishingService(self.store, registry, self.vault)
        approval = self.store.create_approval(str(self.clip), 'reviewer')

        with self.assertRaisesRegex(PublishingValidationError, 'does not support mode'):
            service.submit(PublishRequest(
                provider='local', platform='tiktok', mode='publish',
                clip_path=str(self.clip), approval_id=approval.id,
                title='Do not publish', privacy='private',
            ))

    def test_provider_contract_supports_account_draft_publish_and_schedule(self):
        provider = FakePublishingProvider()
        registry = ProviderRegistry()
        registry.register(provider)
        account = self.store.upsert_account('fake', 'channel-123', 'Test Channel')
        self.vault.store('fake', account.id, {'access_token': 'secure-token'})
        service = PublishingService(self.store, registry, self.vault)
        approval = self.store.create_approval(str(self.clip), 'reviewer')

        published = service.submit(PublishRequest(
            provider='fake', platform='youtube_shorts', mode='publish',
            clip_path=str(self.clip), approval_id=approval.id,
            account_id=account.id, title='Publish this', caption='Allowed',
            hashtags=['one', 'two', 'three', 'four'], privacy='unlisted',
        ))
        scheduled_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat()
        scheduled = service.submit(PublishRequest(
            provider='fake', platform='youtube_shorts', mode='schedule',
            clip_path=str(self.clip), approval_id=approval.id,
            account_id=account.id, title='Schedule this', privacy='private',
            scheduled_at=scheduled_time,
        ))

        self.assertEqual(published.state, 'published')
        self.assertEqual(published.hashtags, ['#one', '#two', '#three'])
        self.assertEqual(scheduled.state, 'scheduled')
        self.assertEqual([call[0] for call in provider.calls], ['publish', 'schedule'])

    def test_provider_validation_runs_before_creating_publication(self):
        provider = FakePublishingProvider()
        registry = ProviderRegistry()
        registry.register(provider)
        account = self.store.upsert_account('fake', 'channel-123', 'Test Channel')
        self.vault.store('fake', account.id, {'access_token': 'secure-token'})
        service = PublishingService(self.store, registry, self.vault)
        approval = self.store.create_approval(str(self.clip), 'reviewer')

        with self.assertRaisesRegex(PublishingValidationError, 'forbidden'):
            service.submit(PublishRequest(
                provider='fake', platform='youtube_shorts', mode='draft',
                clip_path=str(self.clip), approval_id=approval.id,
                account_id=account.id, title='Draft', caption='Forbidden phrase',
            ))
        self.assertEqual(self.store.list_publications(), [])

    def test_oauth_pkce_state_and_tokens_use_separate_secure_vault(self):
        adapter = FakeOAuthAdapter()
        adapters = OAuthAdapterRegistry()
        adapters.register(adapter)
        service = OAuthConnectionService(self.store, self.vault, adapters)

        started = service.begin_connection(
            'fake',
            'http://127.0.0.1:8765/oauth/callback',
        )
        self.assertGreater(len(adapter.authorization[2]), 20)
        with self.assertRaisesRegex(ValueError, 'state validation'):
            service.complete_connection(started.session_id, 'wrong-state', 'code-1')

        account = service.complete_connection(started.session_id, started.state, 'code-1')

        self.assertEqual(account.provider, 'fake')
        self.assertNotIn('access_token_hint', account.metadata)
        self.assertEqual(self.vault.retrieve('fake', account.id)['refresh_token'], 'refresh-token')
        self.assertIsNone(self.store.get_oauth_session(started.session_id))
        self.assertTrue(service.disconnect(account.id))
        self.assertIsNone(self.vault.retrieve('fake', account.id))

        with self.assertRaisesRegex(ValueError, 'credential fields'):
            self.store.upsert_account(
                'fake',
                'unsafe-account',
                'Unsafe',
                {'access_token': 'must-not-enter-sqlite'},
            )

    @unittest.skipUnless(os.name == 'nt', 'Windows DPAPI is only available on Windows')
    def test_windows_dpapi_vault_does_not_store_plaintext(self):
        vault = WindowsDpapiTokenVault(str(self.root / 'credentials'))
        vault.store('fake', 'account', {'access_token': 'test-secret-value'})

        encrypted = next((self.root / 'credentials').glob('*.token')).read_bytes()
        self.assertNotIn(b'test-secret-value', encrypted)
        self.assertEqual(vault.retrieve('fake', 'account')['access_token'], 'test-secret-value')


if __name__ == '__main__':
    unittest.main()
