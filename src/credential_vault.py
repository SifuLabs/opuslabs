"""Secure token-vault implementations for publishing account credentials."""

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional


class SecureStorageUnavailable(RuntimeError):
    """Raised when no OS-backed persistent secret store is available."""


class TokenVault(ABC):
    """Provider-neutral secure storage contract for OAuth token bundles."""

    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def store(self, namespace: str, identifier: str, token_bundle: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, namespace: str, identifier: str) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, namespace: str, identifier: str) -> None:
        raise NotImplementedError


class InMemoryTokenVault(TokenVault):
    """Non-persistent vault for tests and explicitly ephemeral sessions."""

    def __init__(self):
        self._tokens: Dict[str, Dict[str, Any]] = {}

    @property
    def available(self) -> bool:
        return True

    @staticmethod
    def _key(namespace: str, identifier: str) -> str:
        return f'{namespace}:{identifier}'

    def store(self, namespace: str, identifier: str, token_bundle: Dict[str, Any]) -> None:
        self._tokens[self._key(namespace, identifier)] = dict(token_bundle)

    def retrieve(self, namespace: str, identifier: str) -> Optional[Dict[str, Any]]:
        value = self._tokens.get(self._key(namespace, identifier))
        return dict(value) if value is not None else None

    def delete(self, namespace: str, identifier: str) -> None:
        self._tokens.pop(self._key(namespace, identifier), None)


class WindowsDpapiTokenVault(TokenVault):
    """Persist secrets encrypted for the current Windows user through DPAPI."""

    def __init__(self, root: Optional[str] = None):
        self.root = Path(
            root or os.getenv('PUBLISH_CREDENTIAL_DIR', './.opuslabs/credentials')
        ).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def available(self) -> bool:
        return os.name == 'nt'

    @staticmethod
    def _protect(data: bytes) -> bytes:
        if os.name != 'nt':
            raise SecureStorageUnavailable('Windows DPAPI is unavailable on this operating system.')
        import ctypes
        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [
                ('cbData', wintypes.DWORD),
                ('pbData', ctypes.POINTER(ctypes.c_ubyte)),
            ]

        source_buffer = ctypes.create_string_buffer(data)
        source_blob = DataBlob(
            len(data),
            ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        output_blob = DataBlob()
        crypt32 = ctypes.windll.crypt32
        if not crypt32.CryptProtectData(
            ctypes.byref(source_blob),
            'OpusLabs publishing token',
            None,
            None,
            None,
            0x01,
            ctypes.byref(output_blob),
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output_blob.pbData)

    @staticmethod
    def _unprotect(data: bytes) -> bytes:
        if os.name != 'nt':
            raise SecureStorageUnavailable('Windows DPAPI is unavailable on this operating system.')
        import ctypes
        from ctypes import wintypes

        class DataBlob(ctypes.Structure):
            _fields_ = [
                ('cbData', wintypes.DWORD),
                ('pbData', ctypes.POINTER(ctypes.c_ubyte)),
            ]

        source_buffer = ctypes.create_string_buffer(data)
        source_blob = DataBlob(
            len(data),
            ctypes.cast(source_buffer, ctypes.POINTER(ctypes.c_ubyte)),
        )
        output_blob = DataBlob()
        crypt32 = ctypes.windll.crypt32
        if not crypt32.CryptUnprotectData(
            ctypes.byref(source_blob),
            None,
            None,
            None,
            None,
            0x01,
            ctypes.byref(output_blob),
        ):
            raise ctypes.WinError()
        try:
            return ctypes.string_at(output_blob.pbData, output_blob.cbData)
        finally:
            ctypes.windll.kernel32.LocalFree(output_blob.pbData)

    def _path(self, namespace: str, identifier: str) -> Path:
        digest = hashlib.sha256(f'{namespace}:{identifier}'.encode('utf-8')).hexdigest()
        return self.root / f'{digest}.token'

    def store(self, namespace: str, identifier: str, token_bundle: Dict[str, Any]) -> None:
        if not self.available:
            raise SecureStorageUnavailable('Windows DPAPI token storage is unavailable.')
        payload = json.dumps(token_bundle, ensure_ascii=False).encode('utf-8')
        encrypted = self._protect(payload)
        destination = self._path(namespace, identifier)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode='wb',
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            ) as handle:
                handle.write(encrypted)
                temporary_path = Path(handle.name)
            os.replace(temporary_path, destination)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()

    def retrieve(self, namespace: str, identifier: str) -> Optional[Dict[str, Any]]:
        path = self._path(namespace, identifier)
        if not path.exists():
            return None
        decrypted = self._unprotect(path.read_bytes())
        payload = json.loads(decrypted.decode('utf-8'))
        if not isinstance(payload, dict):
            raise ValueError('Decrypted token bundle must be a JSON object.')
        return payload

    def delete(self, namespace: str, identifier: str) -> None:
        self._path(namespace, identifier).unlink(missing_ok=True)


class KeyringTokenVault(TokenVault):
    """Use the operating system keyring through the optional keyring package."""

    def __init__(self, service_name: str = 'OpusLabs Publishing'):
        self.service_name = service_name
        try:
            import keyring
        except ImportError:
            keyring = None
        self._keyring = keyring

    @property
    def available(self) -> bool:
        return self._keyring is not None

    @staticmethod
    def _username(namespace: str, identifier: str) -> str:
        return f'{namespace}:{identifier}'

    def store(self, namespace: str, identifier: str, token_bundle: Dict[str, Any]) -> None:
        if not self.available:
            raise SecureStorageUnavailable('The keyring package is not installed.')
        self._keyring.set_password(
            self.service_name,
            self._username(namespace, identifier),
            json.dumps(token_bundle, ensure_ascii=False),
        )

    def retrieve(self, namespace: str, identifier: str) -> Optional[Dict[str, Any]]:
        if not self.available:
            raise SecureStorageUnavailable('The keyring package is not installed.')
        value = self._keyring.get_password(
            self.service_name,
            self._username(namespace, identifier),
        )
        return json.loads(value) if value else None

    def delete(self, namespace: str, identifier: str) -> None:
        if not self.available:
            raise SecureStorageUnavailable('The keyring package is not installed.')
        try:
            self._keyring.delete_password(
                self.service_name,
                self._username(namespace, identifier),
            )
        except self._keyring.errors.PasswordDeleteError:
            pass


class UnavailableTokenVault(TokenVault):
    @property
    def available(self) -> bool:
        return False

    @staticmethod
    def _raise() -> None:
        raise SecureStorageUnavailable(
            'No secure persistent token vault is available. Install keyring or use Windows DPAPI.'
        )

    def store(self, namespace: str, identifier: str, token_bundle: Dict[str, Any]) -> None:
        self._raise()

    def retrieve(self, namespace: str, identifier: str) -> Optional[Dict[str, Any]]:
        self._raise()

    def delete(self, namespace: str, identifier: str) -> None:
        self._raise()


def create_default_token_vault() -> TokenVault:
    if os.name == 'nt':
        return WindowsDpapiTokenVault()
    keyring_vault = KeyringTokenVault()
    if keyring_vault.available:
        return keyring_vault
    return UnavailableTokenVault()
