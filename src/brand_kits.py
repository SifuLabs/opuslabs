"""Persistent, reusable brand settings for clip exports."""

import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, List, Optional


BRAND_SETTING_KEYS = {
    'brand_label',
    'brand_logo',
    'brand_logo_position',
    'brand_logo_width',
    'brand_logo_opacity',
    'caption_theme',
    'caption_position',
    'caption_color',
    'caption_font_size',
}


class BrandKitStore:
    """Store named brand kits in a small atomic JSON document."""

    def __init__(self, path: Optional[str] = None):
        configured_path = path or os.getenv('BRAND_KITS_PATH', './brand_kits.json')
        self.path = Path(configured_path)

    @staticmethod
    def _normalize_name(name: str) -> str:
        normalized = re.sub(r'\s+', ' ', str(name or '').strip())
        if not normalized or len(normalized) > 48:
            raise ValueError('Brand kit names must contain 1 to 48 characters.')
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9 ._-]*', normalized):
            raise ValueError('Brand kit names may use letters, numbers, spaces, dots, dashes, and underscores.')
        return normalized

    def _read(self) -> Dict[str, Dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f'Could not read brand kits from {self.path}: {error}') from error
        if not isinstance(payload, dict):
            raise ValueError(f'Brand kit file must contain a JSON object: {self.path}')
        return {
            str(name): settings
            for name, settings in payload.items()
            if isinstance(settings, dict)
        }

    def list_names(self) -> List[str]:
        return sorted(self._read(), key=str.casefold)

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        normalized = self._normalize_name(name)
        kits = self._read()
        matching_name = next(
            (saved_name for saved_name in kits if saved_name.casefold() == normalized.casefold()),
            None,
        )
        if matching_name is None:
            return None
        return {
            key: value
            for key, value in kits[matching_name].items()
            if key in BRAND_SETTING_KEYS
        }

    def save(self, name: str, settings: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self._normalize_name(name)
        clean_settings = {
            key: value
            for key, value in settings.items()
            if key in BRAND_SETTING_KEYS and value not in (None, '')
        }
        if not clean_settings:
            raise ValueError('A brand kit needs at least one caption, label, or logo setting.')

        kits = self._read()
        existing_name = next(
            (saved_name for saved_name in kits if saved_name.casefold() == normalized.casefold()),
            None,
        )
        if existing_name and existing_name != normalized:
            del kits[existing_name]
        kits[normalized] = clean_settings

        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=self.path.parent,
                prefix=f'.{self.path.name}.',
                suffix='.tmp',
                delete=False,
            ) as handle:
                json.dump(kits, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write('\n')
                temporary_path = Path(handle.name)
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path and temporary_path.exists():
                temporary_path.unlink()
        return dict(clean_settings)

    def delete(self, name: str) -> bool:
        normalized = self._normalize_name(name)
        kits = self._read()
        matching_name = next(
            (saved_name for saved_name in kits if saved_name.casefold() == normalized.casefold()),
            None,
        )
        if matching_name is None:
            return False
        del kits[matching_name]
        if not kits:
            self.path.unlink(missing_ok=True)
            return True
        self.path.write_text(
            json.dumps(kits, ensure_ascii=False, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        return True
