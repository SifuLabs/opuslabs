"""Optional Gemini-backed subtitle translation with schema validation."""

import json
import os
import re
from typing import List, Optional, Sequence, Tuple


LANGUAGE_ALIASES = {
    'arabic': 'ar',
    'chinese': 'zh',
    'english': 'en',
    'french': 'fr',
    'german': 'de',
    'hindi': 'hi',
    'italian': 'it',
    'japanese': 'ja',
    'korean': 'ko',
    'portuguese': 'pt',
    'spanish': 'es',
    'swahili': 'sw',
}


def normalize_language(value: str) -> Optional[str]:
    """Normalize a language name or short BCP-47-like code."""
    candidate = str(value or '').strip().lower()
    if candidate in LANGUAGE_ALIASES:
        return LANGUAGE_ALIASES[candidate]
    if re.fullmatch(r'[a-z]{2,3}(?:-[a-z0-9]{2,8})?', candidate):
        return candidate
    return None


class SubtitleTranslator:
    """Translate subtitle cue text while preserving the caller's cue timings."""

    def __init__(self):
        self.client = None
        self.types = None
        api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            return
        self.client = genai.Client(api_key=api_key)
        self.types = genai_types
        self.model = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')

    @property
    def available(self) -> bool:
        return self.client is not None and self.types is not None

    def translate_cues(
        self,
        cues: Sequence[Tuple[float, float, str]],
        target_language: str,
    ) -> Optional[List[str]]:
        language = normalize_language(target_language)
        if not language or not cues or not self.available:
            return None
        source_texts = [text for _, _, text in cues]
        schema = {
            'type': 'object',
            'properties': {
                'translations': {
                    'type': 'array',
                    'items': {'type': 'string'},
                },
            },
            'required': ['translations'],
        }
        prompt = (
            f'Translate each subtitle cue into language code {language}. '
            'Keep names, meaning, tone, and concise subtitle phrasing. '
            'Return exactly one translation for each input cue in the same order.\n'
            f'Input cues: {json.dumps(source_texts, ensure_ascii=False)}'
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=self.types.GenerateContentConfig(
                    response_mime_type='application/json',
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            parsed = getattr(response, 'parsed', None)
            if not isinstance(parsed, dict):
                parsed = json.loads(response.text)
            translations = parsed.get('translations', [])
            if len(translations) != len(cues) or not all(isinstance(item, str) for item in translations):
                return None
            return [translation.strip() for translation in translations]
        except Exception as error:
            print(f'Subtitle translation to {language} failed: {error}')
            return None

