"""Safe transcript correction helpers used before clip selection."""

import copy
import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple


def load_transcript_corrections(path: str) -> Dict[str, str]:
    """Load replacement pairs from a compact JSON mapping or replacements list."""
    correction_path = Path(path).expanduser()
    if not correction_path.is_file():
        raise ValueError(f'Transcript correction file was not found: {correction_path}')
    if correction_path.suffix.lower() != '.json':
        raise ValueError('Transcript corrections must be supplied as a JSON file.')
    if correction_path.stat().st_size > 1_000_000:
        raise ValueError('Transcript correction files must be smaller than 1 MB.')
    try:
        payload = json.loads(correction_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f'Could not read transcript corrections: {error}') from error

    if isinstance(payload, dict) and isinstance(payload.get('replacements'), list):
        replacements = {
            str(item.get('from', '')).strip(): str(item.get('to', '')).strip()
            for item in payload['replacements']
            if isinstance(item, dict) and str(item.get('from', '')).strip()
        }
    elif isinstance(payload, dict):
        replacements = {
            str(source).strip(): str(target).strip()
            for source, target in payload.items()
            if str(source).strip() and isinstance(target, (str, int, float))
        }
    else:
        raise ValueError('Transcript correction JSON must be an object or contain a replacements list.')

    if not replacements:
        raise ValueError('Transcript correction file did not contain any replacement pairs.')
    if len(replacements) > 200:
        raise ValueError('Transcript correction files may contain at most 200 replacements.')
    return replacements


def apply_transcript_corrections(
    transcript: Dict[str, Any],
    replacements: Mapping[str, str],
) -> Tuple[Dict[str, Any], int]:
    """Return a corrected transcript copy and the number of replacements made."""
    corrected = copy.deepcopy(transcript)
    clean_replacements = [
        (str(source).strip(), str(target).strip())
        for source, target in replacements.items()
        if str(source).strip()
    ]
    total_replacements = 0

    for segment in corrected.get('segments', []):
        text = str(segment.get('text', ''))
        phrase_replaced = False
        for source, target in clean_replacements:
            pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
            text, count = pattern.subn(lambda _match, replacement=target: replacement, text)
            total_replacements += count
            phrase_replaced = phrase_replaced or (count > 0 and bool(re.search(r'\s', source)))
        segment['text'] = text

        words = segment.get('words') or []
        if phrase_replaced:
            # A phrase can change word count, so static cue timing is safer than
            # attaching corrected text to inaccurate word boundaries.
            segment['words'] = []
            continue
        for word in words:
            word_text = str(word.get('word', ''))
            for source, target in clean_replacements:
                pattern = re.compile(re.escape(source), flags=re.IGNORECASE)
                word_text = pattern.sub(lambda _match, replacement=target: replacement, word_text)
            word['word'] = word_text

    corrected['text'] = ' '.join(
        str(segment.get('text', '')).strip()
        for segment in corrected.get('segments', [])
        if str(segment.get('text', '')).strip()
    )
    corrected['corrections_applied'] = total_replacements
    return corrected, total_replacements

