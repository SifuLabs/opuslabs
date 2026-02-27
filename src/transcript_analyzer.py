"""
Transcript Analysis Module (legacy shim)

This module is superseded by gemini_analyzer.py.
Re-exports GeminiTranscriptAnalyzer as TranscriptAnalyzer for backward compatibility.
"""

from .gemini_analyzer import GeminiTranscriptAnalyzer as TranscriptAnalyzer, EngagingSegment

__all__ = ['TranscriptAnalyzer', 'EngagingSegment']
