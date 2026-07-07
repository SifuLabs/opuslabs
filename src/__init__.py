"""
Video Editing Copilot Agent

A conversational AI agent that transforms long-form videos into engaging short-form clips.
"""

__version__ = "1.0.0"
__author__ = "Video Editing Copilot"

def __getattr__(name):
    """Load optional-heavy modules only when they are requested."""
    if name == 'VideoProcessor':
        from .video_processor import VideoProcessor
        return VideoProcessor
    if name == 'TranscriptAnalyzer':
        from .gemini_analyzer import GeminiTranscriptAnalyzer
        return GeminiTranscriptAnalyzer
    if name == 'ClipGenerator':
        from .clip_generator import ClipGenerator
        return ClipGenerator
    if name == 'ConversationalInterface':
        from .user_interface import ConversationalInterface
        return ConversationalInterface
    if name == 'ContentStrategyBuilder':
        from .content_strategy import ContentStrategyBuilder
        return ContentStrategyBuilder
    if name == 'config':
        from .config import config
        return config
    raise AttributeError(f"module 'src' has no attribute {name!r}")

__all__ = [
    'VideoProcessor',
    'TranscriptAnalyzer', 
    'ClipGenerator',
    'ConversationalInterface',
    'ContentStrategyBuilder',
    'config'
]
