"""
Video Editing Copilot Agent

A conversational AI agent that transforms long-form videos into engaging short-form clips.
"""

__version__ = "1.0.0"
__author__ = "Video Editing Copilot"

from .video_processor import VideoProcessor
from .gemini_analyzer import GeminiTranscriptAnalyzer as TranscriptAnalyzer
from .clip_generator import ClipGenerator
from .user_interface import ConversationalInterface
from .config import config

__all__ = [
    'VideoProcessor',
    'TranscriptAnalyzer', 
    'ClipGenerator',
    'ConversationalInterface',
    'config'
]