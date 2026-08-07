import math
import sys
import unittest
from unittest import mock

from main import VideoEditingCopilot
from src.gemini_analyzer import GeminiTranscriptAnalyzer


class AnalysisValidationTests(unittest.TestCase):
    def test_rejects_invalid_and_overlapping_ai_timestamps(self):
        analyzer = GeminiTranscriptAnalyzer.__new__(GeminiTranscriptAnalyzer)
        segments = [
            {'start': 0.0, 'end': 10.0, 'text': 'First segment', 'words': []},
            {'start': 10.0, 'end': 20.0, 'text': 'Second segment', 'words': []},
            {'start': 20.0, 'end': 30.0, 'text': 'Third segment', 'words': []},
        ]
        base = {
            'hook': 'Hook',
            'engagement_score': 8.0,
            'viral_potential': 7.0,
            'emotion': 'educational',
            'segment_type': 'hook',
            'keywords': [],
            'hashtags': [],
        }
        analysis = {'engaging_moments': [
            dict(base, title='Valid', start_time=0.0, end_time=10.0),
            dict(base, title='Overlap', start_time=5.0, end_time=15.0),
            dict(base, title='Not finite', start_time=math.nan, end_time=20.0),
            dict(base, title='Clamped', start_time=20.0, end_time=40.0),
        ]}

        results = analyzer._create_segments_from_ai_analysis(
            segments, analysis, 'educational'
        )

        self.assertEqual([item.suggested_title for item in results], ['Valid', 'Clamped'])
        self.assertEqual(results[-1].end_time, 30.0)

    def test_full_processing_requires_ffmpeg_and_real_transcription(self):
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.config = {'gemini_api_key': None}
        copilot.available_features = {'demo_mode', 'conversation'}
        copilot.moviepy_available = False
        copilot.ffmpeg_available = False

        with mock.patch.object(copilot, '_check_ffmpeg', return_value=True), \
                mock.patch.dict(sys.modules, {'whisper': mock.Mock()}):
            copilot._try_load_advanced_modules()

        self.assertTrue(copilot.ffmpeg_available)
        self.assertIn('transcription', copilot.available_features)
        self.assertIn('full_processing', copilot.available_features)


if __name__ == '__main__':
    unittest.main()
