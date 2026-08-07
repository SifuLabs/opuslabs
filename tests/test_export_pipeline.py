import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from main import VideoEditingCopilot
from src.clip_generator import ClipGenerator
from src.gemini_analyzer import EngagingSegment


def make_segment() -> EngagingSegment:
    return EngagingSegment(
        start_time=10.0,
        end_time=14.0,
        text='A useful sentence for a short clip.',
        hook='A useful sentence',
        engagement_score=8.0,
        segment_type='hook',
        keywords=['useful', 'sentence'],
        suggested_title='Useful Clip',
        hashtags=['#Shorts'],
        emotion='educational',
        viral_potential=7.0,
        word_segments=[
            {'word': 'A', 'start': 10.0, 'end': 10.2},
            {'word': 'useful', 'start': 10.2, 'end': 10.7},
            {'word': 'sentence', 'start': 10.7, 'end': 11.2},
        ],
    )


class ExportPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.generator = ClipGenerator()
        self.generator.output_dir = self.temp_directory.name
        self.generator.target_width = 180
        self.generator.target_height = 320

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_writes_editable_word_timed_subtitles(self):
        output_path = str(Path(self.temp_directory.name) / 'clip.mp4')
        subtitle_path = self.generator._write_subtitles(make_segment(), output_path)

        self.assertIsNotNone(subtitle_path)
        subtitle_text = Path(subtitle_path).read_text(encoding='utf-8')
        self.assertIn('00:00:00,000 --> 00:00:01,200', subtitle_text)
        self.assertIn('A useful sentence', subtitle_text)

    def test_exports_json_and_csv_manifest(self):
        clips = [{
            'clip_number': 1,
            'title': 'Useful Clip',
            'start_time': 10.0,
            'end_time': 14.0,
            'duration': 4.0,
            'engagement_score': 8.0,
            'output_path': 'clip.mp4',
            'hashtags': ['#Shorts'],
            'content_package': {
                'short_caption': 'Watch this.',
                'hashtags': ['#Shorts', '#Useful'],
            },
        }]

        paths = self.generator.export_clip_manifest(
            clips,
            {'platform': 'youtube_shorts', 'reframe_mode': 'blur'},
        )

        payload = json.loads(Path(paths['json']).read_text(encoding='utf-8'))
        self.assertEqual(payload['platform'], 'youtube_shorts')
        self.assertEqual(payload['clips'][0]['title'], 'Useful Clip')
        self.assertTrue(Path(paths['csv']).exists())

    def test_ffmpeg_modes_and_caption_toggle(self):
        captured_commands = []

        def fake_run(command, **_kwargs):
            captured_commands.append(command)
            Path(command[-1]).touch()
            return mock.Mock(returncode=0, stderr='')

        with mock.patch('src.clip_generator.subprocess.run', side_effect=fake_run):
            for mode in ('blur', 'crop', 'fit'):
                output_path = str(Path(self.temp_directory.name) / f'{mode}.mp4')
                succeeded = self.generator._use_ffmpeg_processing(
                    'source.mp4',
                    make_segment(),
                    output_path,
                    {'reframe_mode': mode, 'add_captions': False},
                )
                self.assertTrue(succeeded)

        blur_command, crop_command, fit_command = captured_commands
        self.assertIn('gblur=sigma=30', blur_command[blur_command.index('-filter_complex') + 1])
        self.assertNotIn('drawtext', ' '.join(blur_command))
        self.assertIn('force_original_aspect_ratio=increase', crop_command[crop_command.index('-vf') + 1])
        self.assertIn('pad=', fit_command[fit_command.index('-vf') + 1])

    def test_natural_language_export_preferences(self):
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.style_keywords = {}
        preferences = copilot._parse_user_preferences(
            'Make 2 YouTube Shorts with no captions and show full frame'
        )

        self.assertEqual(preferences['clip_count'], 2)
        self.assertEqual(preferences['platform'], 'youtube_shorts')
        self.assertFalse(preferences['add_captions'])
        self.assertEqual(preferences['reframe_mode'], 'fit')


if __name__ == '__main__':
    unittest.main()
