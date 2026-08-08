import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from main import VideoEditingCopilot
from src.brand_kits import BrandKitStore
from src.clip_generator import ClipGenerator
from src.gemini_analyzer import EngagingSegment
from src.transcript_tools import apply_transcript_corrections, load_transcript_corrections


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

    def test_writes_translated_subtitles_with_original_timings(self):
        output_path = str(Path(self.temp_directory.name) / 'clip.mp4')
        translator = mock.Mock(available=True)
        translator.translate_cues.side_effect = lambda cues, language: [
            f'{language}: {cue[2]}' for cue in cues
        ]
        self.generator.subtitle_translator = translator

        paths = self.generator._write_translated_subtitles(
            make_segment(),
            output_path,
            ['Spanish', 'fr', 'not-a-language'],
        )

        self.assertEqual(set(paths), {'es', 'fr'})
        spanish_text = Path(paths['es']).read_text(encoding='utf-8')
        self.assertIn('00:00:00,000 --> 00:00:01,200', spanish_text)
        self.assertIn('es: A useful sentence', spanish_text)

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

    def test_smart_reframe_uses_detected_subject_focus(self):
        captured_commands = []

        def fake_run(command, **_kwargs):
            captured_commands.append(command)
            Path(command[-1]).touch()
            return mock.Mock(returncode=0, stderr='')

        output_path = str(Path(self.temp_directory.name) / 'smart.mp4')
        with mock.patch.object(
            self.generator,
            '_detect_subject_track',
            return_value=[(0.0, 0.25, 0.40), (2.0, 0.75, 0.60)],
        ) as detect_subject, mock.patch(
            'src.clip_generator.subprocess.run',
            side_effect=fake_run,
        ):
            succeeded = self.generator._use_ffmpeg_processing(
                'source.mp4',
                make_segment(),
                output_path,
                {'reframe_mode': 'smart', 'add_captions': False},
            )

        self.assertTrue(succeeded)
        detect_subject.assert_called_once()
        video_filter = captured_commands[0][captured_commands[0].index('-vf') + 1]
        self.assertIn('between(t,0.000,2.000)', video_filter)
        self.assertIn('0.2500+(0.5000)', video_filter)
        self.assertIn('0.4000+(0.2000)', video_filter)

    def test_split_screen_and_image_logo_filters(self):
        captured_commands = []

        def fake_run(command, **_kwargs):
            captured_commands.append(command)
            Path(command[-1]).touch()
            return mock.Mock(returncode=0, stderr='')

        logo_path = Path(self.temp_directory.name) / 'logo.png'
        logo_path.touch()
        output_path = str(Path(self.temp_directory.name) / 'split.mp4')
        with mock.patch('src.clip_generator.subprocess.run', side_effect=fake_run):
            succeeded = self.generator._use_ffmpeg_processing(
                'source.mp4',
                make_segment(),
                output_path,
                {
                    'reframe_mode': 'split',
                    'add_captions': False,
                    'brand_logo': str(logo_path),
                    'brand_logo_position': 'bottom-left',
                },
            )

        self.assertTrue(succeeded)
        command = captured_commands[0]
        self.assertIn(str(logo_path.resolve()), command)
        video_filter = command[command.index('-filter_complex') + 1]
        self.assertIn('vstack=inputs=2', video_filter)
        self.assertIn('[1:v]scale=', video_filter)
        self.assertIn('overlay=16:H-h-16', video_filter)

    def test_caption_themes_safe_position_color_size_and_brand_label(self):
        segment = make_segment()
        segment.word_segments = [
            {'word': f'word{index}', 'start': 10 + index * 0.2, 'end': 10.2 + index * 0.2}
            for index in range(7)
        ]

        bold = self.generator._build_caption_filters(segment, 4.0, {'caption_theme': 'bold'})
        clean = self.generator._build_caption_filters(segment, 4.0, {'caption_theme': 'clean'})
        minimal = self.generator._build_caption_filters(
            segment,
            4.0,
            {
                'caption_theme': 'minimal',
                'caption_position': 'top',
                'caption_color': '0xffaa00',
                'caption_font_size': 64,
            },
        )

        self.assertEqual(len(bold), 3)
        self.assertEqual(len(clean), 2)
        self.assertEqual(len(minimal), 1)
        self.assertIn('fontsize=64', minimal[0])
        self.assertIn('fontcolor=0xffaa00', minimal[0])
        self.assertIn(':y=h*0.16', minimal[0])

        brand = self.generator._build_brand_filter({'brand_label': "Opus:Labs's"})
        self.assertIn("text='OPUS\\:LABS’S'", brand)
        self.assertIn(':x=40:y=54:', brand)

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

    def test_natural_language_brand_and_caption_controls(self):
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.style_keywords = {}
        preferences = copilot._parse_user_preferences(
            'Use smart crop with minimal captions at the top, '
            'caption color #ffaa00, caption size 64, and watermark "Opus Labs"'
        )

        self.assertEqual(preferences['reframe_mode'], 'smart')
        self.assertEqual(preferences['caption_theme'], 'minimal')
        self.assertEqual(preferences['caption_position'], 'top')
        self.assertEqual(preferences['caption_color'], '0xffaa00')
        self.assertEqual(preferences['caption_font_size'], 64)
        self.assertEqual(preferences['brand_label'], 'Opus Labs')

    def test_natural_language_split_layout_and_logo(self):
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.style_keywords = {}
        preferences = copilot._parse_user_preferences(
            'Use a split-screen conversation layout with logo "assets/mark.png" '
            'and put the logo bottom right'
        )

        self.assertEqual(preferences['reframe_mode'], 'split')
        self.assertEqual(preferences['brand_logo'], 'assets/mark.png')
        self.assertEqual(preferences['brand_logo_position'], 'bottom-right')

    def test_natural_language_transcript_and_translation_controls(self):
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.style_keywords = {}
        preferences = copilot._parse_user_preferences(
            'Replace "Open Eye" with "OpenAI" and create subtitles in Spanish and Swahili'
        )

        self.assertEqual(preferences['transcript_corrections'], {'Open Eye': 'OpenAI'})
        self.assertEqual(preferences['subtitle_languages'], ['es', 'sw'])

    def test_transcript_corrections_update_analysis_and_word_text(self):
        transcript = {
            'text': 'Welcome to Opus Lab.',
            'segments': [{
                'start': 0.0,
                'end': 2.0,
                'text': 'Welcome to Opus Lab.',
                'words': [
                    {'word': 'Welcome', 'start': 0.0, 'end': 0.5},
                    {'word': 'Opus', 'start': 0.5, 'end': 1.0},
                    {'word': 'Lab.', 'start': 1.0, 'end': 1.5},
                ],
            }],
        }

        corrected, count = apply_transcript_corrections(transcript, {'Lab': 'Labs'})

        self.assertEqual(count, 1)
        self.assertEqual(corrected['text'], 'Welcome to Opus Labs.')
        self.assertEqual(corrected['segments'][0]['words'][-1]['word'], 'Labs.')
        self.assertEqual(transcript['segments'][0]['words'][-1]['word'], 'Lab.')

    def test_loads_structured_transcript_correction_file(self):
        correction_path = Path(self.temp_directory.name) / 'corrections.json'
        correction_path.write_text(
            json.dumps({'replacements': [{'from': 'Open Eye', 'to': 'OpenAI'}]}),
            encoding='utf-8',
        )

        self.assertEqual(
            load_transcript_corrections(str(correction_path)),
            {'Open Eye': 'OpenAI'},
        )

    def test_brand_kits_persist_and_apply_with_explicit_overrides(self):
        kit_path = Path(self.temp_directory.name) / 'brand-kits.json'
        store = BrandKitStore(str(kit_path))
        store.save('Studio', {
            'caption_theme': 'clean',
            'caption_color': 'yellow',
            'brand_label': 'Opus Labs',
            'platform': 'ignored',
        })

        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.brand_kits = store
        resolved = copilot._apply_brand_kit_preferences({
            'use_brand_kit': 'studio',
            'caption_color': 'white',
        })

        self.assertEqual(store.list_names(), ['Studio'])
        self.assertEqual(resolved['caption_theme'], 'clean')
        self.assertEqual(resolved['caption_color'], 'white')
        self.assertEqual(resolved['brand_label'], 'Opus Labs')
        self.assertEqual(resolved['brand_kit'], 'studio')


if __name__ == '__main__':
    unittest.main()
