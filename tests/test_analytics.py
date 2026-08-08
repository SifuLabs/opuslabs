import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from main import VideoEditingCopilot, _run_management_cli
from src.analytics import AnalyticsService, AnalyticsStore
from src.gemini_analyzer import EngagingSegment
from src.publication_store import PublicationStore


class AnalyticsFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.analytics_store = AnalyticsStore(str(self.root / 'analytics.sqlite3'))
        self.publication_store = PublicationStore(str(self.root / 'publishing.sqlite3'))
        self.service = AnalyticsService(self.analytics_store, self.publication_store)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_json(self, name, records):
        path = self.root / name
        path.write_text(json.dumps({'records': records}), encoding='utf-8')
        return path

    def make_publication(self):
        clip = self.root / 'published-clip.mp4'
        clip.write_bytes(b'published-video')
        approval = self.publication_store.create_approval(str(clip), 'reviewer')
        publication = self.publication_store.create_publication({
            'provider': 'local',
            'platform': 'youtube_shorts',
            'mode': 'draft',
            'clip_path': str(clip),
            'clip_sha256': approval.clip_sha256,
            'approval_id': approval.id,
            'account_id': None,
            'title': 'Observed clip',
            'caption': '',
            'hashtags': [],
            'privacy': 'private',
            'scheduled_at': None,
            'idempotency_key': 'analytics-test-publication',
        })
        return clip, publication

    @staticmethod
    def record(**overrides):
        payload = {
            'clip_sha256': 'a' * 64,
            'provider': 'manual',
            'platform': 'youtube_shorts',
            'style': 'educational',
            'segment_type': 'explanation',
            'keywords': ['workflow', 'tip'],
            'predicted_engagement_score': 8.0,
            'views': 1000,
            'engaged_views': 650,
            'retention_percent': 62,
            'shares': 30,
            'conversions': 10,
            'observed_at': '2026-08-01T12:00:00Z',
        }
        payload.update(overrides)
        return payload

    def test_json_import_links_publication_normalizes_and_is_idempotent(self):
        _clip, publication = self.make_publication()
        source = self.write_json('results.json', [{
            'publication_id': publication.id,
            'style': 'Educational',
            'segment_type': 'Explanation',
            'keywords': 'workflow, tips',
            'predicted_engagement': 7.8,
            'views': 1250,
            'engaged_views': 800,
            'retention': 0.64,
            'shares': 42,
            'conversions': 13,
            'observed_at': '2026-08-02T09:30:00Z',
        }])

        first = self.service.import_file(str(source))
        second = self.service.import_file(str(source))
        observations = self.analytics_store.list_observations()

        self.assertFalse(first.already_imported)
        self.assertTrue(second.already_imported)
        self.assertEqual(first.import_record.id, second.import_record.id)
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation.publication_id, publication.id)
        self.assertEqual(observation.clip_sha256, publication.clip_sha256)
        self.assertEqual(observation.platform, 'youtube_shorts')
        self.assertEqual(observation.provider, 'local')
        self.assertEqual(observation.retention_percent, 64.0)
        self.assertAlmostEqual(observation.engaged_view_rate, 0.64)
        self.assertIsNotNone(observation.to_dict()['prediction_delta'])

    def test_csv_manifest_import_enriches_candidate_context(self):
        clip = self.root / 'manifest-clip.mp4'
        clip.write_bytes(b'manifest-video')
        manifest = self.root / 'clip_manifest.json'
        manifest.write_text(json.dumps({
            'schema_version': 1,
            'platform': 'tiktok',
            'style': 'funny',
            'clips': [{
                'clip_number': 1,
                'output_path': str(clip),
                'engagement_score': 7.4,
                'segment_type': 'story',
                'keywords': ['launch', 'mistake'],
            }],
        }), encoding='utf-8')
        source = self.root / 'manifest-results.csv'
        with source.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                'manifest_path', 'clip_number', 'views', 'engaged_views',
                'retention_percent', 'shares', 'conversions',
            ])
            writer.writeheader()
            writer.writerow({
                'manifest_path': manifest.name,
                'clip_number': 1,
                'views': 500,
                'engaged_views': 275,
                'retention_percent': 58,
                'shares': 15,
                'conversions': 4,
            })

        self.service.import_file(str(source))
        observation = self.analytics_store.list_observations()[0]

        self.assertEqual(observation.platform, 'tiktok')
        self.assertEqual(observation.style, 'funny')
        self.assertEqual(observation.segment_type, 'story')
        self.assertEqual(observation.keywords, ['launch', 'mistake'])
        self.assertEqual(observation.predicted_engagement_score, 7.4)
        self.assertIsNotNone(observation.clip_sha256)

    def test_invalid_batch_rolls_back_all_observations(self):
        source = self.write_json('invalid.json', [
            self.record(),
            self.record(clip_sha256='b' * 64, views=100, engaged_views=101),
        ])

        with self.assertRaisesRegex(ValueError, 'row 2.*cannot exceed'):
            self.service.import_file(str(source))

        self.assertEqual(self.analytics_store.observation_count(), 0)
        self.assertEqual(self.analytics_store.list_imports(), [])

    def test_report_compares_predictions_and_cites_observed_performance(self):
        records = [
            self.record(
                clip_sha256='1' * 64,
                style='educational',
                views=2000,
                engaged_views=1600,
                retention_percent=82,
                shares=100,
                conversions=40,
                predicted_engagement_score=7.5,
            ),
            self.record(
                clip_sha256='2' * 64,
                style='educational',
                views=1000,
                engaged_views=750,
                retention_percent=75,
                shares=40,
                conversions=15,
                predicted_engagement_score=7.0,
            ),
            self.record(
                clip_sha256='3' * 64,
                style='funny',
                segment_type='story',
                views=500,
                engaged_views=100,
                retention_percent=24,
                shares=2,
                conversions=0,
                predicted_engagement_score=8.5,
            ),
        ]
        self.service.import_file(str(self.write_json('report.json', records)))

        report = self.service.build_report(platform='youtube_shorts')

        self.assertEqual(report['summary']['observation_count'], 3)
        self.assertEqual(report['summary']['total_views'], 3500)
        self.assertEqual(
            report['prediction_comparison']['matched_observation_count'], 3
        )
        self.assertGreater(report['prediction_comparison']['mean_absolute_error'], 0)
        recommendation = next(
            item for item in report['recommendations']
            if item['type'] == 'style_on_platform'
        )
        self.assertEqual(recommendation['style'], 'educational')
        self.assertIn('3,000 views across 2 result(s)', recommendation['message'])
        self.assertIn('% retention', recommendation['message'])
        self.assertEqual(recommendation['evidence']['total_views'], 3000)

    def test_historical_outcomes_rerank_future_candidates_without_mutation(self):
        records = []
        for index in range(2):
            records.append(self.record(
                clip_sha256=f'{index + 1:x}' * 64,
                style='educational',
                segment_type='explanation',
                keywords=['workflow'],
                views=1000,
                engaged_views=900,
                retention_percent=90,
                shares=50,
                conversions=20,
            ))
        for index in range(2):
            records.append(self.record(
                clip_sha256=f'{index + 10:x}'[0] * 64,
                style='funny',
                segment_type='story',
                keywords=['mistake'],
                views=1000,
                engaged_views=100,
                retention_percent=20,
                shares=0,
                conversions=0,
            ))
        self.service.import_file(str(self.write_json('history.json', records)))
        funny = self.make_segment(
            title='Funny candidate',
            score=8.0,
            emotion='funny',
            segment_type='story',
            keywords=['mistake'],
        )
        educational = self.make_segment(
            title='Educational candidate',
            score=7.0,
            emotion='educational',
            segment_type='explanation',
            keywords=['workflow'],
        )

        ranked, evidence = self.service.rerank_segments(
            [funny, educational],
            platform='youtube_shorts',
            requested_style='engaging',
        )

        self.assertTrue(evidence['applied'])
        self.assertEqual(ranked[0].suggested_title, 'Educational candidate')
        self.assertEqual(funny.engagement_score, 8.0)
        self.assertEqual(educational.engagement_score, 7.0)
        self.assertEqual(self.service.candidate_pool_size(2, 'youtube_shorts'), 4)
        top_evidence = evidence['candidates'][0]
        self.assertGreater(top_evidence['historical_adjustment'], 0)
        self.assertTrue(top_evidence['evidence'])

        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.analytics_service = self.service
        selected, integration_evidence = copilot._rerank_clip_candidates(
            [funny, educational],
            {'platform': 'youtube_shorts', 'style': 'engaging'},
            1,
        )
        self.assertEqual(selected[0].suggested_title, 'Educational candidate')
        self.assertTrue(integration_evidence['applied'])

    def test_analytics_cli_import_report_and_schema(self):
        source = self.write_json('cli.json', [self.record()])
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.analytics_service = self.service
        output = io.StringIO()

        with redirect_stdout(output):
            self.assertEqual(
                _run_management_cli(copilot, ['analytics', 'import', str(source)]),
                0,
            )
        imported = json.loads(output.getvalue())
        self.assertEqual(imported['record_count'], 1)

        output = io.StringIO()
        with redirect_stdout(output):
            _run_management_cli(copilot, ['analytics', 'report', '--style', 'educational'])
        report = json.loads(output.getvalue())
        self.assertEqual(report['summary']['observation_count'], 1)

        output = io.StringIO()
        with redirect_stdout(output):
            _run_management_cli(copilot, ['analytics', 'schema'])
        schema = json.loads(output.getvalue())
        self.assertIn('views', schema['required_metrics'])

    @staticmethod
    def make_segment(title, score, emotion, segment_type, keywords):
        return EngagingSegment(
            start_time=0,
            end_time=30,
            text='Test clip',
            hook='Test hook',
            engagement_score=score,
            segment_type=segment_type,
            keywords=keywords,
            suggested_title=title,
            hashtags=['#Test'],
            emotion=emotion,
            viral_potential=score,
        )


if __name__ == '__main__':
    unittest.main()
