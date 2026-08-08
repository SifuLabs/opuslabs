import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from main import VideoEditingCopilot
from src.job_manager import JobCancelled, JobManager


class JobManagerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.database_path = self.root / 'jobs.sqlite3'
        self.workspace_root = self.root / 'workspaces'
        self.manager = JobManager(
            str(self.database_path),
            str(self.workspace_root),
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def make_sources(self, count=2):
        sources = []
        for index in range(count):
            source = self.root / f'source-{index}.mp4'
            source.touch()
            sources.append(str(source))
        return sources

    def enqueue(self, source_count=1, max_attempts=3):
        project = self.manager.create_project('Test project', self.make_sources(source_count))
        jobs = self.manager.enqueue_project(
            project.id,
            'Create one clean clip',
            {'clip_count': 1, 'caption_theme': 'clean'},
            max_attempts=max_attempts,
        )
        return project, jobs

    def test_persists_multi_source_projects_with_isolated_workspaces(self):
        project, jobs = self.enqueue(source_count=2)

        reopened = JobManager(str(self.database_path), str(self.workspace_root))
        persisted_project = reopened.get_project(project.id)
        persisted_jobs = reopened.list_jobs(project_id=project.id)

        self.assertEqual(persisted_project.name, 'Test project')
        self.assertEqual(len(reopened.list_sources(project.id)), 2)
        self.assertEqual(len(persisted_jobs), 2)
        self.assertNotEqual(jobs[0].workspace_path, jobs[1].workspace_path)
        for job in persisted_jobs:
            directories = reopened.workspace_directories(job.id)
            self.assertTrue(Path(directories['temp']).is_dir())
            self.assertTrue(Path(directories['output']).is_dir())
            self.assertEqual(Path(directories['root']).parent.name, project.id)

    def test_atomic_claims_do_not_give_one_job_to_two_workers(self):
        _, jobs = self.enqueue(source_count=4)

        with ThreadPoolExecutor(max_workers=4) as executor:
            claimed = list(executor.map(lambda _index: self.manager.claim_next_job(), range(4)))

        claimed_ids = [job.id for job in claimed if job]
        self.assertEqual(len(claimed_ids), len(jobs))
        self.assertEqual(len(set(claimed_ids)), len(jobs))
        self.assertTrue(all(job.state == 'running' for job in claimed))
        self.assertTrue(all(job.attempts == 1 for job in claimed))

    def test_cancellation_and_retry_are_persisted(self):
        _, jobs = self.enqueue(max_attempts=3)
        queued = jobs[0]

        cancelled = self.manager.request_cancel(queued.id)
        self.assertEqual(cancelled.state, 'cancelled')
        retried = self.manager.retry_job(queued.id)
        self.assertEqual(retried.state, 'queued')

        running = self.manager.claim_next_job()
        self.manager.request_cancel(running.id)
        with self.assertRaises(JobCancelled):
            self.manager.update_progress(running.id, 50, 'Rendering')
        cancelled = self.manager.mark_cancelled(running.id)
        self.assertEqual(cancelled.state, 'cancelled')
        self.assertEqual(cancelled.attempts, 1)
        self.assertEqual(self.manager.retry_job(running.id).state, 'queued')

    def test_recovery_requeues_interrupted_jobs_and_keeps_checkpoints(self):
        _, jobs = self.enqueue(max_attempts=3)
        running = self.manager.claim_next_job()
        checkpoint_path = self.manager.write_checkpoint(
            running.id,
            'transcript',
            {'segments': [{'start': 0, 'end': 1, 'text': 'Hello'}]},
        )
        self.assertTrue(Path(checkpoint_path).exists())

        reopened = JobManager(str(self.database_path), str(self.workspace_root))
        recovery = reopened.recover_interrupted_jobs(force=True)
        recovered_job = reopened.get_job(jobs[0].id)

        self.assertEqual(recovery, {'requeued': 1, 'failed': 0, 'cancelled': 0})
        self.assertEqual(recovered_job.state, 'queued')
        self.assertEqual(reopened.read_checkpoint(running.id, 'transcript')['segments'][0]['text'], 'Hello')
        self.assertEqual(reopened.claim_next_job().attempts, 2)

    def test_retry_limit_and_fresh_worker_recovery_guard(self):
        _, jobs = self.enqueue(max_attempts=2)
        first_attempt = self.manager.claim_next_job()

        self.assertEqual(
            self.manager.recover_interrupted_jobs(stale_after_seconds=3600),
            {'requeued': 0, 'failed': 0, 'cancelled': 0},
        )
        self.manager.fail_job(first_attempt.id, 'First failure')
        self.manager.retry_job(first_attempt.id)
        second_attempt = self.manager.claim_next_job()
        self.assertEqual(second_attempt.attempts, 2)
        self.manager.fail_job(second_attempt.id, 'Second failure')

        with self.assertRaisesRegex(ValueError, 'exhausted'):
            self.manager.retry_job(jobs[0].id)

    def test_copilot_worker_persists_success_result(self):
        _, jobs = self.enqueue()
        copilot = VideoEditingCopilot.__new__(VideoEditingCopilot)
        copilot.job_manager = self.manager

        def fake_process(_source, _preferences, _request, job_manager, job_id):
            job_manager.update_progress(job_id, 75, 'Rendering')
            output = Path(job_manager.workspace_directories(job_id)['output']) / 'clip.mp4'
            output.touch()
            job_manager.write_checkpoint(
                job_id,
                'clips',
                [{'clip_number': 1, 'output_path': str(output)}],
            )
            return 'Processing complete'

        with mock.patch.object(copilot, '_process_real_video', side_effect=fake_process):
            completed = copilot.run_next_job()

        self.assertEqual(completed.id, jobs[0].id)
        self.assertEqual(completed.state, 'completed')
        self.assertEqual(completed.progress, 100)
        self.assertEqual(completed.result['clips'][0]['clip_number'], 1)
        self.assertTrue(Path(completed.result['output_directory']).is_dir())


if __name__ == '__main__':
    unittest.main()
