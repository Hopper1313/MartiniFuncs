"""Heartbeat and stage lifecycle logging without scientific work."""
from threading import Event
import unittest
from MartiniFuncs.workflows.progress import run_stage


class ProgressTests(unittest.TestCase):
    def test_heartbeat_completion_and_quiet(self):
        messages = []
        def work():
            Event().wait(.06)
            return 42
        self.assertEqual(run_stage('test', work, progress=messages.append, heartbeat_seconds=.01), 42)
        self.assertIn('starting', messages[0])
        self.assertTrue(any('still running' in m for m in messages))
        self.assertIn('completed', messages[-1])
        self.assertEqual(run_stage('quiet', lambda: 42, progress=None), 42)

    def test_failure_is_logged_and_propagated(self):
        messages = []
        def fail():
            raise ValueError('bad input')
        with self.assertRaisesRegex(ValueError, 'bad input'):
            run_stage('test', fail, progress=messages.append)
        self.assertIn('failed', messages[-1])
