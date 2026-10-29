"""Focused tests for EXP93 labelling and configuration guards; no PCAP download."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / 'scripts/v7/exp93/run_pcap_benchmark.py'
SPEC = importlib.util.spec_from_file_location('exp93_runner', MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(RUNNER)


def row(**updates):
    value = dict(src_ip='13.58.98.64', dst_ip='172.31.69.25', src_port=44000, dst_port=22,
                 protocol=6, start_ns=1518631400000000000, end_ns=1518631400100000000,
                 fwd_payload_bytes=120, bwd_payload_bytes=40, fwd_packets=4, bwd_packets=3,
                 fwd_rst_packets=0, bwd_rst_packets=0, flow_duration=100000)
    value.update(updates)
    return value


class Exp93RulesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/experiments/exp93_protocol.json').read_text())
        cls.ssh = next(c for c in cls.config['captures'] if c['id'].endswith('bruteforce'))
        cls.rules = RUNNER.CorrectedRuleSet(cls.ssh, cls.config['label_source'])

    def test_confirmed_and_attempted_ssh_are_disjoint(self):
        direct = self.rules.apply(row())
        attempted = self.rules.apply(row(fwd_payload_bytes=0, bwd_payload_bytes=0))
        self.assertEqual(direct['label'], 'brute_force')
        self.assertEqual(direct['label_status'], 'confirmed_attack')
        self.assertEqual(attempted['label'], 'benign')
        self.assertEqual(attempted['label_status'], 'attempted_to_benign')

    def test_reverse_flow_uses_attack_direction(self):
        reverse = self.rules.apply(row(src_ip='172.31.69.25', dst_ip='13.58.98.64', src_port=22, dst_port=44000,
                                       fwd_payload_bytes=40, bwd_payload_bytes=120))
        self.assertEqual(reverse['label'], 'brute_force')

    def test_interval_crossing_is_not_silently_benign(self):
        boundary = self.rules.apply(row(start_ns=1518631309000000000, end_ns=1518631400000000000))
        self.assertIsNone(boundary['label'])
        self.assertEqual(boundary['label_status'], 'boundary')

    def test_higher_priority_attempted_rule_overrides_bot_direct_annotation(self):
        bot_capture = next(c for c in self.config['captures'] if c['id'].endswith('_bot'))
        bot_rules = RUNNER.CorrectedRuleSet(bot_capture, self.config['label_source'])
        attempted = bot_rules.apply(row(src_ip='172.31.69.13', dst_ip='18.219.211.138', src_port=54000,
                                         dst_port=443, start_ns=1520020430000000000, end_ns=1520020430100000000,
                                         fwd_payload_bytes=100, bwd_payload_bytes=20, fwd_rst_packets=0,
                                         bwd_rst_packets=1))
        self.assertEqual(attempted['annotation_label'], 'Botnet-Ares-Attempted-TearDown')
        self.assertEqual(attempted['label'], 'benign')

    def test_every_target_class_has_a_direct_rule_and_valid_capture(self):
        targets = set(self.config['class_order']) - {'benign'}
        covered = {rule['label'] for capture in self.config['captures'] for rule in capture['rules']
                   if rule.get('label')}
        self.assertSetEqual(targets, covered)
        for capture in self.config['captures']:
            self.assertTrue(capture['path'].startswith('data/pcap/cic2018/'))
            self.assertGreater(capture['max_uncompressed_bytes'], 0)
            for rule in capture['rules']:
                self.assertLessEqual(rule['start_ns'], rule['end_ns'])
                self.assertIn('source', rule)


if __name__ == '__main__':
    unittest.main()
