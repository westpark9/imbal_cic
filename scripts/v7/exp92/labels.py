"""Explicit flow-key/time rules; conflicts and uncovered flows stay unresolved."""
from decimal import Decimal
import ipaddress
import json

def ns(value): return int(Decimal(str(value))*1_000_000_000)

class LabelRules:
    def __init__(self, path):
        self.config = json.loads(path.read_text())
        self.rules = self.config['rules']
        for r in self.rules:
            if not r.get('source'): raise ValueError('Every label rule requires a source')
            if ns(r['end']) < ns(r['start']): raise ValueError('Reversed label interval')

    @staticmethod
    def matches_key(row, rule, reverse=False):
        for prefix in ('src', 'dst'):
            actual = ('dst' if prefix == 'src' else 'src') if reverse else prefix
            if prefix+'_ip' in rule and ipaddress.ip_address(row[actual+'_ip']) not in ipaddress.ip_network(rule[prefix+'_ip'], strict=False): return False
            if prefix+'_port' in rule and row[actual+'_port'] != rule[prefix+'_port']: return False
            if row[actual+'_port'] in rule.get('exclude_'+prefix+'_ports',[]): return False
        return 'protocol' not in rule or row['protocol'] == rule['protocol']

    def apply(self, row):
        matches, boundary = [], []
        for rule in self.rules:
            reverse = False
            if not self.matches_key(row, rule):
                if not rule.get('bidirectional', True) or not self.matches_key(row, rule, True): continue
                reverse = True
            start, end = ns(rule['start']), ns(rule['end'])
            if row['end_ns'] < start or row['start_ns'] > end: continue
            # Refine on complete flow payload, never on the truncated encoder input.
            payload = row['bwd_payload_bytes' if reverse else 'fwd_payload_bytes']
            if 'min_fwd_payload_bytes' in rule and payload < rule['min_fwd_payload_bytes']: continue
            if 'max_fwd_payload_bytes' in rule and payload > rule['max_fwd_payload_bytes']: continue
            if row['start_ns'] < start or row['end_ns'] > end: boundary.append(rule)
            else: matches.append(rule)
        labels = {r['label'] for r in matches}
        if boundary: status, label = 'boundary', None
        elif len(labels) > 1: status, label = 'conflict', None
        elif labels: status, label = 'matched', next(iter(labels))
        else: status, label = 'unmatched', None
        annotation = label
        policy = self.config.get('attempted_policy','exclude')
        if policy not in ['exclude','benign','attack']: raise ValueError('Invalid Attempted label policy')
        applied_policy = 'direct' if label else 'unresolved'
        if label and '_Attempted_' in label:
            # The corrected-dataset authors explicitly discourage learning these
            # diagnostic categories as separate target classes.
            label = 'BENIGN' if policy=='benign' else label.split('_Attempted_')[0] if policy=='attack' else None
            applied_policy = 'attempted_'+policy
        return dict(label=label, annotation_label=annotation, label_status=status,
                    training_label_policy=applied_policy,
                    label_rule_ids=[r['id'] for r in matches+boundary],
                    label_sources=sorted({r['source'] for r in matches+boundary}),
                    label_policy_source=self.config.get('label_policy_source'))
