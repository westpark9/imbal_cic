"""Meaningful regression checks for session labels, leakage and token compatibility."""
from collections import Counter
from decimal import Decimal
import importlib.util
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import dpkt
import numpy as np
from scapy.all import Ether, IP, TCP, UDP, Raw, wrpcap

from flows import iter_flows, packets
from labels import LabelRules
from run_pipeline import project, split_group, save_json
from fetch_assets import ROOT, sha256

class PipelineTest(unittest.TestCase):
    def setUp(self): self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def capture(self, specs):
        path=self.out/'test.pcap'; ps=[]
        for t,reverse,flags,payload,seq in specs:
            p=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP(src='10.0.0.2' if reverse else '10.0.0.1',dst='10.0.0.1' if reverse else '10.0.0.2')/TCP(sport=80 if reverse else 1234,dport=1234 if reverse else 80,flags=flags,seq=seq)
            if payload:p=p/Raw(payload)
            p.time=Decimal(str(t));ps.append(p)
        wrpcap(str(path),ps);return path

    def flows(self,specs,**kw):
        stats=Counter(); rows=list(iter_flows(self.capture(specs),'capture','group',stats,**kw));return rows,stats

    def test_fin_handshake_not_split_after_first_fin(self):
        rows,_=self.flows([(1,0,'S',b'',10),(1.1,1,'SA',b'',20),(1.2,0,'FA',b'',11),(1.3,1,'A',b'',21),(1.4,1,'FA',b'',21),(1.5,0,'A',b'',12)])
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]['packets'],6);self.assertEqual(rows[0]['end_reason'],'fin_handshake')

    def test_rst_separates_sessions_but_preserves_group(self):
        rows,_=self.flows([(1,0,'S',b'',10),(1.01,1,'RA',b'',20),(2,0,'S',b'',30),(2.01,1,'RA',b'',40)])
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['group_id'],rows[1]['group_id'])
        self.assertNotEqual(rows[0]['flow_id'],rows[1]['flow_id'])

    def test_timeout_segments_keep_group_and_single_packet(self):
        rows,_=self.flows([(1,0,'S',b'',10),(4,1,'A',b'',20)],idle_seconds=1,active_seconds=1)
        self.assertEqual(len(rows),2);self.assertEqual(rows[0]['group_id'],rows[1]['group_id']);self.assertEqual(rows[0]['packets'],1)

    def test_reorder_and_timestamp_precision(self):
        stats=Counter();ps=list(packets(self.capture([(1.123456,0,'S',b'',1),(1.123455,1,'SA',b'',2)]),stats))
        self.assertEqual([p.ns for p in ps],[1123455000,1123456000]);self.assertEqual(stats['out_of_order_packets'],1)
        with self.assertRaises(ValueError):list(packets(self.capture([(3,0,'S',b'',1),(1,1,'SA',b'',2)]),Counter(),reorder_seconds=0))

    def test_partial_final_packet_is_not_accepted_as_complete(self):
        path=self.capture([(1,0,'S',b'',1),(1.1,1,'SA',b'payload',2)])
        path.write_bytes(path.read_bytes()[:-3])
        with self.assertRaises(IOError):list(packets(path,Counter()))
        stats=Counter();ps=list(packets(path,stats,allow_truncated=True))
        self.assertEqual(len(ps),1);self.assertEqual(stats['truncated_capture_tail'],1)
        flow=list(iter_flows(path,'x','g',Counter(),allow_truncated=True))[0]
        self.assertTrue(flow['capture_truncated'])

    def test_label_conflict_boundary_and_unmatched(self):
        path=self.out/'labels.json';base=dict(id='a',start=1,end=2,src_ip='10.0.0.1',dst_ip='10.0.0.2',label='attack',source='fixture')
        save_json(path,dict(rules=[base,{**base,'id':'b','label':'benign'}]));rules=LabelRules(path)
        rows,_=self.flows([(1.1,0,'S',b'',1),(1.2,1,'RA',b'',2)])
        self.assertEqual(rules.apply(rows[0])['label_status'],'conflict')
        save_json(path,dict(rules=[base]));rules=LabelRules(path)
        self.assertEqual(rules.apply({**rows[0],'start_ns':999999999})['label_status'],'boundary')
        self.assertEqual(rules.apply({**rows[0],'start_ns':3_000_000_000,'end_ns':4_000_000_000})['label_status'],'unmatched')
        reversed_row={**rows[0],'src_ip':'10.0.0.2','dst_ip':'10.0.0.1'}
        self.assertEqual(rules.apply(reversed_row)['label'],'attack')

    def test_train_only_pca_is_invariant_to_test_features(self):
        rows=[dict(flow_id=str(i),group_id=str(i),split='train' if i<5 else 'validation' if i==5 else 'test',label='A',label_status='matched') for i in range(8)]
        x=np.random.default_rng(43).normal(size=(8,4)).astype('float32')
        (self.out/'flows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        def run(data):
            np.save(self.out/'embeddings.npy',data);save_json(self.out/'encoder.json',dict(embeddings_sha256=sha256(self.out/'embeddings.npy')))
            project(dict(pca_components=3,seed=43),self.out)
            with np.load(self.out/'pca.npz') as z:return z['components'].copy(),z['mean'].copy()
        a=run(x);x[5:]*=1_000_000;b=run(x)
        np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[1],b[1])

    def test_attempted_annotation_is_not_an_independent_target(self):
        path=self.out/'rules.json'
        rule=dict(id='attempt',start=0,end=3,src_ip='10.0.0.1',label='FTP_Attempted_ClosedPort',source='fixture')
        save_json(path,dict(rules=[rule],attempted_policy='benign'))
        row=self.flows([(1,0,'S',b'',1),(1.1,1,'RA',b'',2)])[0][0]
        result=LabelRules(path).apply(row)
        self.assertEqual(result['label'],'BENIGN')
        self.assertEqual(result['annotation_label'],'FTP_Attempted_ClosedPort')
        save_json(path,dict(rules=[rule]))
        self.assertIsNone(LabelRules(path).apply(row)['label'])

    def test_matches_official_burst_tokenizer(self):
        """Compare our retained headers/bursts to upstream's binary-field tokenizer."""
        specs=[(1,0,'S',b'',100),(1.001,1,'SA',b'',200),(1.002,0,'PA',b'abc',101),(1.02,0,'A',b'',104)]
        path=self.capture(specs); rows=list(iter_flows(path,'x','g',Counter()))
        spec=importlib.util.spec_from_file_location('official_tokenizer',ROOT/'tabpfn/third_party/upstream/netFound/pre_process_src/Tokenize.py')
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        official.config=json.loads((ROOT/'tabpfn/third_party/upstream/netFound/configs/DefaultConfigNoTCPOptions.json').read_text())
        official.INTERNAL_IPS=[]
        binary=self.out/'fields.6'
        with binary.open('wb') as f:
            f.write(bytes([6]))
            # Expected relative sequence/ack values from the pinned C++ extractor.
            relative=[(0,0),(0,(-100)&0xffffffff),(1,(-200)&0xffffffff),(4,(-200)&0xffffffff)]
            for p,(seq,ack) in zip(packets(path,Counter()),relative):
                f.write(struct.pack('<QBBHBBIIHHBHIIH',p.ns,*p.ip_header,int.from_bytes(bytes([10,0,0,1 if p.src.endswith('1') else 2]),'big'),int.from_bytes(bytes([10,0,0,2 if p.src.endswith('1') else 1]),'big'),p.sport,p.dport,p.flags,p.window,seq,ack,p.urgent))
                f.write(b'\x00'*12)
        result=official.tokenize_file(str(binary),None)
        expected=official.slice_bytes_to_16bit_tokens(result[1]).tolist()
        self.assertEqual(rows[0]['burst_tokens'],expected)
        for k,v in zip(['directions','bytes','iats','counts'],result[2:6]):self.assertEqual(rows[0][k],v,k)

if __name__=='__main__':unittest.main()
