import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import copy
import json
import tempfile
import unittest
from run import hosts, plan
from merge import merge


class Contracts(unittest.TestCase):
    def test_multihost(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml'; p.write_text('- 10.0.0.1\n- 10.0.0.2\n')
            a=argparse.Namespace(config=p,devices_per_host=16,node_index=1,peer=19,
                                 validation=False,target_run=False,port=18679,timeout=180,no_huge_pages=False)
            result=plan(a)
            self.assertEqual(result['ranks'],list(range(16,32)))
            self.assertEqual(result['endpoint'],'tcp://10.0.0.1:18679')
            self.assertEqual(result['workspace_bytes'],8<<30)
            a.devices_per_host=2
            with self.assertRaises(ValueError): plan(a)
            a.validation=True
            with self.assertRaises(ValueError): plan(a)

    def test_yaml_rejects_duplicates_and_commands(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.yaml'
            for text in ('- 10.0.0.1\n- 10.0.0.1','- $(touch x)','hosts: [10.0.0.1]','- 0.0.0.0'):
                p.write_text(text)
                with self.assertRaises(ValueError): hosts(p)

    def test_missing_nodes(self):
        with self.assertRaises(ValueError): merge([])

    def test_report_completeness(self):
        job={'hosts':['10.0.0.1','10.0.0.2'],'devices_per_host':1,'world':2,'peer':1,
             'endpoint':'tcp://10.0.0.1:18679','workspace_bytes':2<<20,'huge_pages':False,
             'scope':'fixture','dense_qps':'0','identity':{},'executable_sha256':'test',
             'shmem_library':{'cmake_flags':[],'sha256':'test'}}
        reports=[]
        for node in range(2):
            j=dict(job,node_index=node,boot_id=str(node))
            reports.append({'local_success':True,'job':j,'records':[
                {'kind':'complete','rank':node},
                {'kind':'mte_readback','rank':node,'sender':1-node,'passed':True},
                {'kind':'udma_route','rank':node,'peer':1-node,'query_rc':0}]})
        with tempfile.TemporaryDirectory() as d:
            paths=[Path(d)/f'{i}.json' for i in range(2)]
            for p,r in zip(paths,reports): p.write_text(json.dumps(r))
            self.assertTrue(merge(paths)['success'])
            with self.assertRaises(ValueError): merge(paths[:1])
            bad=copy.deepcopy(reports[1]); bad['job']['boot_id']='0'
            paths[1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError): merge(paths)
            bad=copy.deepcopy(reports[1]); bad['records']=bad['records'][1:]
            paths[1].write_text(json.dumps(bad))
            with self.assertRaises(ValueError): merge(paths)


if __name__=='__main__': unittest.main()
