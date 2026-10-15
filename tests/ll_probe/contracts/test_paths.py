"""CPU contracts for renamed checkouts and arbitrary invocation directories."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[3]


class PortablePaths(unittest.TestCase):
    def setUp(self):
        temp = REPO / '.tmp'
        temp.mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='path-contract-', dir=temp)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'renamed project with spaces'
        self.cwd = Path(self.temp.name) / 'unrelated caller'
        self.cwd.mkdir()
        files = ['scripts/a5_validation_env.sh', 'tests/ll_probe/native_ll/run_ll16.sh', 'tests/elastic/test_ep_low_latency.py', 'scripts/project_paths.py', 'scripts/project_env.sh', 'tests/ll_probe/paths.py',
                 'tests/ll_probe/profiling/startup_diagnostics.py', 'tests/ll_probe/profiling/profile_probe.py', 'tests/ll_probe/profiling/profile_common.py', 'tests/ll_probe/profiling/bandwidth_model.py',
                 'tests/ll_probe/network_probe/run.py', 'tests/ll_probe/network_probe/defaults.py',
                 'tests/ll_probe/network_probe/config.yaml', 'tests/ll_probe/deployment.py']
        for name in files:
            dst=self.root/name; dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(REPO/name,dst)
        self.env=dict(os.environ)
        for key in ('CANN_ENV','ASCEND_HOME_PATH','PROBE_ROOT','SHMEM_ROOT','CATLASS_ROOT','SHMEM_LIB_DIR',
                    'RANK','LOCAL_RANK','WORLD_SIZE','DEEPEP_PROJECT_ROOT'):
            self.env.pop(key,None)
        self.env['PYTHONDONTWRITEBYTECODE']='1'

    def run_python(self, *args):
        return subprocess.check_output([sys.executable,*map(str,args)],cwd=self.cwd,env=self.env,text=True)

    def test_ll16_launcher_from_renamed_checkout(self):
        result = json.loads(subprocess.check_output(
            ['bash', str(self.root/'tests/ll_probe/native_ll/run_ll16.sh'), '--plan', '--num-tokens', '1', '33'],
            cwd=self.cwd, env=self.env, text=True))
        self.assertEqual(result['world_size'], 16)
        self.assertEqual(result['ranks_per_host'], 16)
        self.assertEqual(result['case_count'], 2)

    def test_shared_environment_preserves_visibility(self):
        keys = ('PROBE_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES')
        command = 'source "$1"; python3 -c \'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.endswith("VISIBLE_DEVICES")}))\''
        for values in ({}, {'ASCEND_RT_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15'},
                       {'PROBE_VISIBLE_DEVICES': '0,1', 'ASCEND_RT_VISIBLE_DEVICES': '8,9', 'CUDA_VISIBLE_DEVICES': 'unchanged'}):
            env = dict(self.env)
            for key in keys: env.pop(key, None)
            env.update(values)
            result = json.loads(subprocess.check_output(['bash', '-e', '-c', command, 'test',
                str(self.root/'scripts/project_env.sh')], cwd=self.cwd, env=env, text=True))
            self.assertEqual({k:result[k] for k in keys if k in result}, values)
        result = json.loads(subprocess.check_output(['bash', '-e', '-c', command, 'test',
            str(self.root/'scripts/a5_validation_env.sh')], cwd=self.cwd, env=env, text=True))
        self.assertEqual({k:result[k] for k in keys}, dict.fromkeys(keys, '0,1'))

    def test_profile_plan_does_not_assume_two_devices(self):
        for key in ('PROBE_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'ASCEND_RT_VISIBLE_DEVICES', 'ASCEND_VISIBLE_DEVICES'):
            self.env.pop(key, None)
        self.env.update(RANK='15', WORLD_SIZE='16', LOCAL_RANK='15')
        result = json.loads(self.run_python(self.root/'tests/ll_probe/profiling/profile_probe.py',
            '--kind', 'comm', '--endpoint', 'tcp://127.0.0.1:19501', '--output-dir', 'results/plan', '--plan'))
        self.assertEqual((result['world'], result['device']), (16, 15))
        self.env['ASCEND_RT_VISIBLE_DEVICES'] = '0,1'
        result = subprocess.run([sys.executable, str(self.root/'tests/ll_probe/profiling/profile_probe.py'),
            '--kind', 'comm', '--endpoint', 'tcp://127.0.0.1:19501', '--output-dir', 'results/plan', '--plan'],
            cwd=self.cwd, env=self.env, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_network_yaml_uses_checkout_not_cwd(self):
        result=json.loads(self.run_python(self.root/'tests/ll_probe/network_probe/run.py',
            '--config','tests/ll_probe/network_probe/config.yaml','--node-index','1','--peer','19','--plan'))
        self.assertEqual(result['ranks'],list(range(16,32)))
        self.assertEqual(result['world'],32)

    def test_profile_relative_output_and_library(self):
        result=json.loads(self.run_python(self.root/'tests/ll_probe/profiling/profile_probe.py',
            '--kind','metadata','--output-dir','results/example','--plan'))
        self.assertEqual(Path(result['library']),self.root/'tests/ll_probe/metadata/build/libmetadata_profile.so')
        bad=subprocess.run([sys.executable,str(self.root/'tests/ll_probe/profiling/profile_probe.py'),'--kind','metadata',
            '--output-dir','../escape','--plan'],cwd=self.cwd,env=self.env,capture_output=True)
        self.assertNotEqual(bad.returncode,0)

    def test_shell_env_and_relative_sdk(self):
        sdk=self.root/'toolchain'; (sdk/'include').mkdir(parents=True)
        (sdk/'set_env.sh').write_text('export ASCEND_HOME_PATH="$REPO_ROOT/toolchain"\n')
        self.env.update(CANN_ENV='toolchain/set_env.sh',SHMEM_ROOT='/unused/sdk',CATLASS_ROOT='/unused/catlass')
        command='source "$1"; printf "%s\\n" "$REPO_ROOT" "$PROBE_ROOT" "$ASCEND_HOME_PATH" "$SHMEM_LIB_DIR" "${SHMEM_ROOT-unset}" "${CATLASS_ROOT-unset}"'
        result=subprocess.check_output(['bash','-e','-c',command,'paths',str(self.root/'scripts/project_env.sh')],
                                      cwd=self.cwd,env=self.env,text=True).splitlines()
        self.assertEqual(result,[str(self.root),str(self.root),str(sdk),
            str(self.root/'third-party/shmem/install/shmem/lib'),'unset','unset'])

    def test_python_relative_override_and_escape(self):
        spec=importlib.util.spec_from_file_location('portable_path_fixture',self.root/'scripts/project_paths.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        from unittest.mock import patch
        with patch.dict(os.environ,{'PROBE_ROOT':'artifacts'}):
            self.assertEqual(module.workspace_root(),self.root/'artifacts')
            self.assertEqual(module.output_path('artifacts/case'),self.root/'artifacts/case')
            with self.assertRaises(ValueError): module.output_path('../escape')
            outside=self.cwd/'external'; outside.mkdir()
            (self.root/'artifacts').mkdir()
            (self.root/'artifacts/link').symlink_to(outside,target_is_directory=True)
            with self.assertRaises(ValueError): module.output_path('artifacts/link/file')

    def test_profiler_merge_accepts_distinct_machine_output_paths(self):
        import runpy
        sys.path.insert(0,str(REPO/'tests/ll_probe'))
        try:
            analyze=runpy.run_path(str(REPO/'tests/ll_probe/profiling/analyze_profile.py'))['analyze']
        finally:
            sys.path.pop(0)
        paths=[]
        for rank in range(2):
            folder=self.cwd/f'rank{rank}'; folder.mkdir()
            (folder/'native.log').write_text('PASS: normal cleanup\n')
            report={'rank':rank,'world':2,'kernel':'four_comm_1','warmup':10,'capture_steps':2,
                    'sample_window':1,'delay_shape':[64,64],'reference_defaults':False,
                    'measurement':'named_kernel_task_duration_not_end_to_end','git_revision':'fixture',
                    'configuration':{'kind':'comm','layer':3,'scenario':1,'bytes':32,
                                     'output_dir':str(self.cwd/f'machine{rank}/different/output')},
                    'source_sha256':{},'binary_sha256':'fixture','correctness_passed':True,
                    'kernel_statistics':[{'kernel_name':'four_comm_1','task_type':'AI_VECTOR_CORE',
                                          'kernel_us':[2,3],'selected_kernel_us':[3]}]}
            path=folder/'report.json'; path.write_text(json.dumps(report)); paths.append(path)
        self.assertEqual(analyze(paths)['all_rank_audit'],'passed')


if __name__=='__main__': unittest.main()
