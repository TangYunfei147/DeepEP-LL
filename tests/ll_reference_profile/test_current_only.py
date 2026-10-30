"""The current checkout contains only the requested reference tracing pipeline."""
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def test_old_recorders_parsers_and_workflows_are_removed():
    for name in ('ll_stage_timer','ll_stage_timer_legacy','ll_stage_timer_sparse','ll_stage_timer_mte3',
                 'll_stage_timer_bandwidth','ll_stage_timer_systemcycle','ll_expert_timer'):
        assert not (ROOT/'deep_ep/utils'/f'{name}.py').exists()
    for directory in ('tests/ll_systemcycle','tests/ll_stage_timer','tests/ll_control_opt'):
        assert not list((ROOT/directory).glob('*.py'))
    for path in (ROOT/'deep_ep').rglob('*.py'):
        source=path.read_text(encoding='utf-8')
        assert not re.search(r'from .*ll_(?:stage|expert)_timer|import .*ll_(?:stage|expert)_timer',source),path
    shell=(ROOT/'tests/ll_reference_profile/one_click.py').read_text()
    assert '36e2d6a' not in shell and 'DIAGNOSTIC_REVISION' not in shell


def test_sampling_and_sdk_identity_are_explicit_and_not_imported_from_ndr():
    current=(ROOT/'dependencies.lock.json').read_bytes()
    before=subprocess.check_output(['git','-C',str(ROOT),'show','c1c9b3d:dependencies.lock.json'])
    assert current.replace(b'\r\n',b'\n')==before.replace(b'\r\n',b'\n')
    parser=(ROOT/'deep_ep/utils/ll_reference_profile_parse.py').read_text()
    assert 'SCHEMA=6' in parser or 'SCHEMA = 6' in parser
    assert 'll-reference-cycles-mte-shared-v1' in parser
    source=(ROOT/'tests/elastic/test_ep_low_latency.py').read_text()
    assert 'initial-reduce-exp100-batch-v3' in source
    assert 'args.ll_stage_timer' not in source and 'args.ll_expert_timer' not in source
