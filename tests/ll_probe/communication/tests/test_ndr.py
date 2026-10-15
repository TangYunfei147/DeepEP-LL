"""CPU-only NDR mode, ownership and no-fallback contracts."""
from pathlib import Path
import importlib.util
import json
import subprocess
import pytest
from tests.ll_probe.profiling.profile_probe import parser,plan
from tests.ll_probe.profiling.analyze_profile import analyze
from tests.ll_probe.profiling.tests.test_deployment_reports import reports

ROOT=Path(__file__).resolve().parents[4]
P=ROOT/'tests/ll_probe'


def test_ndr_plan_selects_distinct_entry_without_sdk_or_devices():
    args=parser().parse_args(['--kind','comm','--transport','ndr','--scenario','3','--targets','16','31',
                             '--endpoint','tcp://192.0.2.1:29501','--output-dir','results/ndr-plan'])
    result=plan(args,dict(WORLD_SIZE='32',RANK='16',LOCAL_RANK='0'))
    assert result['kernel']=='four_ndr_3'
    assert result['library'].endswith('libcomm_ndr_profile.so')
    assert result['measurement_scope']['completion']=='per-core per-peer QP quiet'
    assert result['hardware_profile'] is None


def test_ub_reference_rejected_for_ndr():
    args=parser().parse_args(['--kind','comm','--transport','ndr','--endpoint','tcp://127.0.0.1:1',
                             '--output-dir','results/plan','--hardware-config',str(P/'configs/hardware/ascend950.yaml')])
    with pytest.raises(ValueError,match='transport'):plan(args,dict(WORLD_SIZE='2'))


def test_disabled_sdk_cannot_fall_back(tmp_path):
    spec=importlib.util.spec_from_file_location('ndr_support_test',P/'communication/ndr_support.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache=tmp_path/'third-party/shmem/build';cache.mkdir(parents=True)
    (cache/'CMakeCache.txt').write_text('ACLSHMEM_RDMA_SUPPORT:BOOL=OFF\n')
    assert module.sdk_backend(tmp_path) is None


def test_actual_ndr_function_uses_private_qp_and_quiet(tmp_path):
    text=(P/'communication/kernel/comm_kernel.cpp').read_text()
    start=text.index('template<uint32_t Scenario>');end=text.index('template<uint32_t Scenario>',start+1)
    edge=text[start:end]
    start=text.rfind('template<uint32_t Scenario>',0,text.index('__aicore__ inline void RunNdr('))
    body=text[start:text.index('#define NDR_ENTRY',start)]
    cpp=tmp_path/'ndr.cpp'
    cpp.write_text(r"""
#include <cassert>
#include <cstdint>
#include <vector>
#include <initializer_list>
#include "tests/ll_probe/communication/tiling.h"
#define __aicore__
#define __gm__
#define __ubuf__
using GM_ADDR=uint8_t*;
unsigned block=0,count=8; unsigned GetBlockIdx(){return block;} unsigned GetBlockNum(){return count;}
void Trap(){assert(false);} void util_set_ffts_config(uint64_t n){assert(n==123);}
enum class TPosition{VECCALC};
template<class T>struct Tensor{uint8_t bytes[256];void* GetPhyAddr(){return bytes;}};
template<TPosition P>struct TBuf{template<class T>Tensor<T> Get(){return {};}};
struct TPipe{template<class T>void InitBuffer(T&,unsigned n){assert(n==256);}};
struct Call{unsigned peer,qp,n;uint8_t*dst;uint8_t*src;bool quiet;};std::vector<Call> calls;
template<class T>void aclshmemx_roce_qp_put_nbi(T*d,T*s,T*,uint32_t n,int pe,uint32_t qp,uint32_t sync){
 assert(qp==block&&sync==0);calls.push_back({unsigned(pe),qp,n,d,s,false});}
template<class T>void aclshmemx_roce_qp_quiet(uint32_t pe,uint32_t qp,T*,uint32_t sync){
 assert(sync==0&&!calls.empty()&&calls.back().peer==pe&&calls.back().qp==qp&&!calls.back().quiet);calls.back().quiet=true;}
"""+edge+body+r"""
int main(){
 std::vector<uint8_t> src(15392),dst(32*15392+32);
 FourLayerTiling td{15392,4096,4,0,32,3,3,3,(1U<<16)|(1U<<18)|(1U<<31),2};
 for(block=0;block<count;++block)RunNdr<3>(src.data(),dst.data(),reinterpret_cast<uint8_t*>(&td),123);
 assert(calls.size()==8*3);
 unsigned sum=0;for(auto c:calls){assert(c.quiet);assert(c.peer==16||c.peer==18||c.peer==31);assert(c.dst-dst.data()==c.src-src.data());sum+=c.n;}
 assert(sum==3*15392);
 calls.clear();td.layer=1;RunNdr<3>(src.data(),dst.data(),reinterpret_cast<uint8_t*>(&td),123);assert(calls.empty());
}
""")
    exe=tmp_path/'ndr';subprocess.run(['g++','-std=c++17','-I',str(ROOT),str(cpp),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)


def test_mixed_transports_rejected(tmp_path):
    paths=reports(tmp_path,2)
    obj=json.loads(paths[-1].read_text());obj['configuration']['transport']='ndr'
    paths[-1].write_text(json.dumps(obj))
    with pytest.raises(ValueError):analyze(paths)


def test_runtime_rejects_stale_identity_before_loading_devices(monkeypatch, tmp_path):
    spec=importlib.util.spec_from_file_location('ndr_stale',P/'communication/ndr_support.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'build').mkdir()
    (tmp_path/'build/ndr-build.json').write_text(json.dumps({'identity':{'backend':'XSCALE'}}))
    monkeypatch.setattr(module,'HERE',tmp_path)
    monkeypatch.setattr(module,'sdk_backend',lambda:'XSCALE')
    monkeypatch.setattr(module,'identity',lambda backend:dict(backend=backend,changed=True))
    with pytest.raises(RuntimeError,match='identity changed'):module.verify_runtime()


def test_deployment_ndr_plan_omits_default_ub_reference():
    result=subprocess.run(['python3',str(P/'run.py'),'--config','tests/ll_probe/configs/ep2.yaml',
                           '--test','comm','--node-index','0','--plan','--','--transport','ndr'],
                          cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'Deployment UB reference ignored for NDR' in result.stdout
    assert '--hardware-config' not in result.stdout
