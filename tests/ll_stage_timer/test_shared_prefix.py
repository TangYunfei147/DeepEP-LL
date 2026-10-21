"""Aligned shared-prefix ownership, prefix equivalence, build/mixed-wheel guards."""
from pathlib import Path
import importlib.util
import shutil
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('r0',[0,1])
@pytest.mark.parametrize('send',[1,28,55])
def test_actual_partition_and_prefix_values(tmp_path,r0,send):
    cc=shutil.which('g++')
    if cc is None:pytest.skip('C++ compiler required on A3')
    src=tmp_path/'prefix.cpp'
    src.write_text(r'''
#include <cassert>
#include <vector>
#include "csrc/kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
using namespace DispatchDedup;
int main(){
 for(unsigned e:{1U,7U,8U,9U,16U,31U,255U,256U,384U,1024U}){
  std::vector<unsigned> count(e),reference(e),actual(e),hits(e);
  unsigned total=0,active=0;
  for(unsigned j=0;j<e;++j){count[j]=(j%7==0?0:(j*13)%19);total+=count[j];reference[j]=total;}
  for(unsigned worker=0;worker<kLlRecvCoreNum;++worker){
   auto p=LlMakePrefixPartition(e,worker);
   if(!p.count)continue;
   ++active;assert(p.begin%8==0 && p.begin+p.count<=e);
   assert(p.count<=LlPrefixMaxRecords(e));
   unsigned offset=0;for(unsigned j=0;j<p.begin;++j)offset+=count[j];
   for(unsigned j=p.begin;j<p.begin+p.count;++j){offset+=count[j];actual[j]=offset;++hits[j];}
  }
  assert(active==LlMakePrefixPartition(e,0).active);
  if(kLlPrefixR0)assert(active==1);
  assert(actual==reference);for(auto h:hits)assert(h==1);
  auto layout=DispatchServerDedupCountLayout(e);
  assert(layout.cumsumFlagOffset==layout.prefixOffset+layout.prefixStride);
  assert(layout.stateBytes<1024U*1024U);
 }
}
''')
    exe=tmp_path/'prefix'
    subprocess.run([cc,'-std=c++17','-O2',f'-I{ROOT}',f'-DDEEPEP_LL_PREFIX_R0={r0}',
                    f'-DDEEPEP_LL_SEND_CORES={send}',f'-DDEEPEP_LL_RECV_CORES={56-send}',str(src),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_build_cache_and_invalid_modes(monkeypatch):
    build=load('prefix_build','build_support.py')
    assert build.build_cache_key(dict(ll_prefix_mode='parallel'))!=build.build_cache_key(dict(ll_prefix_mode='r0'))
    monkeypatch.setattr(build.platform,'system',lambda:'Linux')
    with pytest.raises(ValueError,match='PREFIX_MODE'):
        build.build_configuration(dict(DEEPEP_NPU_ARCH='Ascend950',DEEPEP_LL_PREFIX_MODE='typo'))
    for file in ('cmake/Device.cmake','cmake/CoreSplit.cmake','build_support.py'):
        assert 'DEEPEP_LL_PREFIX_MODE' in (ROOT/file).read_text()
    st=load('prefix_st','tests/elastic/test_ep_low_latency.py')
    for mode,protocol in [('r0',16028),('parallel',17028),(None,16028),('r0',15028)]:
        with pytest.raises(RuntimeError):
            st.validate_core_split_capability(dict(active_aiv=56,send_cores=28,recv_cores=28,prefix_mode=mode,core_protocol=protocol))

def test_single_table_readers_and_publication_order():
    base=ROOT/'csrc/kernels/dispatch/low_latency'
    counts=(base/'device/dispatch_server_dedup_counts.h').read_text(encoding='utf-8')
    publish=counts.split('inline void ServerDedupCounts::CalRecvAndSetFlag()',1)[1].split('inline bool ServerDedupCounts::TryCumSumFlag()',1)[0]
    assert 'index * rscvStatusNum_' not in publish
    assert publish.index('DataCopyPad(workspaceGlobal[startStatusIndex_]')<publish.index('HardEvent::MTE3_V')<publish.index('PIPE_MTE3')<publish.index('layout_.cumsumFlagOffset')
    assert 'DataCopyPad(prefix, workspace, copy, pad)' in counts
    assert 'recStatusNumPerCore_<=255U' in counts
    receiver=(base/'device/dispatch_server_dedup_recv.h').read_text(encoding='utf-8')
    assert 'receiver_) * counts.prefixStride' not in receiver
