"""CPU checks for overlap capture and layout/progress invariants."""
import importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_native_layout_matrix(tmp_path):
    import shutil, subprocess, pytest
    cc=shutil.which("g++") or shutil.which("clang++")
    if not cc: pytest.skip("C++ compiler unavailable; run native layout matrix on target")
    source=tmp_path/"layout.cpp"
    source.write_text(r'''#include "csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h"
#include <cassert>
int main() {
 using namespace DispatchDedup;
 unsigned enabled=0, fallback=0;
 for (auto mode:{EpServerDedupInputMode::Plain16,EpServerDedupInputMode::Quantize16,EpServerDedupInputMode::PrequantizedFp8Packs})
 for (unsigned p:{16U,32U,64U,128U}) for(unsigned bs:{1U,8U,256U,1024U})
 for(unsigned h:{128U,3584U,7168U}) for(unsigned k:{1U,8U}) {
   unsigned e=p*6;
   auto data=DispatchServerDedupDataLayout(bs,h,k,e,mode);
   for(bool dedup:{false,true}) for(unsigned core=0;core<56;++core) {
    auto l=LlMakeOverlapLayout(data,bs,k,e,p,0,core,mode,dedup);
    if(!l.enabled) {++fallback;continue;}
    ++enabled;assert(l.totalBytes<=kLlDispatchUbBytes);
    assert(l.sendLanes>=1 && l.sendLanes<=6 && l.recvLanes>=1 && l.recvLanes<=6);
    assert(l.sendBytes>=l.sendLanes*data.tokenStride);
    assert(((core==kLlSendCoreNum) || (!dedup && bs<=32U && core>=kLlSendCoreNum))==(l.countBytes!=0));
   }
 }
 assert(fallback || enabled);
 if(kLlOverlapExperimentStage==1) assert(!enabled);
 else assert(enabled && fallback);
}''',encoding="utf-8")
    exe=tmp_path/"layout"
    subprocess.run([cc,"-std=c++17","-include","initializer_list","-I",str(ROOT),str(source),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)




def test_delayed_count_and_terminal_dependencies_make_progress():
    # Protocol model, not hardware timing: count publication is independent of
    # initial payload; terminal reception depends on a FIRST_HOP consumption.
    # TryBatch must yield on an unready snapshot so every initial task is issued.
    import random
    for seed in range(40):
        rng=random.Random(seed)
        workers=8
        sends=[[w*1000+i for i in range(rng.randrange(0,67))] for w in range(workers)]
        pending=[[] for _ in range(workers)]
        for source in sends:
            for token in source:
                pending[token%workers].append((token,0))
                pending[(token+1)%workers].append((token,1))
        count_ready=[rng.randrange(0,15) for _ in range(workers)]
        delivered=set();forwarded=set();seen=set();cursor=[0]*workers
        for step in range(2000):
            for w in range(workers):
                # Local submission is unconditional even if no receive is ready.
                batch=sends[w][cursor[w]:cursor[w]+8]
                delivered.update(batch);cursor[w]+=len(batch)
                if step<max(count_ready): continue
                snapshot=pending[w][:16]
                ready=[t for t in snapshot if t[0] in (delivered if t[1]==0 else forwarded)]
                for task in ready:
                    assert task not in seen
                    seen.add(task);pending[w].remove(task)
                    if task[1]==0:forwarded.add(task[0])
                # An empty snapshot must not become an infinite receiver loop.
                if not ready and pending[w]:pending[w]=pending[w][16:]+pending[w][:16]
            if all(cursor[w]==len(sends[w]) and not pending[w] for w in range(workers)):break
        else:raise AssertionError("bounded cooperative schedule failed to progress")
        assert len(seen)==2*sum(map(len,sends))




def test_cooperative_device_callables_are_explicitly_annotated():
    # CPU C++ cannot enforce Bisheng's Host/AICore call-domain rule. Guard the
    # regression explicitly and leave CANN validation to the target build.
    kernel=(ROOT/'csrc/kernels/dispatch/low_latency/dispatch_server_dedup_kernel.h').read_text(encoding='utf-8')
    layout=(ROOT/'csrc/kernels/dispatch/low_latency/common/ll_overlap_layout.h').read_text(encoding='utf-8')
    assert 'auto cooperate=' not in kernel
    assert '__aicore__ inline void operator()() { kernel->AdvanceLocal(*prepared); }' in kernel
    assert '__aicore__ inline void AdvanceLocal(bool& prepared)' in kernel
    assert 'LocalProgress cooperate{this,&prepared};' in kernel
    assert '[](' not in layout and '[&]' not in layout
    assert 'DispatchDedupAlign(' in layout
