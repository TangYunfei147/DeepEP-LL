"""Exact pending permutations, including partial-ready and overlapping ranges."""
from pathlib import Path
import shutil
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]


def test_suffix_shortcut_matches_descending_swap_last(tmp_path):
    compiler=shutil.which('g++') or shutil.which('clang++')
    if not compiler:pytest.skip('Run native CPU guard check in CANN compile environment')
    source=tmp_path/'pending.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include <numeric>
#include "csrc/kernels/dispatch/low_latency/common/ll_pending_batch.h"
int main() {
 for(unsigned size=0;size<80;++size) for(unsigned start=0;start<=size;++start)
  for(unsigned count=0;count<=16 && count<=size-start;++count) {
   std::vector<unsigned> masks={0U,(1U<<count)-1U,((1U<<count)-1U)&0xaaaaU,
                                ((1U<<count)-1U)&0x5555U};
   if(count<8)for(unsigned mask=0;mask<(1U<<count);++mask)masks.push_back(mask);
   for(unsigned mask:masks) {
    std::vector<unsigned> original(size);std::iota(original.begin(),original.end(),0U);
    auto reference=original;
    for(unsigned n=count;n>0;--n)if(mask&(1U<<(n-1))) {
      reference[start+n-1]=reference.back();reference.pop_back();
    }
    if(DispatchDedup::LlCanDropReadySuffix(size,start,count,mask)) {
      original.resize(size-count);assert(original==reference);
    } else {assert(!(count>0 && start+count==size && mask==((1U<<count)-1U)));}
   }
  }
 // This interior overlap must not be mistaken for a simple suffix removal.
 assert(!DispatchDedup::LlCanDropReadySuffix(8,3,3,7));
 assert(!DispatchDedup::LlCanDropReadySuffix(8,0,32,0xffffffffU));
}
''')
    binary=tmp_path/'pending'
    subprocess.run([compiler,'-std=c++17','-O2','-I',str(ROOT),str(source),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
