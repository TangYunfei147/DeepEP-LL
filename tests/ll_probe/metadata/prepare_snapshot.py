"""Extract test variants without modifying production sources."""
import hashlib
import json
from pathlib import Path
import shutil

SOURCE=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
snapshot=HERE/'snapshot'
snapshot.mkdir(exist_ok=True)
if snapshot.resolve()!=snapshot: raise RuntimeError('Refuse snapshot through symlink')
# Prune removed source files only inside the owned generated kernel tree.
# Otherwise deleted headers can survive incremental snapshot regeneration.
kernel_snapshot = snapshot/'kernels'
if kernel_snapshot.is_symlink(): raise RuntimeError('Refuse kernel snapshot symlink')
if kernel_snapshot.exists():
    for generated in kernel_snapshot.rglob('*'):
        if generated.is_symlink(): raise RuntimeError('Refuse symlink inside generated kernel tree')
        if generated.is_file() and not (SOURCE/'csrc/kernels'/generated.relative_to(kernel_snapshot)).exists():
            generated.unlink()
shutil.copytree(SOURCE/'csrc/kernels',snapshot/'kernels',dirs_exist_ok=True)
sender=snapshot/'kernels/dispatch/low_latency/device/dispatch_server_dedup_send.h'
original=sender.read_text()
needle='        const uint32_t tokenId = token, destination = expert / localExperts_;'
if original.count(needle)!=1: raise RuntimeError('Sender changed; extraction requires review')
# Two variants: detailed q dump for correctness; one checksum/core for timing.
# The checksum consumes every q and prevents elimination of prefix work.
replacement='''        GlobalTensor<uint32_t> diagnostic;
        diagnostic.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ XType*>(x_.GetPhyAddr())));
#if PROBE_METADATA_DIAGNOSTIC
        auto diagnosticQ = qWorkBuf_.Get<uint32_t>();
        diagnosticQ.SetValue(0U, q);
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(diagnostic[token * k_ + firstK], diagnosticQ, {1U, 4U, 0U, 0U, 0U});
        Sync<HardEvent::MTE3_S>();
#else
        diagnosticChecksum_ += q;
#endif
        return;
'''+needle
variant=original.replace(needle,replacement)
anchor='    uint32_t usedBytes_ = 0, lanes_ = 0, packets_ = 0, directCount_ = 0;'
if variant.count(anchor)!=1: raise RuntimeError('Sender state changed')
variant=variant.replace(anchor,anchor+'\n    uint32_t diagnosticChecksum_ = 0;')
anchor='        Sync<HardEvent::MTE3_S>();\n    }\n\n    // 摘要：去重发送'
if variant.count(anchor)!=1: raise RuntimeError('Sender completion changed')
finish='''        Sync<HardEvent::MTE3_S>();
#if !PROBE_METADATA_DIAGNOSTIC
        GlobalTensor<uint32_t> diagnostic;
        diagnostic.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(const_cast<__gm__ XType*>(x_.GetPhyAddr())));
        auto checksum = qWorkBuf_.Get<uint32_t>();
        checksum.SetValue(0U, diagnosticChecksum_);
        Sync<HardEvent::S_MTE3>();
        DataCopyPad(diagnostic[GetBlockIdx() * 8U], checksum, {1U, 4U, 0U, 0U, 0U});
        Sync<HardEvent::MTE3_S>();
#endif
    }

    // 摘要：去重发送'''
sender.write_text(variant.replace(anchor,finish))
(snapshot/'identity.json').write_text(json.dumps({'source_sender_sha256':hashlib.sha256(original.encode()).hexdigest(),
    'modification':'SendPacket suppressed; q dump correctness variant or one checksum/core timing variant; no payload packing/send',
    'scope':'Sender route load, prefix q, dedup classify/tasks, source masks/weights; excludes count exchange, packet-tail metadata, quantization and receiver'},indent=2))
# Create an isolated LL ST variant with independent API-to-consumable loops.
st=(SOURCE/'tests/elastic/test_ep_low_latency.py').read_text()
if st.count('def measure(invoke, args, operation, profile_dir, progress=None, kernel_override=None):')!=1:
    raise RuntimeError('LL ST changed; adapter requires review')
st=st.replace('def measure(invoke, args, operation, profile_dir, progress=None, kernel_override=None):',
              'def measure_kernel(invoke, args, operation, profile_dir, progress=None, kernel_override=None):')
adapter=(HERE.parent/'native_ll/api_adapter.txt').read_text()
st=st.replace('\ndef main(argv=None):','\n'+adapter+'\n\ndef main(argv=None):')
# Keep tests.utils and deep_ep imports resolved against the real repository.
st=st.replace('__test__ = False','sys.path.insert(0, str(Path(__file__).resolve().parents[4]))\n__test__ = False')
st=st.replace('str(Path(__file__).with_name("test_dispatch.py"))','str(Path(__file__).resolve().parents[4] / \"tests/elastic/test_dispatch.py\")')
native_snapshot=HERE.parent/'native_ll/snapshot'
native_snapshot.mkdir(exist_ok=True)
(native_snapshot/'test_ll_four_layers.py').write_text(st)
