"""Receiver ordering and ownership invariants for relay-first experiments."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]


def test_relay_first_keeps_metadata_snapshot_and_lane_retirement():
    source=(ROOT/'csrc/kernels/dispatch/low_latency/device/dispatch_server_dedup_recv.h').read_text()
    consume=source[source.index('inline void ConsumeLoaded'):source.index('inline void ClearFlags')]
    body=consume[consume.index('const bool firstHop'): ]
    assert body.index('BuildServerDedupMetadata')<body.index('PrepareServerDedupForward')
    assert body.index('PrepareServerDedupForward')<body.index('ServerDedupReceiveSync<HardEvent::S_MTE3>')
    ready=body.index('WaitFlag<HardEvent::MTE2_MTE3>')
    relay=body.index('DataCopy(output, wire')
    local=body.index('WriteServerDedupPayloadRange(expandX_')
    indices=body.index('DataCopyPad(relayReadIndex_')
    retire=body.index('SetFlag<HardEvent::MTE3_MTE2>')
    assert ready<relay<local<indices<retire
    # After the first asynchronous relay read, no Scalar writes into wire.
    assert '.SetValue(' not in body[relay:retire]


def test_relay_and_local_output_read_the_same_immutable_packet():
    for fanout in range(1,8):
        original=dict(kind='first_hop',token=17,source=3,weights=(1,2,3))
        local_metadata=dict(original)
        wire=dict(original,kind='terminal')
        operations=[('relay',i) for i in range(fanout)]+[('payload',0),('metadata',0),('indices',0)]
        delivered=[]
        for kind,index in operations:
            delivered.append((kind,index,dict(wire) if kind=='relay' else dict(local_metadata)))
        assert all(data['kind']=='terminal' for kind,_,data in delivered if kind=='relay')
        assert local_metadata['kind']=='first_hop'
        assert [i for kind,i,_ in delivered if kind=='relay']==list(range(fanout))
