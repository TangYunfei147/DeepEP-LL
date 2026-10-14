"""CPU-only domain/scene/volume matrix. This never launches any devices."""
import argparse
import json

DOMAINS=[2,4,8,16,32]
def make_case(world,scene,bytes_edge,fanout=None):
    if world not in DOMAINS or scene not in [1,2,3,4] or bytes_edge<=0 or bytes_edge%32:
        raise ValueError('Invalid domain/scene/alignment')
    if bytes_edge>64<<20 or world*bytes_edge>256<<20:raise ValueError('Receiver workspace budget exceeded')
    n=world-1 if fanout is None else fanout
    if not 1<=n<world or (fanout is not None and scene!=3):raise ValueError('fanout only applies to scenario3 and must be in 1..world-1')
    edges=1 if scene==1 else world-1 if scene==2 else n if scene==3 else world*(world-1)
    max_out=n if scene==3 else world-1 if scene==4 else 1
    return {'fanout':n if scene==3 else None,'world':world,'scenario':scene,'bytes_per_edge':bytes_edge,
            'edge_count':edges,'job_wire_bytes':edges*bytes_edge,'max_sender_wire_bytes':max_out*bytes_edge,
            'receiver_bytes':world*bytes_edge+32,'native_LL_supported':world in [16,32]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--worlds',nargs='+',type=int,default=DOMAINS)
    p.add_argument('--bytes-per-edge',nargs='+',type=int,default=[32,512,8192,131072,2097152])
    p.add_argument('--fanout',type=int,help='Receiver count for scenario3; ranks 1..N')
    args=p.parse_args()
    cases=[make_case(world,scene,volume,args.fanout if scene==3 else None) for world in args.worlds for scene in [1,2,3,4] for volume in args.bytes_per_edge]
    print(json.dumps({'cases':cases,'timing_layers':[0,1,3],
          'metadata':'local production sender extraction uses supported logical partition shape',
          'full_native_LL_worlds':[16,32],'A5_actual_device_limit':[0,1]},indent=2))
if __name__=='__main__':main()
