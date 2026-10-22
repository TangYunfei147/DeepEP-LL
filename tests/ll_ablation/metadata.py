"""Exact valid metadata fingerprints, outside all measured sample loops."""
import hashlib
import json
from pathlib import Path

def save_metadata(handle,entries,args,m):
    import torch.distributed as dist
    rows=sum(map(len,entries))
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
    values={"source_weights":digest(handle.source_weights),
            "destination_index":digest(handle.destination_index[:rows]),
            "send_counts":digest(handle.send_counts)}
    if m>32:
        kinds=handle.token_type[:rows].cpu()
        values['token_type']=digest(kinds)
        values['source_mask']=digest(handle.source_mask)
        relay=handle.relay_read_index[:rows].cpu()
        valid=hashlib.sha256()
        for row,kind in enumerate(kinds.tolist()):
            if kind not in (1,2,3):raise AssertionError('Invalid token type')
            if kind==2:
                members=int(relay[row,0,0])
                if not 1<=members<args.num_topk:raise AssertionError('Invalid relay member count')
                valid.update(row.to_bytes(8,'little'))
                valid.update(relay[row,:members+1].contiguous().numpy().tobytes())
        values['valid_relay_entries']=valid.hexdigest()
    key=json.dumps(dict(B=args.num_tokens,M=m,H=args.hidden,K=args.num_topk,E=args.num_experts,
                        dtype=args.dtype,routes=args.ablation_route_digest),sort_keys=True)
    path=Path(args.run_directory)/f'ablation_metadata_rank{dist.get_rank():03}.json'
    saved=json.loads(path.read_text()) if path.exists() else {}
    if key in saved and saved[key]!=values:
        raise AssertionError('Ablation metadata changed across identical route replay')
    saved[key]=values
    path.write_text(json.dumps(saved,indent=2,sort_keys=True))
