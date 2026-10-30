"""CPU teaching example for the c3d8640 LL walkthrough; no Torch/NPU needed.

Only rank 0's two tokens are expanded here. Other ranks may choose experts
4/5/6/7, so they add no rows to the highlighted rank16 expert0 example.
Two displayed vector components stand for H=3584; this is not a device test.
"""
import json

P, E, B, H, K, C, M = 32, 384, 2, 3584, 4, 56, 4
ROUTES = [[0, 192, 205, 218], [1, 192, 229, 242]]
WEIGHTS = [[.25, .25, .25, .25], [.5, .25, .125, .125]]
EXPERT_OUTPUTS = [[[4, 8], [8, 4], [12, 0], [16, 12]],
                  [[2, 6], [4, 8], [8, 16], [16, 8]]]


def align(n, a):
    return (n + a - 1) // a * a


def replay():
    local = E // P
    receivers = (C - M) // 2
    rows, seen = [], {}
    for t, routes in enumerate(ROUTES):
        groups = {}
        for k, expert in enumerate(routes):
            groups.setdefault((expert // local) // 16, []).append(k)
        remote = groups[1]
        mesh = {}
        for k in remote:
            mesh.setdefault(((routes[k] // local) % 16) // 4, []).append(k)
        selected = min(mesh.values(), key=lambda ks: (-len(ks), min(ks)))
        first_k = min(selected)
        assert first_k == 1
        slots = []
        for k, expert in enumerate(routes):
            q = seen.get(expert, 0)
            seen[expert] = q + 1
            dst, local_e = divmod(expert, local)
            slot = (((0 + dst) % P) * local + local_e) * B + q
            slots.append(dict(k=k, expert=expert, destination=dst, q=q, slot=slot,
                              forward_owner=slot % receivers))
        members = [k for k in remote if k != first_k]
        partial = [sum(WEIGHTS[t][k] * EXPERT_OUTPUTS[t][k][d] for k in remote) for d in range(2)]
        final = [WEIGHTS[t][0] * EXPERT_OUTPUTS[t][0][d] + partial[d] for d in range(2)]
        direct_math = [sum(WEIGHTS[t][k] * EXPERT_OUTPUTS[t][k][d] for k in range(K)) for d in range(2)]
        assert final == direct_math
        mask = (1 << 0) | (1 << first_k)
        rows.append(dict(token=t, routes=slots, first_k=first_k, members=members,
                         mask=mask, source_weights_active=[WEIGHTS[t][0], 1.0],
                         partial=partial, output=final, return_slots=[t*K, t*K+first_k]))
    # Rank16 receives exactly these two rows in local-expert/source/q order.
    assignments = []
    for r in range(receivers):
        begin, end = 2*r//receivers, 2*(r+1)//receivers
        assignments += [dict(row=row, output_owner=r, physical_core=M+receivers+r) for row in range(begin,end)]
    assert assignments == [dict(row=0,output_owner=12,physical_core=42),
                           dict(row=1,output_owner=25,physical_core=55)]
    assert rows[0]['routes'][1]['slot'] == 384 and rows[1]['routes'][1]['slot'] == 385
    assert [rows[t]['routes'][1]['forward_owner'] for t in range(2)] == [20,21]
    # Actual prequantized layout helper formula, with the full H.
    quant = align(H,256)
    scale = (H+127)//128*4
    header = align(quant+scale,32)
    index = header+32
    if index%480 + K*8 > 480:
        index=align(index,480)
    payload=align(index+K*8,32)
    packet=(payload+479)//480*512
    result=(2*H+479)//480*512
    assert (packet,result,131072//packet,131072//result)==(4096,7680,32,17)
    assert [r['output'] for r in rows]==[[10.,6.],[5.,8.]]
    return dict(P=P,E=E,B=B,H=H,K=K,C=C,M=M,local_experts=local,receivers=receivers,
                tokens=rows,rank16_output_owners=assignments,
                dispatch_bytes=packet,combine_bytes=result,dispatch_gm_slots=32,combine_gm_slots=17)


if __name__ == '__main__':
    print(json.dumps(replay(),ensure_ascii=False,indent=2))
    print('PASS: routes, q, fixed slots, balanced output owners, weights and packet sizes')
