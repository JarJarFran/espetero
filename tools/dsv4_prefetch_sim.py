# Token-by-token: per layer an LRU cache of C experts; before layer l runs, the predicted experts that are not
# cached are prefetched (top-k of layer l's router on layer l-1's MoE input, or only those above a margin).
# Counts per token: experts read in all, and those still missing when the layer runs (on the critical path).
import sys, json, struct, collections, numpy as np
D, E, K = 5120, 384, 6
dump, model = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
def tensor(name):
    f = open(f"{model}/{idx[name]}", "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))[name]
    f.seek(8 + n + h["data_offsets"][0]); b = f.read(h["data_offsets"][1] - h["data_offsets"][0])
    if h["dtype"] == "BF16": return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(h["shape"])
    return np.frombuffer(b, np.float32).reshape(h["shape"])
C = int(sys.argv[3]) if len(sys.argv) > 3 else 126
rows = slice(4000, 16384)  # the cache warms on the first rows of the sequence, measured after
ks = [0, 6, 8, 10]
tot = {k: [0, 0] for k in ks}; ntok = 0
for l in range(1, 40):
    G = tensor(f"layers.{l}.ffn.gate.weight"); b = tensor(f"layers.{l}.ffn.gate.bias")
    Xp = np.fromfile(f"{dump}/l{l-1}.x16", np.float16).reshape(-1, D).astype(np.float32)
    S = np.fromfile(f"{dump}/l{l}.sel", np.int32).reshape(-1, K)
    n = min(len(Xp), len(S)); Xp, S = Xp[:n], S[:n]
    sc = np.sqrt(np.logaddexp(0, Xp @ G.T)) + b
    order = np.argsort(-sc, axis=1)[:, :max(ks)]
    for k in ks:
        cache = collections.OrderedDict(); reads = crit = 0
        for t in range(n):
            meas = t >= rows.start
            if k:
                for e in order[t, :k]:
                    e = int(e)
                    if e not in cache:
                        reads += meas; cache[e] = 1
                        if len(cache) > C: cache.popitem(last=False)
            for e in S[t]:
                e = int(e)
                if e in cache: cache.move_to_end(e)
                else:
                    reads += meas; crit += meas; cache[e] = 1
                    if len(cache) > C: cache.popitem(last=False)
        tot[k][0] += reads; tot[k][1] += crit
    ntok = n - rows.start
    print(f"layer {l} done", file=sys.stderr, flush=True)
for k in ks:
    r, c = tot[k][0] / ntok, tot[k][1] / ntok
    name = "no prefetch" if k == 0 else f"prefetch top-{k}"
    print(f"{name:16s}: experts read per token {r:5.1f}, still missing when needed {c:5.1f}")
