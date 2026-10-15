# Recall of next-layer expert predictors on dumped MoE inputs (ESPETERO_DS4_DUMPX).
import sys, json, struct, numpy as np
D, E, K = 5120, 384, 6
dump, model = sys.argv[1], sys.argv[2]
idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
def tensor(name):
    f = open(f"{model}/{idx[name]}", "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n))[name]
    f.seek(8 + n + h["data_offsets"][0]); b = f.read(h["data_offsets"][1] - h["data_offsets"][0])
    if h["dtype"] == "BF16": return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(h["shape"])
    return np.frombuffer(b, np.float32).reshape(h["shape"])
def X(l): return np.fromfile(f"{dump}/l{l}.x16", np.float16).reshape(-1, D).astype(np.float32)
def sel(l): return np.fromfile(f"{dump}/l{l}.sel", np.int32).reshape(-1, K)
def scores(z, b): return np.sqrt(np.logaddexp(0, z)) + b
def recall(s, truth, k):
    top = np.argpartition(-s, k, axis=1)[:, :k]
    hit = (top[:, :, None] == truth[:, None, :]).any(1)
    return hit.mean()
ntr = 13000
layers = [int(a) for a in sys.argv[3].split(",")] if len(sys.argv) > 3 else range(0, 39)
ahead = int(sys.argv[4]) if len(sys.argv) > 4 else 1
print(f"predicting layer l+{ahead} from layer l's MoE input; recall of the 6 routed experts within the top 6 / 8 / 12")
for l in layers:
    t = l + ahead
    G = tensor(f"layers.{t}.ffn.gate.weight"); b = tensor(f"layers.{t}.ffn.gate.bias")
    Xl, Xt, St = X(l), X(t), sel(t)
    n = min(len(Xl), len(Xt))
    Xl, Xt, St = Xl[:n], Xt[:n], St[:n]
    Z = Xt @ G.T
    tr, te = slice(0, ntr), slice(ntr, n)
    base = scores(Xl[te] @ G.T, b)
    A = Xl[tr]; C = A.T @ A; lam = 1e-3 * np.trace(C) / D
    W = np.linalg.solve(C + lam * np.eye(D, dtype=np.float32), A.T @ Z[tr])
    pred = scores(Xl[te] @ W, b)
    true_chk = recall(scores(Z[te], b), St[te], 6)  # sanity: the true logits give the true routes
    print(f"layer {l:2d}->{t:2d}: base {recall(base, St[te], 6):.2f}/{recall(base, St[te], 8):.2f}/{recall(base, St[te], 12):.2f}  "
          f"trained {recall(pred, St[te], 6):.2f}/{recall(pred, St[te], 8):.2f}/{recall(pred, St[te], 12):.2f}  (check {true_chk:.2f})", flush=True)
