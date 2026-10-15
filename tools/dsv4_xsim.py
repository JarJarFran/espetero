#!/usr/bin/env python3
"""Expert-cache eviction policies replayed on an access trace (ESPETERO_DS4_TRACE=FILE: "pass layer expert rows").

usage: dsv4_xsim.py TRACE [SLOTS]   (SLOTS: the cache size the server logs, 5168 at 97 GB)
Prints misses overall and in decode passes for LRU, ARC, SLRU, decayed LFU and Belady's optimum."""
import sys, heapq, math, collections
tr = [tuple(map(int, l.split())) for l in open(sys.argv[1])]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5168
calls = collections.OrderedDict()
for p, l, e, r in tr: calls.setdefault((p, l), []).append((l * 384 + e, r))
pcount = collections.Counter(p for p, l in calls)
decode = {p for p in pcount if all(len(calls[(p, l)]) <= 6 for l in range(40) if (p, l) in calls)}

def run(policy):
    m_dec = m_all = a_dec = 0
    pol = policy()
    for (p, l), acc in calls.items():
        keys = [k for k, _ in acc]
        miss = pol.access(p, l, acc)
        m_all += miss
        if p in decode: m_dec += miss; a_dec += len(acc)
    return m_all, m_dec, a_dec

class LRU:
    def __init__(s): s.d = collections.OrderedDict()
    def access(s, p, l, acc):
        miss = 0
        for k, r in acc:
            if k in s.d: s.d.move_to_end(k)
            else:
                miss += 1
                if len(s.d) >= N:
                    # evict the oldest not in this call
                    cur = {kk for kk, _ in acc}
                    for old in s.d:
                        if old not in cur: break
                    del s.d[old]
                s.d[k] = 1
        for k, r in acc: s.d.move_to_end(k)
        return miss

class LFU:  # decayed count per pass (rows weight w), half-life h passes; new entries start at init
    def __init__(s, h, w=False, init=0.0):
        s.h, s.w, s.init = h, w, init; s.sc = {}; s.hp = []; s.ver = {}
    def key(s, k): sc, sp = s.sc[k]; return (math.log2(max(sc, 1e-30)) + sp / s.h)
    def access(s, p, l, acc):
        miss = 0
        cur = {k for k, _ in acc}
        for k, r in acc:
            if k not in s.sc:
                miss += 1
                if len(s.sc) >= N:
                    held = []
                    while True:
                        v, kk, ver = heapq.heappop(s.hp)
                        if kk not in s.sc or s.ver[kk] != ver: continue
                        if kk in cur: held.append((v, kk, ver)); continue
                        del s.sc[kk]; break
                    for x in held: heapq.heappush(s.hp, x)
                s.sc[k] = (s.init, p); s.ver[k] = 0
        for k, r in acc:
            sc, sp = s.sc[k]
            sc = sc * 2 ** (-(p - sp) / s.h) + (r if s.w else 1)
            s.sc[k] = (sc, p); s.ver[k] += 1
            heapq.heappush(s.hp, (s.key(k), k, s.ver[k]))
        return miss

class SLRU:  # probation / protected (fraction f of the cache)
    def __init__(s, f): s.f = f; s.pr = collections.OrderedDict(); s.pt = collections.OrderedDict()
    def access(s, p, l, acc):
        miss = 0; cur = {k for k, _ in acc}
        for k, r in acc:
            if k in s.pt: s.pt.move_to_end(k); continue
            if k in s.pr:
                del s.pr[k]; s.pt[k] = 1
                while len(s.pt) > s.f * N:
                    old = next(iter(s.pt)); del s.pt[old]; s.pr[old] = 1; s.pr.move_to_end(old, last=False)
                continue
            miss += 1
            if len(s.pr) + len(s.pt) >= N:
                src = s.pr if s.pr else s.pt
                for old in src:
                    if old not in cur: break
                del src[old]
            s.pr[k] = 1
        return miss

class ARC:
    def __init__(s): s.t1 = collections.OrderedDict(); s.t2 = collections.OrderedDict(); s.b1 = collections.OrderedDict(); s.b2 = collections.OrderedDict(); s.p = 0
    def rep(s, k, cur):
        if s.t1 and (len(s.t1) > s.p or (k in s.b2 and len(s.t1) == s.p)):
            src, gh = s.t1, s.b1
        else: src, gh = (s.t2, s.b2) if s.t2 else (s.t1, s.b1)
        for old in src:
            if old not in cur: break
        del src[old]; gh[old] = 1
    def access(s, p, l, acc):
        miss = 0; cur = {k for k, _ in acc}
        for k, r in acc:
            if k in s.t1: del s.t1[k]; s.t2[k] = 1; continue
            if k in s.t2: s.t2.move_to_end(k); continue
            miss += 1
            if k in s.b1:
                s.p = min(N, s.p + max(len(s.b2) / len(s.b1), 1)); s.rep(k, cur); del s.b1[k]; s.t2[k] = 1; continue
            if k in s.b2:
                s.p = max(0, s.p - max(len(s.b1) / len(s.b2), 1)); s.rep(k, cur); del s.b2[k]; s.t2[k] = 1; continue
            if len(s.t1) + len(s.b1) == N:
                if len(s.t1) < N: s.b1.popitem(last=False); s.rep(k, cur)
                else:
                    for old in s.t1:
                        if old not in cur: break
                    del s.t1[old]
            elif len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) >= N:
                if len(s.t1) + len(s.t2) + len(s.b1) + len(s.b2) >= 2 * N: s.b2.popitem(last=False)
                s.rep(k, cur)
            s.t1[k] = 1
        return miss

pols = {"LRU": LRU, "ARC": ARC}
for f in (0.5, 0.8, 0.9): pols[f"SLRU{f}"] = lambda f=f: SLRU(f)
for h in (16, 64, 256): pols[f"LFU h{h}"] = lambda h=h: LFU(h)
pols["LFU h256 rows"] = lambda: LFU(256, True)
print(f"{len(tr)} accesses, {len(calls)} calls, {len(pcount)} passes ({len(decode)} decode), N={N}")
for name, P in pols.items():
    a, d, ad = run(P)
    print(f"{name:14s} misses {a:6d}  decode misses {d:6d} ({100*(1-d/ad):.1f}% hit, {d/len(decode):.1f}/token)")

# Belady: evict the entry used furthest in the future (a bound, not a policy)
seq = [(i, k) for i, ((p, l), acc) in enumerate(calls.items()) for k, _ in acc]
pl = [p for (p, l), acc in calls.items() for _ in acc]
nxt = [0] * len(seq); last = {}
for j in range(len(seq) - 1, -1, -1):
    i, k = seq[j]; nxt[j] = last.get(k, 10**12); last[k] = i
cache = {}; hp = []; miss = md = ad = 0
for j, (i, k) in enumerate(seq):
    dec = pl[j] in decode; ad += dec
    if k not in cache:
        miss += 1; md += dec
        if len(cache) >= N:
            while True:
                nn, kk = heapq.heappop(hp)
                if cache.get(kk) == -nn and -nn > i: del cache[kk]; break
    cache[k] = nxt[j]; heapq.heappush(hp, (-nxt[j], k))
print(f"OPT misses {miss}  decode misses {md} ({100*(1-md/ad):.1f}% hit, {md/len(decode):.1f}/token)")
