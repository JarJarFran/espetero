#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Block-Choice Join (BCJ) prototype over /v1/decisions.

A semantic join L |><|_P R scored by the model's own choice readouts: a block
of B documents of R is the shared context (one root, checkpointed once), every
query of L is a field whose options are the block's documents plus "none", and
the readout is the distribution over those options. Candidate generation costs
|R| root tokens plus |L| x ceil(|R| / B) leaves instead of |L| x |R| pairs.

  bcj_join.py --url http://127.0.0.1:8080 --synthetic 96 48 --block-size 32 --out DIR
  bcj_join.py --url URL --beir scifact/ --limit-queries 100 --limit-docs 512 --block-size 48 --out DIR
  bcj_join.py --url URL --beir scifact/ --mode pairwise --limit-queries 30 --limit-docs 128 --out DIR

Modes: bcj (default) and pairwise, the forest pairwise baseline: one document
is the root, up to --fields queries are its boolean leaves. Both rank every
document for every query; recall@k and MRR come from the gold pairs.

Outputs in --out: requests.jsonl (one line per call: request summary, server
cost fields, parsed scores), scores.jsonl (one line per query: ranked
documents), summary.json (metrics and cost), and the same summary on stdout.
"""

import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# ----------------------------------------------------------------------------- data


def load_beir_index(d):
    """BEIR layout: corpus.jsonl (_id, title, text), queries.jsonl (_id, text),
    qrels/test.tsv (query-id, corpus-id, score). Returns the corpus and query
    texts by id, and each query's gold document ids."""
    corpus = {}
    with open(os.path.join(d, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            corpus[str(o["_id"])] = {"id": str(o["_id"]), "title": o.get("title", "") or "", "text": o.get("text", "") or ""}
    qtext = {}
    with open(os.path.join(d, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            qtext[str(o["_id"])] = o["text"]
    gold = {}
    qrels = os.path.join(d, "qrels", "test.tsv")
    with open(qrels, encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            if i == 0 and not parts[-1].strip().lstrip("-").isdigit():
                continue  # header
            if len(parts) < 3:
                continue
            q, c, s = parts[0], parts[1], parts[2]
            if float(s) > 0 and c in corpus and q in qtext:
                gold.setdefault(q, set()).add(c)
    return corpus, qtext, gold


def load_beir(d, limit_queries, limit_docs, seed):
    """Keeps limit_queries queries that have gold documents, all their gold
    documents, and random others up to limit_docs."""
    rng = random.Random(seed)
    corpus, qtext, gold = load_beir_index(d)
    qids = sorted(gold)
    rng.shuffle(qids)
    if limit_queries:
        qids = qids[:limit_queries]
    queries = [{"id": q, "text": qtext[q], "gold": set(gold[q])} for q in qids]
    need = set()
    for q in queries:
        need |= q["gold"]
    others = [c for c in corpus if c not in need]
    rng.shuffle(others)
    doc_ids = sorted(need)
    if limit_docs:
        if len(doc_ids) > limit_docs:
            sys.exit(f"--limit-docs {limit_docs} is below the {len(doc_ids)} gold documents of these queries")
        doc_ids += others[: limit_docs - len(doc_ids)]
    else:
        doc_ids += others
    docs = [corpus[c] for c in doc_ids]
    return docs, queries


def synthetic(n_docs, n_queries, seed):
    """A join with known answers for plumbing checks: company cards as
    documents, paraphrased statements about one card as queries."""
    rng = random.Random(seed)
    cats = ["bakery", "shipyard", "software studio", "vineyard", "bicycle maker", "law firm", "observatory", "dairy",
            "toy workshop", "printing house", "ferry line", "seed bank"]
    cities = ["Cádiz", "Tromsø", "Osaka", "Valparaíso", "Lviv", "Tunis", "Perth", "Quito", "Bergen", "Porto", "Sapporo",
              "Leeds", "Riga", "Mendoza", "Cork", "Tampere"]
    products = ["rye loaves", "fishing trawlers", "accounting tools", "sparkling wine", "cargo bikes", "patent filings",
                "star catalogues", "goat cheese", "wooden puzzles", "art books", "island crossings", "heirloom beans"]
    docs, queries = [], []
    seen = set()
    for i in range(n_docs):
        while True:
            cat, city, prod = rng.choice(cats), rng.choice(cities), rng.choice(products)
            year, staff = rng.randint(1890, 2020), rng.randint(5, 900)
            key = (cat, city, prod)
            if key not in seen:
                seen.add(key)
                break
        name = f"{city} {cat.title()} No. {i + 1}"
        docs.append({"id": f"d{i + 1}", "title": name,
                     "text": f"{name} is a {cat} founded in {year} in {city}. It makes {prod} and employs {staff} people.",
                     "_f": (cat, city, prod, year, staff)})
    forms = [
        "The {cat} based in {city} that produces {prod} was established in {year}.",
        "A company making {prod} out of {city} has about {staff} employees and dates from {year}.",
        "Founded in {year}, this {cat} in {city} sells {prod}.",
        "{staff} people work at the {city} business whose product is {prod}.",
    ]
    for j in range(n_queries):
        d = docs[j % n_docs]
        cat, city, prod, year, staff = d["_f"]
        queries.append({"id": f"q{j + 1}", "text": rng.choice(forms).format(cat=cat, city=city, prod=prod, year=year, staff=staff),
                        "gold": {d["id"]}})
    for d in docs:
        d.pop("_f", None)
    return docs, queries


# ----------------------------------------------------------------------------- http


def post(url, body, api_key=None, timeout=1800):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + "/v1/decisions", data=data,
                                 headers={"content-type": "application/json",
                                          **({"authorization": "Bearer " + api_key} if api_key else {})})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {msg[:600]}") from None
    return res, time.perf_counter() - t0


def cost_fields(res):
    metal = res.get("metal") or {}
    usage = res.get("usage") or {}
    nf = int(res.get("scored_fields") or 0)
    prefix = int(metal.get("prefix_tokens") or 0)
    cached = int(metal.get("cached_prefix_tokens") or 0)
    prompt = int(usage.get("prompt_tokens") or 0)
    return {"elapsed_ms": res.get("elapsed_ms"), "cache": metal.get("cache"), "prefix_tokens": prefix,
            "cached_prefix_tokens": cached, "root_tokens_computed": max(0, prefix - cached),
            "leaf_tokens": max(0, prompt - nf * prefix), "fields": nf, "metal": metal}


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def clip(s, n):
    return s if len(s) <= n else s[: n - 1] + "…"


# ----------------------------------------------------------------------------- bcj


def doc_line(i, d, max_chars):
    body = (d["title"] + ". " if d["title"] else "") + d["text"]
    return f"[{i}] " + clip(" ".join(body.split()), max_chars)


def probe_labels(a, n):
    """The engine's answer-label alphabet for n choices (each candidate of a
    response carries its label); --tag-labels tags the documents with it."""
    body = {"context": "Probe.", "schema": {"p": {"type": "enum", "description": "Probe field.",
                                                  "choices": [str(i) for i in range(n)]}}}
    res, _ = post(a.url, body, a.api_key)
    labels = [str(c["label"]) for c in res["fields"]["p"]["candidates"]]
    if len(labels) != n or len(set(labels)) != n:
        sys.exit(f"probe: expected {n} distinct labels, got {labels}")
    return labels


def run_bcj(a, docs, queries, out):
    rng = random.Random(a.seed)
    order = list(range(len(docs)))
    if a.shuffle_docs:
        rng.shuffle(order)
    blocks = list(chunks(order, a.block_size))
    none = not a.no_none
    # scores[q][doc] = (score, within-block prob, block index, position in block)
    scores = {q["id"]: {} for q in queries}
    none_obs = []  # (has_gold, max lp_doc - lp_none) per (query, block): the match detector
    lock = threading.Lock()
    costs = []
    calls = []
    labels = None
    if a.tag_labels:
        # The documents carry the engine's own answer labels, so an answer names
        # its document directly (no label -> number -> document indirection).
        n_labels = len(blocks[0]) + (1 if none else 0)
        labels = [f"L{k}" for k in range(n_labels)] if a.dry_run else probe_labels(a, n_labels)
        a.labels_used = labels
    for bi, block in enumerate(blocks):
        tags = labels[: len(block)] if labels else [str(k + 1) for k in range(len(block))]
        lines = [doc_line(tags[k], docs[di], a.doc_chars) for k, di in enumerate(block)]
        context = ("Labeled" if labels else "Numbered") + " documents:\n\n" + "\n\n".join(lines)
        if len(context) > 100000:
            sys.exit(f"block {bi}: context of {len(context)} characters exceeds the API's 100,000; lower --block-size or --doc-chars")
        choices = tags + (["none"] if none else [])
        for gi, group in enumerate(chunks(list(range(len(queries))), a.fields)):
            schema = {}
            for qi in group:
                q = queries[qi]
                desc = (a.predicate + " Answer with the document's " + ("label" if labels else "number")
                        + (", or none if no document does." if none else ".")
                        + "\nClaim: " + clip(" ".join(q["text"].split()), 3000))
                schema[f"q{qi}"] = {"type": "enum", "description": desc, "choices": choices}
            body = {"context": context, "schema": schema, "score_temperature": a.temperature}
            calls.append((bi, block, gi, group, body, tags))

    if a.dry_run:
        print(f"bcj: {len(docs)} docs x {len(queries)} queries, {len(blocks)} blocks of {a.block_size}, {len(calls)} requests, "
              f"longest context {max(len(c[4]['context']) for c in calls)} chars")
        sys.exit(0)

    def one(call):
        bi, block, gi, group, body, tags = call
        res, wall = post(a.url, body, a.api_key)
        c = cost_fields(res)
        parsed = {}
        for qi in group:
            q = queries[qi]
            cands = res["fields"][f"q{qi}"]["candidates"]
            lp = {str(x["value"]): float(x["logprob"]) for x in cands}
            pr = {str(x["value"]): float(x["probability"]) for x in cands}
            lp_none = lp.get("none")
            best = -math.inf
            with lock:
                for k, di in enumerate(block):
                    v = tags[k]
                    s = lp[v] - lp_none if lp_none is not None else lp[v]
                    best = max(best, lp[v])
                    scores[q["id"]][docs[di]["id"]] = (s, pr[v], bi, k, lp[v])
                if lp_none is not None:
                    has_gold = any(docs[di]["id"] in q["gold"] for di in block)
                    none_obs.append((has_gold, best - lp_none))  # "some document matches" score
            parsed[q["id"]] = {"argmax": res["fields"][f"q{qi}"]["selected_value"], "p_argmax": res["fields"][f"q{qi}"]["probability"],
                               "p_none": pr.get("none")}
        with lock:
            costs.append(c)
            out.write(json.dumps({"mode": "bcj", "tag_labels": bool(labels), "block": bi, "group": gi, "block_docs": [docs[di]["id"] for di in block],
                                  "queries": [queries[qi]["id"] for qi in group], "wall_s": round(wall, 3), "cost": c,
                                  "parsed": parsed}) + "\n")
            out.flush()
        return bi

    t0 = time.perf_counter()
    # Requests of one block share its root. The first group runs alone, so the
    # root is computed and checkpointed once (two jobs with the same uncached
    # prefix in one pass each compute it); the other groups then run together
    # and hit the checkpoint.
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for bi in range(len(blocks)):
            block_calls = [c for c in calls if c[0] == bi]
            one(block_calls[0])
            list(ex.map(one, block_calls[1:]))
    wall = time.perf_counter() - t0
    return scores, none_obs, costs, wall, len(blocks)


# ----------------------------------------------------------------------------- pairwise


def run_pairwise(a, docs, queries, out, pairs=None):
    """Every query against every document, or, with pairs (doc id -> query
    indices), only those pairs (the verification stage of a BCJ cascade)."""
    scores = {q["id"]: {} for q in queries}
    lock = threading.Lock()
    costs = []
    calls = []
    for di, d in enumerate(docs):
        context = "Document:\n\n" + doc_line(1, d, a.doc_chars)[len("[1] "):]
        qidx = list(range(len(queries))) if pairs is None else pairs.get(d["id"], [])
        for gi, group in enumerate(chunks(qidx, a.fields)):
            schema = {}
            for qi in group:
                q = queries[qi]
                schema[f"q{qi}"] = {"type": "boolean", "description": a.pair_predicate + "\nClaim: " + clip(" ".join(q["text"].split()), 3000)}
            calls.append((di, gi, group, {"context": context, "schema": schema, "score_temperature": a.temperature}))
    if a.dry_run:
        print(f"{a.mode}: {len(docs)} docs, {sum(len(c[2]) for c in calls)} pairs, {len(calls)} requests")
        sys.exit(0)

    def one(call):
        di, gi, group, body = call
        res, wall = post(a.url, body, a.api_key)
        c = cost_fields(res)
        with lock:
            for qi in group:
                q = queries[qi]
                cands = res["fields"][f"q{qi}"]["candidates"]
                lp = {str(x["value"]).lower(): float(x["logprob"]) for x in cands}
                pr = {str(x["value"]).lower(): float(x["probability"]) for x in cands}
                scores[q["id"]][docs[di]["id"]] = (lp["true"] - lp["false"], pr["true"], -1, 0, lp["true"])
            costs.append(c)
            out.write(json.dumps({"mode": a.mode, "doc": docs[di]["id"], "group": gi,
                                  "queries": [queries[qi]["id"] for qi in group], "wall_s": round(wall, 3), "cost": c}) + "\n")
            out.flush()

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        list(ex.map(one, calls))
    return scores, [], costs, time.perf_counter() - t0, 0


def verify_candidates(a, corpus, qtext, gold):
    """The top-k documents of every query of a bcj run (--from DIR/scores.jsonl):
    the pairs a cascade re-scores pairwise. Returns docs, queries, pairs (doc id
    -> query indices) and the bcj run's summary."""
    with open(os.path.join(a.from_dir, "summary.json"), encoding="utf-8") as f:
        bcj_summary = json.load(f)
    queries, pairs, need = [], {}, []
    with open(os.path.join(a.from_dir, "scores.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["query"])
            if qid not in qtext:
                sys.exit(f"query {qid} of {a.from_dir} is not in the BEIR data")
            qi = len(queries)
            queries.append({"id": qid, "text": qtext[qid], "gold": set(gold.get(qid, ())) | set(r["gold"])})
            ranked = sorted(r["ranked"], key=lambda x: -x["score"])[: a.top_k]
            for x in ranked:
                pairs.setdefault(str(x["doc"]), []).append(qi)
                need.append(str(x["doc"]))
    docs = [corpus[d] for d in sorted(set(need))]
    return docs, queries, pairs, bcj_summary


def join_decisions(scores, queries, thresholds=(0.5, 0.7, 0.9)):
    """Precision and recall of accepting a (query, doc) pair when its verified
    probability passes a threshold; recall over every gold pair of the queries,
    so candidates the bcj stage dropped count as misses."""
    total_gold = sum(len(q["gold"]) for q in queries)
    out = {}
    for t in thresholds:
        acc = tp = 0
        for q in queries:
            for d, v in scores[q["id"]].items():
                if v[1] >= t:
                    acc += 1
                    tp += d in q["gold"]
        out[str(t)] = {"accepted": acc, "precision": round(tp / acc, 4) if acc else None, "recall": round(tp / total_gold, 4)}
    # one answer per query: the best verified candidate
    acc = tp = 0
    for q in queries:
        if scores[q["id"]]:
            acc += 1
            tp += max(scores[q["id"]].items(), key=lambda kv: kv[1][0])[0] in q["gold"]
    out["top1"] = {"accepted": acc, "precision": round(tp / acc, 4) if acc else None, "recall": round(tp / total_gold, 4)}
    return out


# ----------------------------------------------------------------------------- metrics


def auc(obs):
    """Mann-Whitney AUC of score as a detector of has_gold=True."""
    pos = sorted(s for g, s in obs if g)
    neg = sorted(s for g, s in obs if not g)
    if not pos or not neg:
        return None
    import bisect
    wins = 0.0
    for s in pos:
        lo = bisect.bisect_left(neg, s)
        hi = bisect.bisect_right(neg, s)
        wins += lo + 0.5 * (hi - lo)
    return wins / (len(pos) * len(neg))


def metrics(scores, queries, docs, ks=(1, 3, 5, 10, 20), key=0):
    """key: index of the score used for ranking (0: anchored score, 4: raw label logprob)."""
    n = 0
    rec = {k: 0.0 for k in ks}
    mrr = 0.0
    within_top1 = [0, 0]  # hits, blocks with gold
    pos_bins = {}  # position quartile -> [hits, total] within block
    gold_rank_within = []
    for q in queries:
        sc = scores[q["id"]]
        if not sc or not q["gold"]:
            continue
        n += 1
        ranked = sorted(sc.items(), key=lambda kv: -kv[1][key])
        ids = [d for d, _ in ranked]
        first = next((r for r, d in enumerate(ids) if d in q["gold"]), None)
        if first is not None:
            mrr += 1.0 / (first + 1)
        for k in ks:
            hit = len([d for d in ids[:k] if d in q["gold"]])
            rec[k] += hit / len(q["gold"])
        # within-block statistics (bcj only: block index and position recorded)
        by_block = {}
        for d, v in sc.items():
            if v[2] < 0:
                continue  # pairwise: no blocks
            by_block.setdefault(v[2], []).append((d, v[key], v[3]))
        for b, items in by_block.items():
            golds = [(d, s, k) for d, s, k in items if d in q["gold"]]
            if not golds:
                continue
            within_top1[1] += 1
            best = max(items, key=lambda x: x[1])
            hit = best[0] in q["gold"]
            within_top1[0] += hit
            for d, s, k in golds:
                B = len(items)
                quart = min(3, (4 * k) // max(1, B))
                pb = pos_bins.setdefault(quart, [0, 0])
                pb[0] += hit
                pb[1] += 1
                gold_rank_within.append(1 + sum(1 for _, s2, _ in items if s2 > s))
    if n == 0:
        return {}
    out = {"queries": n, "docs": len(docs), "mrr": round(mrr / n, 4)}
    for k in ks:
        out[f"recall@{k}"] = round(rec[k] / n, 4)
    if within_top1[1]:
        out["within_block_top1"] = round(within_top1[0] / within_top1[1], 4)
        out["within_block_gold_rank_mean"] = round(sum(gold_rank_within) / len(gold_rank_within), 2)
        out["within_block_top1_by_gold_quartile"] = {f"q{qt + 1}": (round(h / t, 3) if t else None, t)
                                                     for qt, (h, t) in sorted(pos_bins.items())}
    return out


# ----------------------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--api-key")
    ap.add_argument("--mode", choices=["bcj", "pairwise", "verify"], default="bcj",
                    help="bcj: blocks as roots; pairwise: every pair; verify: re-score pairwise the top-k of a bcj run (--from)")
    ap.add_argument("--from", dest="from_dir", metavar="DIR", help="verify: the bcj run whose top-k to re-score")
    ap.add_argument("--top-k", type=int, default=5, help="verify: candidates per query")
    ap.add_argument("--dry-run", action="store_true", help="plan the requests and exit")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--beir", metavar="DIR", help="BEIR dataset directory (corpus.jsonl, queries.jsonl, qrels/test.tsv)")
    src.add_argument("--synthetic", nargs=2, type=int, metavar=("N_DOCS", "N_QUERIES"))
    ap.add_argument("--limit-queries", type=int, default=0)
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--block-size", type=int, default=32, help="documents per block (bcj)")
    ap.add_argument("--fields", type=int, default=32, help="queries per request (<= 64)")
    ap.add_argument("--doc-chars", type=int, default=1800, help="characters kept per document")
    ap.add_argument("--no-none", action="store_true", help="no 'none' option in the blocks")
    ap.add_argument("--tag-labels", action="store_true",
                    help="bcj: tag the documents with the engine's answer labels (probed from the server) and offer "
                         "those labels as the choices, instead of numbers that the options dict maps to labels")
    ap.add_argument("--no-shuffle", dest="shuffle_docs", action="store_false")
    ap.add_argument("--temperature", type=float, default=1.0, help="score_temperature of the decisions API")
    ap.add_argument("--predicate", default="Which numbered document provides scientific evidence that supports or refutes the following claim?")
    ap.add_argument("--pair-predicate", default="Does the document provide scientific evidence that supports or refutes the following claim?")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if not 1 <= a.fields <= 64:
        sys.exit("--fields: 1 to 64")
    if a.mode == "bcj" and not 1 <= a.block_size <= 254:
        sys.exit("--block-size: 1 to 254 (255 choices with 'none')")

    bcj_summary = None
    if a.mode == "verify":
        if not (a.beir and a.from_dir):
            sys.exit("--mode verify needs --beir and --from")
        docs, queries, pairs, bcj_summary = verify_candidates(a, *load_beir_index(a.beir))
    elif a.beir:
        docs, queries = load_beir(a.beir, a.limit_queries, a.limit_docs, a.seed)
    else:
        docs, queries = synthetic(a.synthetic[0], a.synthetic[1], a.seed)
        if a.limit_queries:
            queries = queries[: a.limit_queries]
        if a.predicate.startswith("Which numbered document provides scientific"):
            a.predicate = "Which numbered document describes the same company as the following statement?"
            a.pair_predicate = "Does the document describe the same company as the following statement?"
    if a.tag_labels:
        if a.mode != "bcj":
            sys.exit("--tag-labels: bcj mode only")
        a.predicate = a.predicate.replace("numbered document", "labeled document")
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "requests.jsonl"), "w", encoding="utf-8") as out:
        if a.mode == "bcj":
            scores, none_obs, costs, wall, n_blocks = run_bcj(a, docs, queries, out)
        elif a.mode == "verify":
            scores, none_obs, costs, wall, n_blocks = run_pairwise(a, docs, queries, out, pairs)
        else:
            scores, none_obs, costs, wall, n_blocks = run_pairwise(a, docs, queries, out)
    with open(os.path.join(a.out, "scores.jsonl"), "w", encoding="utf-8") as f:
        for q in queries:
            ranked = sorted(scores[q["id"]].items(), key=lambda kv: -kv[1][0])
            f.write(json.dumps({"query": q["id"], "gold": sorted(q["gold"]),
                                "ranked": [{"doc": d, "score": round(v[0], 4), "p": round(v[1], 5), "block": v[2], "pos": v[3],
                                            "logprob": round(v[4], 4)}
                                           for d, v in ranked]}) + "\n")
    m = metrics(scores, queries, docs)
    m_raw = metrics(scores, queries, docs, key=4) if a.mode == "bcj" and not a.no_none else None
    cost = {"requests": len(costs), "wall_s": round(wall, 1),
            "server_elapsed_s": round(sum(float(c["elapsed_ms"] or 0) for c in costs) / 1000, 1),
            "root_tokens_computed": sum(c["root_tokens_computed"] for c in costs),
            "leaf_tokens": sum(c["leaf_tokens"] for c in costs),
            "cache": {}}
    for c in costs:
        cost["cache"][str(c["cache"])] = cost["cache"].get(str(c["cache"]), 0) + 1
    cost["tokens_total"] = cost["root_tokens_computed"] + cost["leaf_tokens"]
    if wall:
        cost["tokens_per_s"] = round(cost["tokens_total"] / wall, 1)
    summary = {"mode": a.mode, "model_url": a.url, "docs": len(docs), "queries": len(queries),
               "block_size": a.block_size if a.mode == "bcj" else None, "blocks": n_blocks, "fields_per_request": a.fields,
               "none_option": (not a.no_none) if a.mode == "bcj" else None,
               "tag_labels": a.tag_labels if a.mode == "bcj" else None, "labels": getattr(a, "labels_used", None),
               "temperature": a.temperature,
               "metrics": m, "metrics_raw_logprob": m_raw, "cost": cost}
    if a.mode == "verify":
        bc = bcj_summary["cost"]
        pairs_all = bcj_summary["queries"] * bcj_summary["docs"]
        summary["verify"] = {"from": a.from_dir, "top_k": a.top_k, "pairs": sum(len(v) for v in pairs.values()),
                             "join_decisions": join_decisions(scores, queries),
                             "cascade_cost": {"bcj_tokens": bc["tokens_total"], "verify_tokens": cost["tokens_total"],
                                              "total_tokens": bc["tokens_total"] + cost["tokens_total"],
                                              "tokens_per_pair": round((bc["tokens_total"] + cost["tokens_total"]) / pairs_all, 2),
                                              "pairwise_tokens_per_pair_est": None}}
    if none_obs:
        a_ = auc(none_obs)  # AUC of (max doc logprob - none logprob) as a "block holds a match" detector
        summary["match_detector_auc"] = None if a_ is None else round(a_, 4)
        summary["blocks_with_gold_fraction"] = round(sum(1 for g, _ in none_obs if g) / len(none_obs), 4)
    with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
