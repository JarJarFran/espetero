#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Compares bcj_join.py runs: block-size arms against each other and against
the pairwise baseline.

    python3 -I tools/bcj_analyze.py RUN_DIR [RUN_DIR...] [--json OUT]

Each RUN_DIR holds a bcj_join.py output (summary.json, scores.jsonl,
requests.jsonl). Prints:
  1. one row per arm: cost (root/leaf tokens, tokens per candidate pair,
     wall) and quality (recall@k, MRR, within-block top-1, match-detector AUC);
  2. within-block position bias (top-1 rate by the gold's position quartile);
  3. the leaf-token model: tokens per field = a + b * choices, fitted across the
     bcj arms, and the cost a native BCJ path (options in the stem, not in
     every leaf) would have;
  4. when a pairwise run is given: every bcj arm restricted to the pairwise
     run's queries and documents, next to the pairwise numbers;
  5. the verification budget: pairs to re-score pairwise to keep BCJ's top-k,
     as a fraction of all pairs.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bcj_join  # noqa: E402  (metrics)

KS = (1, 3, 5, 10, 20)


def load_run(d):
    with open(os.path.join(d, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    scores, queries, docs = {}, [], set()
    with open(os.path.join(d, "scores.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            queries.append({"id": r["query"], "gold": set(r["gold"])})
            sc = {}
            for x in r["ranked"]:
                sc[x["doc"]] = (x["score"], x["p"], x.get("block", -1), x.get("pos", 0), x.get("logprob", x["score"]))
                docs.add(x["doc"])
            scores[r["query"]] = sc
    reqs = []
    p = os.path.join(d, "requests.jsonl")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                c = r["cost"]
                reqs.append({"fields": c["fields"], "leaf": c["leaf_tokens"], "root": c["root_tokens_computed"],
                             "cache": c["cache"], "wall": r["wall_s"], "prefix": c["prefix_tokens"]})
    name = os.path.basename(os.path.normpath(d))
    return {"name": name, "dir": d, "summary": summary, "scores": scores, "queries": queries,
            "docs": sorted(docs), "reqs": reqs, "mode": summary["mode"], "B": summary.get("block_size")}


def fmt(x, w=6, nd=3):
    if x is None:
        return " " * w
    if isinstance(x, float):
        return f"{x:{w}.{nd}f}"
    return f"{x:>{w}}"


def table(rows, header):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    out = ["  ".join(str(h).ljust(w) for h, w in zip(header, widths)),
           "  ".join("-" * w for w in widths)]
    out += ["  ".join(str(c).ljust(w) for c, w in zip(r, widths)) for r in rows]
    return "\n".join(out)


def arm_row(run):
    s, m, c = run["summary"], run["summary"]["metrics"], run["summary"]["cost"]
    pairs = s["queries"] * s["docs"]
    tot = c["tokens_total"]
    label = run["B"] if run["mode"] == "bcj" else ("pair" if run["mode"] == "pairwise" else f"verify k={s['verify']['top_k']}")
    return [run["name"], label, s["queries"], s["docs"], s.get("blocks"), c["requests"],
            c["root_tokens_computed"], c["leaf_tokens"], tot, f"{tot / pairs:.1f}", f"{c['wall_s']:.0f}", f"{c['tokens_per_s']:.0f}",
            *(fmt(m.get(f"recall@{k}"), 5) for k in KS), fmt(m.get("mrr"), 5),
            fmt(m.get("within_block_top1"), 5), fmt(s.get("match_detector_auc"), 5)]


def leaf_model(runs):
    """Tokens per field against the number of choices (B + 1 with none), least squares."""
    pts = []
    for r in runs:
        if r["mode"] != "bcj" or not r["reqs"]:
            continue
        fields = sum(q["fields"] for q in r["reqs"])
        leaf = sum(q["leaf"] for q in r["reqs"])
        choices = r["B"] + (1 if r["summary"].get("none_option") else 0)
        pts.append((r["name"], choices, leaf / fields, fields, r["summary"]["cost"]["root_tokens_computed"],
                    r["summary"]["queries"] * r["summary"]["docs"]))
    if not pts:
        return None, []
    fit = None
    if len({p[1] for p in pts}) >= 2:
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
        a = my - b * mx
        fit = (a, b)
    return fit, pts


def restricted(run, qids, dset):
    sc = {q: {d: v for d, v in run["scores"][q].items() if d in dset} for q in qids if q in run["scores"]}
    qs = [q for q in run["queries"] if q["id"] in sc]
    return bcj_join.metrics(sc, qs, sorted(dset), ks=KS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--json", help="also write the computed tables here")
    a = ap.parse_args()
    runs = [load_run(d) for d in a.dirs]
    runs.sort(key=lambda r: (r["mode"] != "bcj", r["mode"] == "verify", r["B"] or 0))
    bcj = [r for r in runs if r["mode"] == "bcj"]
    pair = [r for r in runs if r["mode"] == "pairwise"]
    verify = [r for r in runs if r["mode"] == "verify"]
    out = {}

    print("1. Arms")
    hdr = ["arm", "B", "|L|", "|R|", "blocks", "reqs", "root tok", "leaf tok", "total", "tok/pair", "wall s", "tok/s",
           *(f"R@{k}" for k in KS), "MRR", "wb-top1", "AUC"]
    rows = [arm_row(r) for r in runs]
    print(table(rows, hdr))
    out["arms"] = [dict(zip(hdr, r)) for r in rows]
    raw = [(r["name"], r["summary"].get("metrics_raw_logprob")) for r in bcj if r["summary"].get("metrics_raw_logprob")]
    if raw:
        print("\n   Ablation, ranking by the raw label logprob instead of logprob(label) - logprob(none):")
        print(table([[n, *(fmt(m.get(f"recall@{k}"), 5) for k in KS), fmt(m.get("mrr"), 5), fmt(m.get("within_block_top1"), 5)]
                     for n, m in raw], ["arm", *(f"R@{k}" for k in KS), "MRR", "wb-top1"]))

    if bcj:
        print("\n2. Within-block position bias: top-1 rate when the gold sits in each quartile of the block (hits/blocks)")
        rows = []
        for r in bcj:
            qb = r["summary"]["metrics"].get("within_block_top1_by_gold_quartile") or {}
            rows.append([r["name"], *(f"{qb[q][0]} ({qb[q][1]})" if q in qb and qb[q][0] is not None else "-" for q in ("q1", "q2", "q3", "q4")),
                         fmt(r["summary"]["metrics"].get("within_block_gold_rank_mean"), 5, 2),
                         fmt(r["summary"].get("blocks_with_gold_fraction"), 5)])
        print(table(rows, ["arm", "q1 (first)", "q2", "q3", "q4 (last)", "gold rank", "blocks w/ gold"]))
        out["position_bias"] = rows

    fit, pts = leaf_model(runs)
    if pts:
        print("\n3. Leaf cost: tokens per field against the choices it lists (the options dict is repeated in every leaf)")
        rows = [[n, ch, f"{t:.1f}", f, root, pairs] for n, ch, t, f, root, pairs in pts]
        print(table(rows, ["arm", "choices", "tok/field", "fields", "root tok", "pairs"]))
        if fit:
            a_, b_ = fit
            print(f"   fit: tokens/field = {a_:.1f} + {b_:.2f} x choices  (a: claim + instructions; b: per option)")
            print("   Native BCJ (options once in the stem, leaves carry the claim only): root + fields x a")
            rows = []
            for n, ch, t, f, root, pairs in pts:
                measured = root + t * f
                native = root + a_ * f
                rows.append([n, f"{measured:.0f}", f"{native:.0f}", f"{measured / pairs:.1f}", f"{native / pairs:.1f}", f"{measured / native:.2f}x"])
            print(table(rows, ["arm", "measured tok", "native tok", "tok/pair", "native tok/pair", "saving"]))
            out["leaf_model"] = {"a": a_, "b": b_, "rows": rows}

    if pair and bcj:
        p = pair[0]
        qids = [q["id"] for q in p["queries"]]
        # The random fill of the pairwise subset differs from the bcj one, so
        # both sides rank the documents every run scored (the shared gold and
        # fill): same claims, same distractors. Costs stay those of the full runs.
        dset = set(p["docs"])
        for r in bcj:
            dset &= set(r["docs"])
        print(f"\n4. Same claims and documents: {len(qids)} claims x {len(dset)} docs scored by the pairwise run and every bcj arm "
              f"(pairwise ran {len(p['docs'])} docs); each run restricted to them")
        rows = []
        pm = restricted(p, qids, dset) or {}
        pc = p["summary"]["cost"]
        p_pairs = len(qids) * len(p["docs"])
        rows.append([p["name"], *(fmt(pm.get(f"recall@{k}"), 5) for k in KS), fmt(pm.get("mrr"), 5), f"{pc['tokens_total'] / p_pairs:.1f}", "1.00x"])
        for r in bcj:
            m = restricted(r, qids, dset)
            if not m:
                continue
            tp = r["summary"]["cost"]["tokens_total"] / (r["summary"]["queries"] * r["summary"]["docs"])
            rows.append([r["name"], *(fmt(m.get(f"recall@{k}"), 5) for k in KS), fmt(m.get("mrr"), 5), f"{tp:.1f}",
                         f"{(pc['tokens_total'] / p_pairs) / tp:.2f}x"])
        print(table(rows, ["arm", *(f"R@{k}" for k in KS), "MRR", "tok/pair", "pairwise/bcj"]))
        out["restricted"] = rows
        gold = set()
        for q in p["queries"]:
            gold |= set(q["gold"])
        print(f"   (shared docs: {len(dset & gold)} gold of these claims + {len(dset - gold)} fill; the bcj scores come from "
              f"blocks of the full {bcj[0]['summary']['docs']}-doc runs, the pairwise ones from single-document roots)")

    if bcj:
        print("\n5. Verification budget: re-scoring BCJ's top-k pairwise keeps recall@k; pairs = |L| x k, as a share of |L| x |R|")
        rows = []
        for r in bcj:
            s, m = r["summary"], r["summary"]["metrics"]
            L, R = s["queries"], s["docs"]
            rows.append([r["name"], *(f"R@{k} {m.get(f'recall@{k}', 0):.3f} @ {100 * k / R:.1f}%" for k in (1, 5, 10, 20))])
        print(table(rows, ["arm", "k=1", "k=5", "k=10", "k=20"]))
        out["verification"] = rows

    if verify:
        print("\n6. Cascade: BCJ top-k re-scored pairwise; join output = pairs whose verified probability passes the threshold")
        rows = []
        for r in verify:
            v = r["summary"]["verify"]
            jd, cc = v["join_decisions"], v["cascade_cost"]
            for t, d in jd.items():
                rows.append([r["name"], os.path.basename(os.path.normpath(v["from"])), v["top_k"], v["pairs"], t, d["accepted"],
                             fmt(d["precision"], 5), fmt(d["recall"], 5), cc["bcj_tokens"], cc["verify_tokens"], cc["tokens_per_pair"]])
        print(table(rows, ["run", "from", "k", "pairs", "accept if p >=", "accepted", "precision", "recall", "bcj tok", "verify tok", "cascade tok/pair"]))
        out["cascade"] = rows

    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
