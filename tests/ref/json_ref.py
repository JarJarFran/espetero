# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

import json, random, math, sys
random.seed(7)
def rnd_str():
    pool = ["a","Z"," ","\n","\t","\"","\\","/","é","ñ","😀","\x00","\x1f","\x7f","中","\u2028","}", "{", ","]
    return "".join(random.choice(pool) for _ in range(random.randint(0,12)))
def rnd(depth=0):
    r = random.random()
    if depth > 3 or r < 0.3:
        return random.choice([None, True, False, random.randint(-10**12, 10**12), rnd_str(),
            random.choice([0.0, -0.0, 1.0, 1e-05, 1.5e-7, 123456.789, 1e16, 1e15, 3.14159, 2.5e-300, 1e300, 0.1, 100.0, -2.0, 1/3])])
    if r < 0.65:
        return [rnd(depth+1) for _ in range(random.randint(0,4))]
    d = {}
    for _ in range(random.randint(0,4)):
        d[rnd_str()] = rnd(depth+1)
    return d
cases = []
for i in range(3000):
    v = rnd()
    for opts in [dict(item_sep=",", key_sep=":", ensure_ascii=False), dict(item_sep=", ", key_sep=": ", ensure_ascii=True),
                 dict(item_sep=", ", key_sep=": ", ensure_ascii=False, indent=2), dict(item_sep=",", key_sep=":", ensure_ascii=False, sort_keys=True)]:
        exp = json.dumps(v, ensure_ascii=opts["ensure_ascii"], separators=(opts["item_sep"], opts["key_sep"]) if "indent" not in opts else None, indent=opts.get("indent"), sort_keys=opts.get("sort_keys", False))
        cases.append({"value": v, "opts": opts, "expect": exp})
json.dump(cases, open(sys.argv[1], "w"), ensure_ascii=False)
