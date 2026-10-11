# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

"""Reference decision prompts from the jevfire Python stack (HF tokenizer +
transformers chat template): header, prefix/suffix/stem/leaf tokens and label
ids for every benchmark case plus stress cases. Output: JSON for
tests/decisions_test.cpp.

    python tests/ref/decisions_ref.py <model dir> <out.json>
"""
import json
import os
import sys

_JEVFIRE = os.environ.get("JEVFIRE")
if not _JEVFIRE:
    sys.exit("set JEVFIRE to a JEVfire checkout (https://github.com/kikoncuo/jevfire)")
sys.path.insert(0, _JEVFIRE)
from transformers import AutoTokenizer  # noqa: E402

from jevfire.cases import benchmark_cases  # noqa: E402
from jevfire.core import make_labels  # noqa: E402
from jevfire.models import DecisionRequest  # noqa: E402
from jevfire.metal.engine import PromptBuilder  # noqa: E402

model_dir, out_path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model_dir)
labels = make_labels(tok)
builder = PromptBuilder(tok, labels, "")
cases = []
for c in benchmark_cases():
    cases.append({"context": c["context"], "schema": c["schema"]})
# Stress: unicode, quotes, JSON-looking context, long choice lists, many fields.
cases.append({"context": 'Él dijo "hola" \\ {"a": [1, 2]} 😀 \n\t línea', "schema": {
    "q": {"type": "boolean", "description": "¿Es un saludo?"}}})
cases.append({"context": "x" * 3000 + " end", "schema": {
    f"f{i}": {"type": "enum", "description": f"Field number {i} with ünïcode", "choices": [f"c{j}" for j in range(2 + i)]}
    for i in range(12)}})
cases.append({"context": "Choose.", "schema": {"big": {"type": "enum", "description": "255 options",
                                                        "choices": [f"option {j}" for j in range(255)]}}})
out = {"header": builder.header, "labels": [[l.text, l.token_id] for l in labels], "cases": []}
for c in cases:
    req = DecisionRequest.model_validate(c)
    plan = builder.plan(req)
    out["cases"].append({"request": c, "prefix": plan.prefix, "suffixes": plan.suffixes, "stem": plan.stem,
                         "leaves": plan.leaves})
json.dump(out, open(out_path, "w"), ensure_ascii=False)
print(f"{len(out['cases'])} cases, header {len(builder.header)} tokens, {len(labels)} labels")
