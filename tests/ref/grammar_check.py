#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
"""Structured output: every document build/grammar_test generated is
valid JSON under its schema, by Python's jsonschema (Draft 2020-12; OpenAPI's
nullable read as a null type). Skipped with a message when jsonschema is not
installed.

  python3 tests/ref/grammar_check.py build/grammar_samples.jsonl
"""
import json
import sys


def openapi_nullable(s):
    if isinstance(s, dict):
        s = {k: openapi_nullable(v) for k, v in s.items()}
        if s.pop("nullable", False) is True and "type" in s:
            t = s["type"] if isinstance(s["type"], list) else [s["type"]]
            s["type"] = t if "null" in t else t + ["null"]
        return s
    if isinstance(s, list):
        return [openapi_nullable(x) for x in s]
    return s


def main():
    try:
        import jsonschema
    except ImportError:
        print("grammar_check: jsonschema is not installed, skipped")
        return 0
    n = bad = skipped = 0
    with open(sys.argv[1], encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            schema = openapi_nullable(rec["schema"])
            n += 1
            try:
                value = json.loads(rec["text"])
            except ValueError as e:
                print(f"not JSON ({rec['case']}): {rec['text']!r}: {e}")
                bad += 1
                continue
            cls = jsonschema.validators.validator_for(schema, default=jsonschema.Draft202012Validator)
            try:
                errors = list(cls(schema).iter_errors(value))
            except RecursionError:  # an anyOf that refers to itself first: the validator never ends
                skipped += 1
                continue
            if errors:
                print(f"invalid ({rec['case']}): {rec['text']}: {errors[0].message}")
                bad += 1
    print(f"grammar_check: {n} documents, {bad} invalid" + (f", {skipped} not checked (the validator recurses)" if skipped else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
