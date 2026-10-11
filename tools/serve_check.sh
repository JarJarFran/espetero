#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero

# Functional check of the three serving roles on one loaded model:
#   1. chat completion (agentic / assistant use, OpenAI-compatible, streaming off)
#   2. plain completion
#   3. native classification (/v1/decisions, the JEV-like structured scoring)
# Starts `espetero serve` on a free port, waits for /v1/models, runs the three
# requests, prints the answers and timings, stops the server.
#   tools/serve_check.sh <model path> [port=8080] [extra espetero serve args...]
set -uo pipefail
cd "$(dirname "$0")/.."
MODEL=${1:?model path}; PORT=${2:-8080}; shift 2 2>/dev/null || shift $#
[ -x build/espetero ] || make -j8 build/espetero >/dev/null
build/espetero serve --model "$MODEL" --port "$PORT" --max-ctx 8192 --slots 8 --kv-gb 2 "$@" > build/serve_check.log 2>&1 &
SP=$!
trap 'kill $SP 2>/dev/null' EXIT
for i in $(seq 1 240); do
  curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
  sleep 1
done
echo "== models"; curl -s "http://127.0.0.1:$PORT/v1/models" | head -c 400; echo
echo "== chat (assistant role)"
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" -H 'content-type: application/json' -d '{
  "messages": [{"role": "system", "content": "You are a concise assistant."},
               {"role": "user", "content": "In one sentence, what does a mixture-of-experts layer do?"}],
  "max_tokens": 80, "temperature": 0}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(r["choices"][0]["message"]["content"][:400]); print("usage:", r.get("usage")); print("timings:", r.get("timings"))'
echo "== completion"
curl -s "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' -d '{
  "prompt": "The three primary colors are", "max_tokens": 24, "temperature": 0}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print(repr(r["choices"][0]["text"][:200]))'
echo "== decisions (native classifier)"
curl -s "http://127.0.0.1:$PORT/v1/decisions" -H 'content-type: application/json' -d '{
  "context": "Customer message: The package arrived two weeks late and the box was crushed, but the product inside works fine.",
  "schema": {"sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"], "description": "overall tone"},
             "product_damaged": {"type": "boolean", "description": "Is the product itself damaged?"},
             "shipping_problem": {"type": "boolean", "description": "Was there a shipping or delivery problem?"}}}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print("parsed:", r.get("parsed_json")); m=r.get("metal",{}); print("metal:", {k:m.get(k) for k in ("cache","computed_tokens","gpu_ms","dispatches")}); print("elapsed_ms:", r.get("elapsed_ms"))'
echo "== second decision (cached header)"
curl -s "http://127.0.0.1:$PORT/v1/decisions" -H 'content-type: application/json' -d '{
  "context": "Customer message: Great service, fast delivery, exactly as described.",
  "schema": {"sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"], "description": "overall tone"},
             "shipping_problem": {"type": "boolean", "description": "Was there a shipping or delivery problem?"}}}' | python3 -c 'import json,sys; r=json.load(sys.stdin); print("parsed:", r.get("parsed_json")); m=r.get("metal",{}); print("metal:", {k:m.get(k) for k in ("cache","computed_tokens","gpu_ms")}); print("elapsed_ms:", r.get("elapsed_ms"))'
echo "== stats"; curl -s "http://127.0.0.1:$PORT/v1/stats" | head -c 300; echo
