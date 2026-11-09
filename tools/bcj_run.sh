#!/bin/zsh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
#
# Runs tools/bcj_join.py against a fresh espetero server, inside one gputurn job:
#
#   gputurn submit -s --model MODEL -- zsh tools/bcj_run.sh MODEL PORT OUTDIR [serve flags...] -- [bcj_join.py args...]
#
# Starts `espetero serve --model MODEL --port PORT` (plus the serve flags), waits
# for /health, runs the harness with --url and --out set, then stops the server.
# The server's log goes to OUTDIR/server.log. Exit code: the harness's.
set -u
MODEL=$1; PORT=$2; OUT=$3; shift 3
SERVE=()
while [[ $# -gt 0 && $1 != "--" ]]; do SERVE+=("$1"); shift; done
[[ ${1:-} == "--" ]] && shift
HERE=${0:A:h}
BIN=${ESPETERO_BIN:-$HERE/../build/espetero}
mkdir -p "$OUT"
echo "[bcj_run] $BIN serve --model $MODEL --port $PORT --no-webui ${SERVE[*]}" | tee "$OUT/server.log"
"$BIN" serve --model "$MODEL" --port "$PORT" --no-webui "${SERVE[@]}" >> "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 600); do
  if curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then break; fi
  if ! kill -0 $SRV 2>/dev/null; then echo "[bcj_run] server died; log:"; tail -40 "$OUT/server.log"; exit 70; fi
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null || { echo "[bcj_run] server not healthy after 600 s"; tail -40 "$OUT/server.log"; exit 70; }
echo "[bcj_run] server up after ${i}s; running harness: $*"
python3 -I "$HERE/bcj_join.py" --url "http://127.0.0.1:$PORT" --out "$OUT" "$@"
RC=$?
echo "[bcj_run] harness exit $RC"
exit $RC
