#!/bin/zsh
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Francisco José García Barbero
#
# One server, several bcj_join.py runs (a block-size sweep and the pairwise
# baseline), inside one gputurn job:
#
#   gputurn submit -s --model MODEL --est SECONDS -- zsh tools/bcj_sweep.sh MODEL PORT OUTROOT BEIR_DIR [serve flags...]
#
# Environment: BLOCKS="16 32 48" (block sizes), QUERIES=100 DOCS=512 (bcj
# subset), PAIR_QUERIES=30 PAIR_DOCS=128 (pairwise subset; 0 skips it),
# FIELDS=32, CONC=4, SEED=1. Results: OUTROOT/bcj-B<n>/ and OUTROOT/pairwise/.
set -u
MODEL=$1; PORT=$2; OUT=$3; DATA=$4; shift 4
HERE=${0:A:h}
BIN=${ESPETERO_BIN:-$HERE/../build/espetero}
BLOCKS=${BLOCKS:-"16 32 48"}; QUERIES=${QUERIES:-100}; DOCS=${DOCS:-512}
PAIR_QUERIES=${PAIR_QUERIES:-30}; PAIR_DOCS=${PAIR_DOCS:-128}; FIELDS=${FIELDS:-32}; CONC=${CONC:-4}; SEED=${SEED:-1}
mkdir -p "$OUT"
echo "[bcj_sweep] $BIN serve --model $MODEL --port $PORT --no-webui $*" | tee "$OUT/server.log"
"$BIN" serve --model "$MODEL" --port "$PORT" --no-webui "$@" >> "$OUT/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null' EXIT
for i in $(seq 1 900); do
  curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "[bcj_sweep] server died"; tail -40 "$OUT/server.log"; exit 70; }
  sleep 1
done
curl -fsS "http://127.0.0.1:$PORT/health" > /dev/null || { echo "[bcj_sweep] server not healthy after 900 s"; exit 70; }
echo "[bcj_sweep] server up after ${i}s"
RC=0
for B in ${=BLOCKS}; do
  echo "[bcj_sweep] === bcj B=$B queries=$QUERIES docs=$DOCS ==="
  python3 -I "$HERE/bcj_join.py" --url "http://127.0.0.1:$PORT" --beir "$DATA" --limit-queries $QUERIES --limit-docs $DOCS \
    --block-size $B --fields $FIELDS --concurrency $CONC --seed $SEED --out "$OUT/bcj-B$B" || RC=$?
done
if [[ $PAIR_QUERIES -gt 0 ]]; then
  echo "[bcj_sweep] === pairwise queries=$PAIR_QUERIES docs=$PAIR_DOCS ==="
  python3 -I "$HERE/bcj_join.py" --url "http://127.0.0.1:$PORT" --beir "$DATA" --mode pairwise --limit-queries $PAIR_QUERIES \
    --limit-docs $PAIR_DOCS --fields $FIELDS --concurrency $CONC --seed $SEED --out "$OUT/pairwise" || RC=$?
fi
echo "[bcj_sweep] done, exit $RC"
exit $RC
