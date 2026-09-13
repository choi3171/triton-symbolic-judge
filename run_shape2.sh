#!/bin/sh
# The LLM corpus at a second shape (tvj/judge/shape2_run.py), resumed until complete.
# One process at a time; the python retires itself every 1200 s and is restarted
# here, so `timeout` below is a backstop, not the normal stop.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. ./_python.sh
export VOLTA_MEM_GB=${VOLTA_MEM_GB:-4} TVJ_ROW_TIMEOUT=${TVJ_ROW_TIMEOUT:-90} CARGO_BUILD_JOBS=2
LOG=results/tr_shape2.log
for attempt in $(seq 1 60); do
  timeout 1500 "$PY" -u -m tvj.judge.shape2_run >> "$LOG" 2>&1; rc=$?
  if [ "$rc" -eq 0 ]; then echo "shape2 complete" >> "$LOG"; exit 0; fi
  echo "-- attempt $attempt ended rc=$rc; resuming" >> "$LOG"; sleep 2
done
echo "shape2: gave up after 60 attempts" >> "$LOG"; exit 1
