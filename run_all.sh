#!/bin/sh
# Both corpora, end to end, restarting after a poisoned CUDA context.
#
# A device-side assert (torch's BCE asserts 0 <= input <= 1, and the D4
# sign-flip check feeds it negatives) kills the context for the whole process:
# every later launch fails the same way.  One run produced 114 consecutive bogus
# ERROR rows before the judge learned to stop.  It stops now, and this restarts
# it from the next index in a fresh process -- which also frees the term pool.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export VOLTA_MEM_GB=${VOLTA_MEM_GB:-4}
N_KB=$(python3 -c "import json;print(len(json.load(open('data/kernelbook_400.json'))))")
N_TR=$(python3 -c "import json;print(sum(1 for r in json.load(open('data/triton_traces.json')) if r['source']=='kernelbook'))")

i=0; tries=0
while [ "$i" -lt "$N_KB" ] && [ "$tries" -lt 12 ]; do
  echo "== kernelbook from $i =="
  python3 -u -m tvj.judge.kernelbook_run "$i" $((N_KB - i)) >> results/kb_full.log 2>&1
  nxt=$(grep -o '^!! row [0-9]*' results/kb_full.log | tail -1 | awk '{print $3}')
  [ -z "$nxt" ] && break
  [ "$((nxt + 1))" -le "$i" ] && break
  i=$((nxt + 1)); tries=$((tries + 1))
done

i=0; tries=0
while [ "$i" -lt "$N_TR" ] && [ "$tries" -lt 12 ]; do
  echo "== traces from $i =="
  python3 -u -m tvj.judge.traces_run "$i" $((N_TR - i)) >> results/tr_full.log 2>&1
  nxt=$(grep -o '^!! row [0-9]*' results/tr_full.log | tail -1 | awk '{print $3}')
  [ -z "$nxt" ] && break
  [ "$((nxt + 1))" -le "$i" ] && break
  i=$((nxt + 1)); tries=$((tries + 1))
done

python3 -m tvj.judge.report both > results/report.txt 2>&1
echo RUN-ALL-DONE
