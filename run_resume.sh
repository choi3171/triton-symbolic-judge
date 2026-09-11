#!/bin/sh
# Resume a corpus run from wherever it actually got to, whatever stopped it.
#
# run_all.sh only reacted to a poisoned CUDA context.  Two other things stop a
# run and neither leaves that marker:
#   * a hang -- the 150 s alarm can land inside a ctypes call in Triton's driver
#     binding, where it surfaces as ctypes.ArgumentError instead of our Timeout
#     and the row never gets a verdict (row 372, SevenLayerFC_Net);
#   * anything that kills the process outright.
# So: cap each invocation in wall clock, and resume from the last row actually
# WRITTEN plus one, rather than from a marker the run may never have printed.
#
#   ./run_resume.sh kb       KernelBook
#   ./run_resume.sh traces   the LLM corpus
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. ./_python.sh
export VOLTA_MEM_GB=${VOLTA_MEM_GB:-4}
CAP=${CAP:-900}                     # seconds per invocation before it is killed

case "$1" in
  kb) SCRIPT=tvj.judge.kernelbook_run; REC=data/kb_live.jsonl; LOG=results/kb_full.log
      N=$("$PY" -c "import json;print(len(json.load(open('data/kernelbook_400.json'))))") ;;
  traces) SCRIPT=tvj.judge.traces_run; REC=results/triton_traces.jsonl; LOG=results/tr_full.log
      N=$("$PY" -c "import json;print(sum(1 for r in json.load(open('data/triton_traces.json')) if r['source']=='kernelbook'))") ;;
  *) echo "usage: $0 kb|traces"; exit 2 ;;
esac

last() { "$PY" -c "
import json, os
p = '$REC'
m = -1
if os.path.exists(p):
    for l in open(p):
        try: m = max(m, json.loads(l)['i'])
        except Exception: pass
print(m)"; }

# `floor` only ever increases, so a row that hangs and therefore writes no record
# is stepped over instead of being retried forever.
floor=$(( $(last) + 1 ))
skipped=""
while [ "$floor" -lt "$N" ]; do
  echo "== $1 from $floor =="
  # both runners take `start count`
  timeout "$CAP" "$PY" -u -m "$SCRIPT" "$floor" $((N - floor)) >> "$LOG" 2>&1
  next=$(( $(last) + 1 ))
  if [ "$next" -le "$floor" ]; then
    echo "row $floor produced no record (hang or hard kill) -- stepping over it"
    skipped="$skipped $floor"
    next=$(( floor + 1 ))
  fi
  floor=$next
done
echo "$1 complete: last record $(last) of $((N - 1)); stepped over:${skipped:- none}"
