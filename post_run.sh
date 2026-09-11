#!/bin/sh
# Re-judge every row the last full run could not decide, with the current
# front-end, then regenerate both reports.  Rows are keyed by index and the
# report takes the latest record for each, so a re-judge overrides in place.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. ./_python.sh
KB=$("$PY" -c "
import json, os
rs = {}
# published record first, then the one the last run appended to: same precedence
# as tvj/judge/report.py, so a clean clone can re-judge from what is committed
for p in ('results/kernelbook.jsonl', 'data/kb_live.jsonl'):
    for l in (open(p) if os.path.exists(p) else []):
        r = json.loads(l); rs[r['i']] = r
print(','.join(str(i) for i, r in sorted(rs.items()) if r['verdict'] not in ('PASS', 'FAIL')))")
echo "re-judging $(echo $KB | tr ',' '\n' | wc -l) KernelBook rows"
"$PY" -u -m tvj.judge.kernelbook_run --rows "$KB" > results/kb_rejudge.log 2>&1

TR=$("$PY" -c "
import json, os
rs = {}
p = 'results/triton_traces.jsonl'
for l in (open(p) if os.path.exists(p) else []):
    r = json.loads(l); rs[r['i']] = r
print(','.join(str(i) for i, r in sorted(rs.items()) if r['verdict'] not in ('PASS', 'FAIL')))")
echo "re-judging $(echo $TR | tr ',' '\n' | wc -l) trace rows"
"$PY" -u -m tvj.judge.traces_run --rows "$TR" > results/tr_rejudge.log 2>&1

# Publish the KernelBook record beside the LLM one.  `data/` is generated and
# gitignored, so leaving it there was why a clean clone could regenerate half the
# Limits table and half the report -- see tvj/measure/limits.py load().
[ -f data/kb_live.jsonl ] && cp data/kb_live.jsonl results/kernelbook.jsonl
"$PY" -m tvj.judge.report both > results/report.txt 2>&1
echo POSTRUN-DONE
