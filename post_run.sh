#!/bin/sh
# Re-judge every row the last full run could not decide, with the current
# front-end, then regenerate both reports.  Rows are keyed by index and the
# report takes the latest record for each, so a re-judge overrides in place.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
KB=$(python3 -c "
import json
rs = {}
for l in open('data/kb_live.jsonl'):
    r = json.loads(l); rs[r['i']] = r
print(','.join(str(i) for i, r in sorted(rs.items()) if r['verdict'] not in ('PASS', 'FAIL')))")
echo "re-judging $(echo $KB | tr ',' '\n' | wc -l) KernelBook rows"
python3 -u -m tvj.judge.kernelbook_run --rows "$KB" > results/kb_rejudge.log 2>&1

TR=$(python3 -c "
import json, os
rs = {}
p = 'results/triton_traces.jsonl'
for l in (open(p) if os.path.exists(p) else []):
    r = json.loads(l); rs[r['i']] = r
print(','.join(str(i) for i, r in sorted(rs.items()) if r['verdict'] not in ('PASS', 'FAIL')))")
echo "re-judging $(echo $TR | tr ',' '\n' | wc -l) trace rows"
python3 -u -m tvj.judge.traces_run --rows "$TR" > results/tr_rejudge.log 2>&1

python3 -m tvj.judge.report both > results/report.txt 2>&1
echo POSTRUN-DONE
