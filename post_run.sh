#!/bin/sh
# Re-judge every row the last full run could not decide, with the current
# front-end, then regenerate both reports.  Rows are keyed by index and the
# report takes the latest record for each, so a re-judge overrides in place.
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
. ./_python.sh
# A scratch record left in data/ is a run nobody published; appending a new run to
# it and publishing the mixture is how an old run's verdicts reach the record.
"$PY" -m tvj.judge.record require-clean || exit 1
KB=$("$PY" -c "
from tvj.judge import record
print(','.join(str(r['i']) for r in (record.load('kb') or []) if r['verdict'] not in ('PASS', 'FAIL', 'PASS-ASSUMING')))")
echo "re-judging $(echo $KB | tr ',' '\n' | wc -l) KernelBook rows"
"$PY" -u -m tvj.judge.kernelbook_run --rows "$KB" > results/kb_rejudge.log 2>&1

TR=$("$PY" -c "
from tvj.judge import record
print(','.join(str(r['i']) for r in (record.load('traces') or []) if r['verdict'] not in ('PASS', 'FAIL', 'PASS-ASSUMING')))")
echo "re-judging $(echo $TR | tr ',' '\n' | wc -l) trace rows"
"$PY" -u -m tvj.judge.traces_run --rows "$TR" > results/tr_rejudge.log 2>&1

# Publish by merging this run's scratch into the committed records, then archive the
# scratch.  This was `cp data/kb_live.jsonl results/kernelbook.jsonl`, which replaced
# the whole record with whatever the scratch held -- including an old run's, on any
# machine that had one -- and the LLM rows never went through a scratch at all.
"$PY" -m tvj.judge.record publish || exit 1
"$PY" -m tvj.judge.report both > results/report.txt 2>&1
echo POSTRUN-DONE
