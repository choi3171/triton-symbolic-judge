#!/bin/sh
# after the 40-400 run: re-judge every non-PASS row with the current front-end, then report
cd /home/naana/triton_function_correctness/tv
ROWS=$(python3 -c "
import json; r=json.load(open('data/kb_40_360.json')); print(','.join(str(x['i']) for x in r if x['verdict'] not in ('PASS','TIMEOUT')))")
echo "re-judging $(echo $ROWS | tr ',' '\n' | wc -l) rows"
python3 -u kernelbook_run.py --rows $ROWS > data/kb_rejudge.log 2>&1
python3 kb_report.py > data/kb_report.txt 2>&1
echo POSTRUN-DONE
