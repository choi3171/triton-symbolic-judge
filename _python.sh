# Which interpreter the shell scripts should use.  Sourced, not run.
#
# `python3` is not a given.  A conda notebook environment can expose only
# `python`, Windows installs only the `py` launcher, and a shell that re-reads
# its profile can lose the PATH the notebook process had -- so `!python` works
# in a cell and `python3` does not work in a `%%bash` block two lines later.
# Hard-coding the name meant the first sign of any of that was ./setup.sh dying
# on its own preflight, which is the one thing the preflight exists to prevent.
#
# Set PYTHON=/path/to/python to override.
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for _c in python3 python py; do
    command -v "$_c" >/dev/null 2>&1 && { PY=$_c; break; }
  done
fi
[ -n "$PY" ] || {
  echo "no python3, python or py on PATH.  Set PYTHON=/path/to/python and re-run;" >&2
  echo "  in a conda notebook that is usually /opt/conda/bin/python." >&2
  exit 1; }
# A `python` that is Python 2 is worse than none: it gets past the check above and
# fails somewhere inside a script instead.
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info[0] == 3 else 1)' 2>/dev/null || {
  echo "$PY is not Python 3.  Set PYTHON=/path/to/python3 and re-run." >&2; exit 1; }
