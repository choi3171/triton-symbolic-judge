"""Where the repository is, independent of which package a module lives in.

`bridge/`, `data/` and `results/` sit at the repository root, and modules used to
find them relative to their own file.  That broke the moment the modules moved
into packages, so it is computed once here: two levels up from this file."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def at(*parts):
    return os.path.join(ROOT, *parts)
