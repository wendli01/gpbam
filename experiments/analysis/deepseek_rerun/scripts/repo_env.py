"""Find the .env, including from inside a linked git worktree.

.env is untracked, so it exists only in the primary checkout. Inside a worktree
``ROOT/.env`` does not exist, and the failure surfaces as a KeyError on the API
key thirty lines later -- which is what once killed the judging stage of a
completed generation run, silently, after the expensive part had finished.

judge_0731.py carried this logic first; it lives here so the seat runner gets it
too rather than the tree growing a second copy that can drift.
"""
import subprocess
from pathlib import Path


def dotenv_path(root):
    """``root/.env``, or the primary checkout's if this is a worktree."""
    root = Path(root)
    here = root / '.env'
    if here.exists():
        return here
    common = subprocess.run(
        ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'],
        cwd=root, capture_output=True, text=True).stdout.strip()
    main = Path(common).parent / '.env' if common else None
    if main and main.exists():
        print(f'note: .env from the primary checkout, {main}', flush=True)
        return main
    raise SystemExit(f'no .env at {here} and none in the primary checkout')
