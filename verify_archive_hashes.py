#!/usr/bin/env python3
"""Verify SHA256SUMS.txt without changing any submitted file."""
from pathlib import Path
import hashlib, sys
ROOT=Path(__file__).resolve().parents[1]
rows=(ROOT/'SHA256SUMS.txt').read_text().splitlines()
failures=[]
for row in rows:
    expected,rel=row.split('  ',1)
    path=ROOT/rel
    if not path.is_file(): failures.append('MISSING '+rel);continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected:failures.append('MISMATCH '+rel)
print('Verified',len(rows)-len(failures),'/',len(rows),'distributed files')
for e in failures:print(e,file=sys.stderr)
sys.exit(1 if failures else 0)
