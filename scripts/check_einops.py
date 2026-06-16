"""Linter to check for einops usecases."""

import pathlib
import re
import sys

combined = re.compile(
    r"\.reshape\(|\.view\(|\.permute\(|\.unsqueeze\(|\.squeeze\(|torch\.einsum\("
    r"|\.norm\(|torch\.linalg\.vector_norm\(|torch\.linalg\.norm\(",
)
violations = []
for f in sys.argv[1:]:
    for i, line in enumerate(pathlib.Path(f).read_text().splitlines(), 1):
        if line.strip().startswith("#"):
            continue
        if combined.search(line):
            violations.append(f"{f}:{i}: {line.strip()}")
if violations:
    print(
        "Replace raw tensor ops with einops equivalents (rearrange/einsum/reduce):",
    )
    for v in violations:
        print(v)
    sys.exit(1)
