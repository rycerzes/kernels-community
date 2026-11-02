"""Vendor ffpa-attn and the quack modules it uses into `torch-ext/ffpa_attn`, so the kernel is
self-contained (`quack-kernels` is not an allowed Hub-kernel Python dependency).

Mirrors the flash-attn4 sync steps in AGENTS.md:
- copy upstream `src/ffpa_attn` except what `__init__` never reaches (`cli/`, `bench.py`, `ray/`,
  `autotune.py`);
- copy the quack modules ffpa-attn imports, transitively, and quack's LICENSE. `quack/__init__.py`
  only imports `.dsl`, whose import installs the CuTe tensor-indexing patch `copy_utils` relies on;
- make every `ffpa_attn` / `quack` import relative;
- register ffpa-attn's torch ops under the kernel's unique namespace (`add_op_namespace_prefix` and
  `ops` from `._ops`).

The default revisions are the source of the ffpa-attn 0.2.4 PyPI release (its Python files are
identical to upstream commit 918ae3c) and the quack version it pins (`quack-kernels==0.6.4`).

Usage: python vendor.py [--ffpa-src DIR] [--ffpa-rev REV] [--quack-src DIR] [--quack-rev REV]
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FFPA_REV, QUACK_REV = "918ae3c", "v0.6.4"


def checkout(url: str, path: Path, rev: str) -> Path:
    if not path.is_dir():
        subprocess.run(["git", "clone", "-q", url, str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", "-q", rev], check=True)
    return path


ap = argparse.ArgumentParser()
ap.add_argument("--ffpa-src", type=Path, default=HERE / "ffpa-attn-src")
ap.add_argument("--ffpa-rev", default=FFPA_REV)
ap.add_argument("--quack-src", type=Path, default=HERE / "quack-src")
ap.add_argument("--quack-rev", default=QUACK_REV)
args = ap.parse_args()
ffpa_repo = checkout("https://github.com/xlite-dev/ffpa-attn.git", args.ffpa_src, args.ffpa_rev)
quack_repo = checkout("https://github.com/Dao-AILab/quack.git", args.quack_src, args.quack_rev)
dest = HERE
SRC = ffpa_repo / "src" / "ffpa_attn"
PKG = dest / "torch-ext" / "ffpa_attn"
SKIP = {"cli", "ray", "bench.py", "autotune.py", ".gitignore"}
QUACK = ["activation.py", "compile_utils.py", "copy_utils.py", "cute_dsl_utils.py", "layout_utils.py", "rounding.py",
         "sm90_utils.py", "utils.py", "dsl/__init__.py", "dsl/cute_tensor.py", "dsl/cute_tensor_indexing.py",
         "dsl/mixed_constexpr_if.py", "dsl/torch_library_op.py"]

if PKG.exists():
    shutil.rmtree(PKG)
PKG.mkdir(parents=True)

for p in sorted(SRC.rglob("*")):
    rel = p.relative_to(SRC)
    if rel.parts[0] in SKIP or "__pycache__" in rel.parts or p.is_dir():
        continue
    (PKG / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, PKG / rel)
shutil.copy2(ffpa_repo / "LICENSE", PKG / "LICENSE")
# setuptools-scm writes _version.py at build time; this is the PyPI 0.2.4 content's value. Update when syncing.
(PKG / "_version.py").write_text('# Version of the vendored upstream release. Update when syncing.\n'
                                 'version = __version__ = "0.2.4"\n')

QDIR = PKG / "quack"
for rel in QUACK:
    (QDIR / rel).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(quack_repo / "quack" / rel, QDIR / rel)
shutil.copy2(quack_repo / "LICENSE", QDIR / "LICENSE")
(QDIR / "__init__.py").write_text(
    '"""Vendored subset of quack 0.6.4 (https://github.com/Dao-AILab/quack), the modules ffpa-attn uses.\n\n'
    'Importing `.dsl` installs quack\'s CuTe tensor indexing patch, exactly as upstream\'s `quack/__init__.py`\n'
    'does; `copy_utils` indexes tensors with `...`, which CuTe does not support without it.\n"""\n\n'
    'from . import dsl  # noqa: F401\n')


def quack_base(path: Path) -> str:
    """Relative module path of the vendored quack package as seen from `path`."""
    if path.is_relative_to(QDIR):
        return "." * (len(path.parent.relative_to(QDIR).parts) + 1)        # "." from quack/x.py
    return "." * (len(path.parent.relative_to(PKG).parts) + 1) + "quack"   # ".quack" from ffpa_attn/x.py


def join(base: str, sub: str) -> str:
    return base + sub if base.endswith(".") else f"{base}.{sub}"


def relativize(path: Path) -> None:
    qbase = quack_base(path)
    out = []
    for line in path.read_text().splitlines(keepends=True):
        m1 = re.match(r"^(\s*)from quack(?:\.([\w.]+))? import (.*)$", line, re.S)
        m2 = re.match(r"^(\s*)import quack\.([\w.]+)(\s+as\s+\w+)?(.*)$", line, re.S)
        if m1:
            ind, sub, rest = m1.groups()
            line = f"{ind}from {join(qbase, sub) if sub else qbase} import {rest}"
        elif m2:
            ind, dotted, alias, tail = m2.groups()
            parent, _, leaf = dotted.rpartition(".")
            line = f"{ind}from {join(qbase, parent) if parent else qbase} import {leaf}{alias or ''}{tail}"
        elif re.match(r"^\s*(from|import) (ffpa_attn|quack)\b", line):
            raise SystemExit(f"unhandled absolute import in {path}: {line.strip()}")
        out.append(line)
    path.write_text("".join(out))


for p in PKG.rglob("*.py"):
    relativize(p)


def edit(rel: str, pairs: list[tuple[str, str]], regex: bool = False) -> None:
    p = PKG / rel
    s = p.read_text()
    for a, b in pairs:
        if regex:
            s, n = re.subn(a, b, s)
        else:
            n = s.count(a)
            s = s.replace(a, b)
        if n == 0:
            raise SystemExit(f"{rel}: pattern not found: {a!r}")
    p.write_text(s)


def add_import(rel: str, line: str) -> None:
    """Insert after the last top-level `import torch` (the op modules all import it)."""
    p = PKG / rel
    s = p.read_text()
    i = s.index("\nimport torch\n") + len("\nimport torch\n")
    p.write_text(s[:i] + line + "\n" + s[i:])


# CuTe ops: string names and torch.ops.ffpa_attn lookups.
edit("cute/__init__.py", [(r'"ffpa_attn::(\w+)"', r'add_op_namespace_prefix("\1")'),
                          (r"torch\.ops\.ffpa_attn\.", "_ffpa_ops.")], regex=True)
add_import("cute/__init__.py", "from .._ops import add_op_namespace_prefix, ops as _ffpa_ops")
# Triton and (optional, not shipped compiled) CUDA backends build names from _OP_NAMESPACE.
for rel in ("triton/__init__.py", "cuda/__init__.py"):
    edit(rel, [('_OP_NAMESPACE = "ffpa_attn"', '_OP_NAMESPACE = add_op_namespace_prefix("").removesuffix("::")')])
    add_import(rel, "from .._ops import add_op_namespace_prefix")
for rel in ("triton/_ffpa_fwd.py", "triton/_ffpa_bwd.py", "cuda/_ffpa_fwd.py"):
    edit(rel, [(r"torch\.ops\.ffpa_attn\.", "_ffpa_ops.")], regex=True)
    add_import(rel, "from .._ops import ops as _ffpa_ops")

# Nothing may still name the fixed namespace outside comments/docstrings.
left = []
for p in PKG.rglob("*.py"):
    for i, line in enumerate(p.read_text().splitlines(), 1):
        code = re.sub(r"``[^`]*``", "", line.split("#", 1)[0])  # prose in docstrings
        if re.search(r'torch\.ops\.ffpa_attn|"ffpa_attn::|\'ffpa_attn::', code):
            left.append(f"{p.relative_to(PKG)}:{i}: {line.strip()}")
if left:
    raise SystemExit("fixed op namespace left:\n" + "\n".join(left))
print("vendored", sum(1 for _ in PKG.rglob("*.py")), "python files,",
      sum(1 for _ in PKG.rglob("*.json")), "json files")
