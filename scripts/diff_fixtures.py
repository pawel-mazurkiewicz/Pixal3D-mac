#!/usr/bin/env python3
"""Diff two Pixal3D fixture captures (CUDA vs MPS) stage by stage.

Reads `.pt` fixtures dumped by inference.py's `_dump_fixture()` helper
(when `PIXAL3D_DUMP_FIXTURES` is set during a run), walks the nested
tensor/dict/list structures, and prints per-tensor max-abs-diff stats.

The first fixture flagged RED is the divergence boundary.  See
`DIVERGENCE_HUNT.md` for the workflow that uses this.

Usage:
    python diff_fixtures.py CUDA_DIR MPS_DIR
    python diff_fixtures.py CUDA_DIR MPS_DIR --threshold-red 1e-2 --threshold-yellow 1e-4
    python diff_fixtures.py CUDA_DIR MPS_DIR --json results.json
    python diff_fixtures.py CUDA_DIR MPS_DIR --stop-on-red

Exit codes:
    0 — all GREEN/YELLOW (no real divergence found)
    1 — at least one RED diff (real divergence)
    2 — usage / IO error
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import torch


# --- ANSI colours ---------------------------------------------------------

class C:
    """Minimal ANSI helpers — degrade to plain text when stdout isn't a tty."""
    _enabled = sys.stdout.isatty()
    RESET = "\033[0m" if _enabled else ""
    BOLD  = "\033[1m" if _enabled else ""
    DIM   = "\033[2m" if _enabled else ""
    RED   = "\033[31m" if _enabled else ""
    GREEN = "\033[32m" if _enabled else ""
    YEL   = "\033[33m" if _enabled else ""
    CYAN  = "\033[36m" if _enabled else ""


# --- diff data model ------------------------------------------------------

@dataclass
class Diff:
    path: str
    kind: str          # 'numeric' | 'shape_mismatch' | 'missing' | 'type_mismatch' | 'equal'
    max_abs: float | None = None
    mean_abs: float | None = None
    detail: str = ""

    def severity(self, red: float, yellow: float) -> str:
        if self.kind in ("shape_mismatch", "missing", "type_mismatch"):
            return "RED"
        if self.kind == "equal":
            return "GREEN"
        if self.max_abs is None:
            return "GREEN"
        if self.max_abs >= red:
            return "RED"
        if self.max_abs >= yellow:
            return "YELLOW"
        return "GREEN"

    def format_value(self) -> str:
        if self.kind == "equal":
            return "equal"
        if self.kind == "shape_mismatch":
            return f"SHAPE {self.detail}"
        if self.kind == "missing":
            return f"MISSING on {self.detail}"
        if self.kind == "type_mismatch":
            return f"TYPE {self.detail}"
        return f"max={self.max_abs:.3e}  mean={self.mean_abs:.3e}"


@dataclass
class FixtureResult:
    name: str
    diffs: list[Diff] = field(default_factory=list)
    error: str | None = None

    def worst(self, red: float, yellow: float) -> str:
        if self.error:
            return "RED"
        levels = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        worst = "GREEN"
        for d in self.diffs:
            sev = d.severity(red, yellow)
            if levels[sev] > levels[worst]:
                worst = sev
        return worst


# --- recursive walker -----------------------------------------------------

def _is_tensor_like(x: Any) -> bool:
    return torch.is_tensor(x)


def _is_scalar_equal(a: Any, b: Any) -> bool:
    """Loose equality for non-tensor leaf values (ints, strs, bools, etc.)."""
    if type(a) is not type(b):
        return False
    try:
        return a == b
    except Exception:
        return False


def walk(a: Any, b: Any, path: str = "") -> list[Diff]:
    """Recursively compare two fixture payloads, returning flat list of Diff."""
    out: list[Diff] = []

    if a is None and b is None:
        return out
    if a is None:
        out.append(Diff(path, "missing", detail="cuda side"))
        return out
    if b is None:
        out.append(Diff(path, "missing", detail="mps side"))
        return out

    if _is_tensor_like(a) and _is_tensor_like(b):
        if a.shape != b.shape:
            out.append(Diff(path, "shape_mismatch",
                            detail=f"{tuple(a.shape)} vs {tuple(b.shape)}"))
            return out
        if a.numel() == 0:
            out.append(Diff(path, "equal"))
            return out
        # Cast to float for diff; works for int and float tensors alike.
        # bool tensors need an explicit cast.
        af = a.detach().float() if a.dtype != torch.bool else a.detach().int().float()
        bf = b.detach().float() if b.dtype != torch.bool else b.detach().int().float()
        d = (af - bf).abs()
        out.append(Diff(
            path, "numeric",
            max_abs=float(d.max().item()),
            mean_abs=float(d.mean().item()),
        ))
        return out

    if _is_tensor_like(a) ^ _is_tensor_like(b):
        out.append(Diff(path, "type_mismatch",
                        detail=f"{type(a).__name__} vs {type(b).__name__}"))
        return out

    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out.extend(walk(a.get(k), b.get(k), f"{path}.{k}" if path else f".{k}"))
        return out

    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(Diff(path, "shape_mismatch",
                            detail=f"len {len(a)} vs {len(b)}"))
            n = min(len(a), len(b))
        else:
            n = len(a)
        for i in range(n):
            out.extend(walk(a[i], b[i], f"{path}[{i}]"))
        return out

    # Scalar / leaf: exact equality check, no severity.
    if _is_scalar_equal(a, b):
        out.append(Diff(path, "equal"))
    else:
        out.append(Diff(path, "type_mismatch",
                        detail=f"{repr(a)[:40]} vs {repr(b)[:40]}"))
    return out


# --- metadata sanity ------------------------------------------------------

def load_metadata(d: str) -> dict | None:
    p = os.path.join(d, "00_metadata.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        print(f"  WARN: failed to parse {p}: {e}", file=sys.stderr)
        return None


def print_sanity(cuda_dir: str, mps_dir: str) -> tuple[dict | None, dict | None]:
    cm = load_metadata(cuda_dir)
    mm = load_metadata(mps_dir)
    print(f"{C.BOLD}== sanity =={C.RESET}")
    for label, d, m in (("cuda", cuda_dir, cm), ("mps ", mps_dir, mm)):
        if m is None:
            print(f"  {label}  {C.RED}NO METADATA at {d}/00_metadata.json{C.RESET}")
            continue
        seed = m.get("seed")
        sha = (m.get("image_sha256") or "")[:12]
        torchv = m.get("torch_version")
        dev = m.get("device_kind")
        print(f"  {label}  device={dev:5s}  seed={seed}  sha={sha}  torch={torchv}")
    print()
    # Warn loudly on mismatches that would invalidate the comparison.
    if cm and mm:
        problems = []
        if cm.get("seed") != mm.get("seed"):
            problems.append(f"seed mismatch: {cm.get('seed')} vs {mm.get('seed')}")
        if cm.get("image_sha256") != mm.get("image_sha256"):
            problems.append(
                f"image SHA mismatch: {cm.get('image_sha256','?')[:12]} vs "
                f"{mm.get('image_sha256','?')[:12]}"
            )
        for p in problems:
            print(f"  {C.RED}{C.BOLD}⚠ {p}{C.RESET}")
        if problems:
            print(f"  {C.YEL}Comparison may be meaningless. Continuing anyway.{C.RESET}\n")
    return cm, mm


# --- per-fixture comparison ----------------------------------------------

def compare_fixture(cuda_path: str, mps_path: str) -> FixtureResult:
    name = os.path.basename(cuda_path)
    res = FixtureResult(name=name)
    if not os.path.exists(mps_path):
        res.diffs.append(Diff(name, "missing", detail="mps side"))
        return res
    try:
        a = torch.load(cuda_path, weights_only=False)
        b = torch.load(mps_path,  weights_only=False)
    except Exception as e:
        res.error = f"load failed: {e}"
        return res
    res.diffs = walk(a, b, "")
    if not res.diffs:
        # Empty payload (e.g. None on both sides) — treat as equal.
        res.diffs.append(Diff("", "equal"))
    return res


# --- printer --------------------------------------------------------------

def print_fixture(res: FixtureResult, red: float, yellow: float,
                  width: int = 70) -> str:
    """Print one fixture's diffs, return worst severity for caller to track."""
    worst = res.worst(red, yellow)
    colour = {"GREEN": C.GREEN, "YELLOW": C.YEL, "RED": C.RED}[worst]
    print(f"{C.BOLD}== {res.name} =={C.RESET}  "
          f"({colour}worst={worst}{C.RESET})")
    if res.error:
        print(f"  {C.RED}ERROR: {res.error}{C.RESET}")
        return worst
    for d in res.diffs:
        sev = d.severity(red, yellow)
        sev_colour = {"GREEN": C.GREEN, "YELLOW": C.YEL, "RED": C.RED}[sev]
        path = d.path or "<root>"
        # Truncate long paths so the severity column lines up.
        if len(path) > width:
            path = "…" + path[-(width - 1):]
        print(f"  {path:<{width}}  {d.format_value():<35}  {sev_colour}{sev}{C.RESET}")
    print()
    return worst


# --- main -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cuda_dir", help="directory of CUDA-side fixtures")
    ap.add_argument("mps_dir",  help="directory of MPS-side fixtures")
    ap.add_argument("--threshold-red",    type=float, default=1e-2,
                    help="max-abs-diff at or above this -> RED (default: 1e-2)")
    ap.add_argument("--threshold-yellow", type=float, default=1e-4,
                    help="max-abs-diff at or above this (but below red) -> YELLOW "
                         "(default: 1e-4)")
    ap.add_argument("--stop-on-red", action="store_true",
                    help="stop printing after the first RED fixture")
    ap.add_argument("--json", type=str, default=None,
                    help="also write structured results to this JSON file")
    args = ap.parse_args()

    if not os.path.isdir(args.cuda_dir):
        print(f"Not a directory: {args.cuda_dir}", file=sys.stderr)
        return 2
    if not os.path.isdir(args.mps_dir):
        print(f"Not a directory: {args.mps_dir}", file=sys.stderr)
        return 2

    cm, mm = print_sanity(args.cuda_dir, args.mps_dir)

    cuda_files = sorted(glob.glob(os.path.join(args.cuda_dir, "*.pt")))
    if not cuda_files:
        print(f"No .pt files in {args.cuda_dir}", file=sys.stderr)
        return 2

    results: list[FixtureResult] = []
    overall = "GREEN"
    levels = {"GREEN": 0, "YELLOW": 1, "RED": 2}
    first_red: str | None = None

    for cuda_path in cuda_files:
        name = os.path.basename(cuda_path)
        mps_path = os.path.join(args.mps_dir, name)
        res = compare_fixture(cuda_path, mps_path)
        results.append(res)
        worst = print_fixture(res, args.threshold_red, args.threshold_yellow)
        if levels[worst] > levels[overall]:
            overall = worst
        if worst == "RED" and first_red is None:
            first_red = name
        if args.stop_on_red and worst == "RED":
            print(f"{C.YEL}--stop-on-red: skipping remaining fixtures.{C.RESET}\n")
            break

    # Also flag fixtures present only on the MPS side (we walk cuda_files).
    mps_only = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(args.mps_dir, "*.pt")))
                      - set(os.path.basename(p) for p in cuda_files))
    if mps_only:
        print(f"{C.YEL}{C.BOLD}== mps-only fixtures =={C.RESET}")
        for n in mps_only:
            print(f"  {n}")
        print()

    # --- summary ---
    summary_colour = {"GREEN": C.GREEN, "YELLOW": C.YEL, "RED": C.RED}[overall]
    print(f"{C.BOLD}== summary =={C.RESET}")
    print(f"  fixtures compared:    {len(results)}")
    print(f"  overall severity:     {summary_colour}{overall}{C.RESET}")
    if first_red:
        print(f"  first RED fixture:    {C.RED}{first_red}{C.RESET}  "
              f"{C.DIM}<- divergence boundary{C.RESET}")
    print(f"  thresholds:           red≥{args.threshold_red}, yellow≥{args.threshold_yellow}")

    # --- optional JSON dump ---
    if args.json:
        out = {
            "cuda_dir": os.path.abspath(args.cuda_dir),
            "mps_dir":  os.path.abspath(args.mps_dir),
            "metadata_cuda": cm,
            "metadata_mps": mm,
            "thresholds": {"red": args.threshold_red, "yellow": args.threshold_yellow},
            "overall": overall,
            "first_red": first_red,
            "fixtures": [
                {
                    "name": r.name,
                    "worst": r.worst(args.threshold_red, args.threshold_yellow),
                    "error": r.error,
                    "diffs": [
                        {
                            "path": d.path,
                            "kind": d.kind,
                            "max_abs": d.max_abs,
                            "mean_abs": d.mean_abs,
                            "detail": d.detail,
                            "severity": d.severity(args.threshold_red, args.threshold_yellow),
                        }
                        for d in r.diffs
                    ],
                }
                for r in results
            ],
        }
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  JSON results:         {args.json}")

    return 1 if overall == "RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
