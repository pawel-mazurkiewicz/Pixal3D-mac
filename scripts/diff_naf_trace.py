#!/usr/bin/env python3
"""Diff PIXAL3D_NAF_TRACE fixture directories.

The NAF trace payloads are intentionally compact: most files contain scalar
stats plus deterministic flat samples, and selected files may also include a
full tensor.  This comparer uses full tensors when present in both dirs,
otherwise it compares the deterministic samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def _label(path: Path) -> str | None:
    stem = path.name.removesuffix(".pt")
    marker = "_naf_"
    if marker not in stem:
        return None
    return stem.split(marker, 1)[1]


def _collect(root: Path, prefix: str | None) -> dict[str, Path]:
    paths = {}
    pattern = f"{prefix}_naf_*.pt" if prefix else "*_naf_*.pt"
    for path in root.glob(pattern):
        label = _label(path)
        if label is not None:
            paths[label] = path
    return paths


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _values(payload: dict) -> tuple[str, torch.Tensor, torch.Tensor | None]:
    if "full" in payload:
        return "full", payload["full"].reshape(-1).float(), None
    samples = payload.get("samples") or {}
    values = samples.get("values")
    indices = samples.get("indices")
    if values is None:
        raise ValueError("payload has neither full tensor nor samples.values")
    return "samples", values.reshape(-1).float(), indices


def _diff_one(left_path: Path, right_path: Path) -> dict:
    left = _load(left_path)
    right = _load(right_path)
    mode_l, values_l, indices_l = _values(left)
    mode_r, values_r, indices_r = _values(right)
    n = min(values_l.numel(), values_r.numel())
    values_l = values_l[:n]
    values_r = values_r[:n]
    diff = (values_l - values_r).abs()
    max_abs = float(diff.max().item()) if n else 0.0
    mean_abs = float(diff.mean().item()) if n else 0.0
    rms = float(torch.sqrt((diff * diff).mean()).item()) if n else 0.0
    denom = max(float(values_l.abs().max().item()) if n else 0.0, 1e-12)
    index_match = None
    if indices_l is not None and indices_r is not None:
        m = min(indices_l.numel(), indices_r.numel())
        index_match = bool(torch.equal(indices_l[:m], indices_r[:m]))
    return {
        "label": _label(left_path),
        "left": str(left_path),
        "right": str(right_path),
        "shape_left": tuple(left.get("shape", ())),
        "shape_right": tuple(right.get("shape", ())),
        "mode_left": mode_l,
        "mode_right": mode_r,
        "n": int(n),
        "index_match": index_match,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "rms": rms,
        "rel_to_left_max": max_abs / denom,
    }


def _status(max_abs: float, green: float, yellow: float) -> str:
    if max_abs <= green:
        return "GREEN"
    if max_abs <= yellow:
        return "YELLOW"
    return "RED"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path, help="Reference fixture directory")
    parser.add_argument("right", type=Path, help="Candidate fixture directory")
    parser.add_argument(
        "--prefix",
        default="01b_image_cond_shape_512",
        help="Stage prefix to compare, or empty string for all NAF files",
    )
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--green", type=float, default=1e-4)
    parser.add_argument("--yellow", type=float, default=1e-3)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    prefix = args.prefix or None
    left = _collect(args.left, prefix)
    right = _collect(args.right, prefix)
    common = sorted(set(left) & set(right))
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))

    rows = []
    for label in common:
        try:
            rows.append(_diff_one(left[label], right[label]))
        except Exception as exc:
            rows.append({"label": label, "error": repr(exc)})

    ranked = sorted(
        [r for r in rows if "max_abs" in r],
        key=lambda r: r["max_abs"],
        reverse=True,
    )

    print(f"left:  {args.left}")
    print(f"right: {args.right}")
    print(
        f"common={len(common)} missing_left={len(missing_left)} "
        f"missing_right={len(missing_right)}"
    )
    if missing_left:
        print("missing from left:", ", ".join(missing_left[:10]))
    if missing_right:
        print("missing from right:", ", ".join(missing_right[:10]))

    print("\nTop drift:")
    print(f"{'status':<7} {'max_abs':>11} {'mean_abs':>11} {'rms':>11} {'n':>8}  label")
    for row in ranked[: args.top]:
        print(
            f"{_status(row['max_abs'], args.green, args.yellow):<7} "
            f"{row['max_abs']:11.4e} {row['mean_abs']:11.4e} "
            f"{row['rms']:11.4e} {row['n']:8d}  {row['label']}"
        )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "left": str(args.left),
            "right": str(args.right),
            "prefix": prefix,
            "missing_left": missing_left,
            "missing_right": missing_right,
            "rows": rows,
        }, indent=2))
        print(f"\njson -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
