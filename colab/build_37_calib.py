# -*- coding: utf-8 -*-
"""Build the last candidate by recalibrating the submit_35 predictor.

Only the global logit shift changes.  All model bytes and runtime code are
copied from submit_35, so this candidate is a low-risk calibration A/B.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "submit_35"
OUT = ROOT / "submit_37"
ZIP = ROOT / "submit_jaemin_37.zip"
OLD = "CALIB_LOGIT_SHIFT = -0.007"
NEW = "CALIB_LOGIT_SHIFT = -0.04"


def main() -> None:
    if not SRC.is_dir():
        raise SystemExit(f"missing source package: {SRC}")
    if OUT.exists() or ZIP.exists():
        raise SystemExit(f"refusing to overwrite existing output: {OUT} / {ZIP}")

    (OUT / "model").mkdir(parents=True)
    for name in ("preprocess.py", "features44.py", "requirements.txt"):
        shutil.copy2(SRC / name, OUT / name)

    script = (SRC / "script.py").read_text(encoding="utf-8")
    if script.count(OLD) != 1:
        raise SystemExit(f"expected exactly one calibration constant, got {script.count(OLD)}")
    script = script.replace(OLD, NEW, 1)
    (OUT / "script.py").write_text(script, encoding="utf-8", newline="\n")

    for src in sorted((SRC / "model").iterdir()):
        if src.is_file():
            shutil.copy2(src, OUT / "model" / src.name)

    files = ["script.py", "preprocess.py", "features44.py", "requirements.txt"]
    files += [f"model/{p.name}" for p in sorted((OUT / "model").iterdir()) if p.is_file()]
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for rel in files:
            z.write(OUT / rel, rel)
    with zipfile.ZipFile(ZIP) as z:
        if z.testzip() is not None:
            raise SystemExit("zip integrity check failed")
        if set(z.namelist()) != set(files):
            raise SystemExit("zip file list differs from package file list")

    print(f"built {OUT}")
    print(f"built {ZIP} ({ZIP.stat().st_size / 1e6:.2f} MB)")
    print(f"script sha256={hashlib.sha256((OUT / 'script.py').read_bytes()).hexdigest()}")
    print(f"zip entries={len(files)}")


if __name__ == "__main__":
    main()
