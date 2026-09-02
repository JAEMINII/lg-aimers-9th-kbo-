"""Evaluate DIN against the closest archived submit_41-style fold baselines.

The 2022/2023 CatBoost archive is the exact 50-feature route trained anew
with the submit_41 hyperparameters and routing.  The TabM archive is the
deployment-style (friend preprocess, c4, route, seed 42) OOF prediction.
It is not the literal production ensemble because the 44-feature CatBoost
incumbent's historical OOF file is unavailable, so this is a strict gate,
not a submission-score estimate.
"""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import features44 as F  # noqa: E402

DATA = os.environ.get("AIMERS_DATA", os.path.join(os.path.dirname(ROOT), "open (1)", "data"))
DL = os.environ.get("AIMERS_DL", os.path.join(ROOT, "_dl"))


def main():
    for vs in (2022, 2023, 2024):
        d = F.build(DATA, VS=vs)
        gate = d["season"] == vs
        y = d["y"][gate].astype(np.float64)
        tab = np.load(os.path.join(DL, f"h2h_{vs}_friend_s42.npy")).astype(np.float64)
        if vs == 2024:
            cb = (.70 * np.load(os.path.join(DL, "pcgpu2024_c12_cmh_10.npy"))
                  + .30 * np.load(os.path.join(DL, "pcgpu2024_base44_10.npy")))
        else:
            cb = np.load(os.path.join(DL, f"cb50fixed_{vs}.npy")).astype(np.float64)
        din = np.load(os.path.join(DL, f"cz{vs}_DIN.npy")).mean(0).astype(np.float64)
        if not (len(y) == len(tab) == len(cb) == len(din)):
            raise ValueError((vs, len(y), len(tab), len(cb), len(din)))
        base = .70 * tab + .30 * cb
        base_best, shift = F.best_shift(base, y)
        base_raw = F.bss(base, y)
        print(f"VS{vs}: base={base_best:.2f} raw={base_raw:.2f} shift={shift:+.4f} "
              f"DIN={F.best_shift(din, y)[0]:.2f} corr={np.corrcoef(base, din)[0, 1]:.4f}")
        for w in (.05, .10, .15, .20, .25):
            p = (1 - w) * base + w * din
            best, _ = F.best_shift(p, y)
            raw = F.bss(p, y)
            common = F.bss(F.shift(p, shift), y)
            base_common = F.bss(F.shift(base, shift), y)
            print(f"  DIN w={w:.2f}: best {best-base_best:+.2f}  raw {raw-base_raw:+.2f}  "
                  f"shared-shift {common-base_common:+.2f}")


if __name__ == "__main__":
    main()
