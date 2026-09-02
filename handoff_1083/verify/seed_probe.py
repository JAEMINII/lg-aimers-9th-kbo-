# -*- coding: utf-8 -*-
"""submit_41/model/trees.npz 의 시드 집합과 학습 장치를 실측으로 확정한다.

방법
    export 는 [all, regular, futures] 순, 각 그룹당 n_per 개를 SEEDS 순서로 담는다.
    따라서 모델 0 = group "all", seed SEEDS[0] 이고 트리 0..399 가 그것이다.
    같은 자료로 CPU 에서 seed 1 을 학습해 그 400그루와 대조한다.
    맞으면 시드가 1부터이고 CPU 빌드다. 틀리면 GPU 빌드이거나 시드가 다르다.
"""
import json
import os
import sys
import tempfile
import time

import numpy as np

def _find_root(start):
    """`open (1)/data` 를 가진 디렉토리를 위로 올라가며 찾는다.
    이 파일이 저장소 안에 있든 handoff 묶음 안에 있든 동작해야 한다."""
    d = os.environ.get("AIMERS_ROOT")
    if d and os.path.isdir(os.path.join(d, "open (1)", "data")):
        return d
    d = start
    for _ in range(6):
        if os.path.isdir(os.path.join(d, "open (1)", "data")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("open (1)/data 를 못 찾았다. AIMERS_ROOT 를 지정하라.")


ROOT = _find_root(os.path.dirname(os.path.abspath(__file__)))
HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (os.path.join(HERE, "..", "build"), os.path.join(ROOT, "nn_experiments"),
           os.path.join(ROOT, "handoff_1083", "build")):
    if os.path.isfile(os.path.join(_c, "train_c12_submit.py")):
        sys.path.insert(0, os.path.abspath(_c))
        break
else:
    raise SystemExit("train_c12_submit.py 를 못 찾았다")
os.environ.setdefault("AIMERS_ROOT", ROOT)
import train_c12_submit as T                                    # noqa: E402
import features44 as F                                          # noqa: E402
from catboost import CatBoostClassifier                         # noqa: E402

z = np.load(os.path.join(ROOT, "submit_41", "model", "trees.npz"),
            allow_pickle=False)
feats = [str(c) for c in z["features"].tolist()]
n_per, n_models = int(z["n_per_group"]), int(z["n_models"])
per_model = z["cb_leaf"].shape[0] // n_models
print(f"  실린 것  모델 {n_models}  그룹당 {n_per}  모델당 트리 {per_model}  "
      f"피처 {len(feats)}")

t0 = time.time()
built = F.build(os.path.join(ROOT, "open (1)", "data"), VS=2025, return_frame=True)
frame = built["frame"]
base = list(built["F44"])
frame, _ = T.add_c12(frame, return_tables=True)
frame, _ = T.add_cmh(frame, return_tables=True)
want = base + ["pc_c12_rate", "pc_c12_dev", "pc_c12_n",
               "pc_cmh_rate", "pc_cmh_dev", "pc_cmh_n"]
assert want == feats, f"피처 순서 불일치\n  코드 {want}\n  실린것 {feats}"
print(f"  피처 순서 일치 ({len(want)}열)   자료 준비 {time.time()-t0:.0f}초")

X = frame[want].to_numpy(np.float32)
y = frame[T.TARGET].to_numpy(np.float32)
w = T.DECAY ** (frame["season"].to_numpy(np.float64) - 2019.0)
print(f"  행 {len(y):,}  표적평균 {y.mean():.6f}")


def trees_of(model):
    fd, tmp = tempfile.mkstemp(suffix=".json"); os.close(fd)
    model.save_model(tmp, format="json")
    obj = json.load(open(tmp, encoding="utf-8")); os.remove(tmp)
    fmap = {int(x["feature_index"]): x
            for x in obj["features_info"]["float_features"]}
    fi, th = [], []
    for tree in obj["oblivious_trees"]:
        a, b = [], []
        for sp in tree.get("splits") or []:
            ff = fmap[int(sp["float_feature_index"])]
            a.append(int(ff["flat_feature_index"]))
            b.append(float(sp["border"]) if "border" in sp
                     else float(ff["borders"][int(sp["border_idx"])]))
        fi.append(a + [0] * (4 - len(a)))
        th.append(b + [np.inf] * (4 - len(b)))
    return np.asarray(fi, np.int32), np.asarray(th, np.float32)


hp = dict(iterations=400, learning_rate=0.05, depth=4, l2_leaf_reg=1.0,
          verbose=0, allow_writing_files=False)
for seed in (1, 42):
    t0 = time.time()
    m = CatBoostClassifier(random_seed=seed, **hp)
    m.fit(X, y, sample_weight=w)                 # group "all"
    fi, th = trees_of(m)
    ok_f = bool((fi == z["cb_feat"][:per_model]).all())
    ok_t = bool(np.allclose(np.nan_to_num(th, posinf=1e30),
                            np.nan_to_num(z["cb_thr"][:per_model], posinf=1e30)))
    same = int((fi == z["cb_feat"][:per_model]).sum())
    print(f"  seed={seed:>3d} CPU  {time.time()-t0:5.0f}초   "
          f"분할피처 일치 {same}/{fi.size}  ({same/fi.size*100:5.1f}%)   "
          f"완전일치 {'예' if ok_f and ok_t else '아니오'}")
