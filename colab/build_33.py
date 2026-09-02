# -*- coding: utf-8 -*-
"""submit_33 = submit_30(LB 1069) 에서 두 가지를 바꾼다.

    ① 1군 TabM 을 weight_decay 0 으로 재학습 (3e-4 -> 0)
       정규화 스윕에서 유일하게 양수였던 팔이다 (+1.6, 2/3).
       판정선(+5 & 3/3)에는 못 미쳤다는 걸 적어둔다.

    ② 혼합에 MNCA 를 넣는다.  CB 0.20 / MNCA 0.20 / TabM 0.60
       현행은 CB 0.30 / TabM 0.70 이다.

MNCA 가 뭔가
    ModernNCA (Ye et al., ICLR 2025). 시험 행을 인코딩한 뒤 **학습 행 후보
    16,384개와의 거리**로 소프트맥스 가중을 만들어 그들의 라벨을 평균낸다.
        z = E(x)
        d = ||z - cand_z||          (C, ) 거리
        w = softmax(-d / tau)
        p = sum w_i * cand_y_i

    후보를 **미리 인코딩해서** 저장했으므로 추론에서 후보를 다시 통과시킬
    필요가 없다. cand_z (16384, 256) + cand_y (16384,) 만 있으면 된다.

    후보는 학습 행에서만 뽑았고 고정 배열이라 시험 행의 순서·개수와 무관하다.
    규칙 4 는 구조적으로 안전하고 감사로 확인한다.

관문 근거와 한계
    MNCA 는 후보 열 몇 개 중 **유일하게 두 조건을 동시에 만족**했다.
        상관 < 0.97   (0.98 위는 뭘 넣어도 0 이거나 음수)
        단독 > 880    (상관이 낮아도 품질이 낮으면 안 됨)
        C=16384  +4.3 +- 1.4  t=2.98  3/3
        C=65536  +6.6 +- 2.7  t=2.48  3/3
    C=16384 을 쓴 이유는 추론 비용이다 (65536 이면 L4 에서 4~8분).

    다만 조합 축이라 오늘 잰 전이율이 0.08 이다. 관문 +4.3 이 리더보드로
    +0.3 쯤일 수 있다. 그리고 MNCA 는 **누출판 plat_dev 로 학습**됐다 —
    TabM 만 고치고 나머지는 누출을 남기는 게 이겼던 구성과 같은 모양이다.

torch 필요
    MNCA 는 cdist 를 245,789 x 16,384 로 돌려야 해서 GPU 가 필요하다.
    numpy 되돌림 경로에서는 MNCA 를 **건너뛰고** 비중을 CB 0.30 / TabM 0.70
    으로 되돌린다. 그러면 최악의 경우에도 submit_30 과 같은 동작으로 완주한다.
"""
import io
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "submit_jaemin_30.zip")
NPZ_T = os.path.join(ROOT, "colab", "_dl", "npz_wd0")
NPZ_M = os.path.join(ROOT, "colab", "_dl", "mnca_npz")
OUT = os.path.join(ROOT, "submit_33")
MNCA_SEEDS = (42, 1, 777)
W_CB_NEW, W_MN, W_TM = 0.20, 0.20, 0.60

GPU_BLOCK = '''
# ---------------------------------------------------------------- GPU / MNCA
#
# torch 가 없거나 CUDA 가 없으면 MNCA 를 건너뛰고 CB 0.30 / TabM 0.70 으로
# 되돌린다. 느려도 완주하는 쪽을 택한다 — 제출 슬롯은 하루 5개다.
_TORCH = None
_GPU = False
try:
    import torch as _TORCH
    _GPU = _TORCH.cuda.is_available()
except Exception:
    _TORCH = None


def _mnca_predict(Xv, seeds, chunk=4096):
    """ModernNCA 순전파. 후보는 미리 인코딩돼 npz 에 들어 있다.

        z = E(x);  d = cdist(z, cand_z);  w = softmax(-d/tau);  p = w @ cand_y
    """
    dev = _TORCH.device("cuda")
    acc = None
    for sd in seeds:
        z = np.load(resolve(f"model/mnca_s{sd}.npz"), allow_pickle=False)
        mt = json.loads(str(z["meta"].item()))
        want = [str(c) for c in mt["features"]]
        nums, cats = _fm_prep(Xv, z)
        t = lambda a: _TORCH.as_tensor(np.ascontiguousarray(a), device=dev)
        W0, B0 = t(z["num_embedding_weight"]), t(z["num_embedding_bias"])
        CE = [t(z[f"cat_emb_{j}"]) for j in range(len(mt["cards"]))]
        E0w, E0b = t(z["E0_weight"]), t(z["E0_bias"])
        bw, bb = t(z["bn_weight"]), t(z["bn_bias"])
        bm, bv = t(z["bn_mean"]), t(z["bn_var"])
        E4w, E4b = t(z["E4_weight"]), t(z["E4_bias"])
        cz, cy = t(z["cand_z"]), t(z["cand_y"])
        tau = float(z["tau"])
        out = []
        with _TORCH.no_grad():
            for s in range(0, len(nums), chunk):
                xn = t(nums[s:s + chunk])
                xc = t(cats[s:s + chunk].astype(np.int64))
                e = [_TORCH.clamp(xn[:, :, None] * W0[None] + B0[None],
                                  min=0.0).reshape(len(xn), -1)]
                for j, emb in enumerate(CE):
                    e.append(emb[xc[:, j]])
                h = _TORCH.cat(e, 1) @ E0w.T + E0b
                h = (h - bm) / _TORCH.sqrt(bv + 1e-5) * bw + bb
                h = _TORCH.clamp(h, min=0.0) @ E4w.T + E4b
                d = _TORCH.cdist(h, cz)
                w = _TORCH.softmax(-d / max(tau, 1e-3), 1)
                out.append((w * cy[None, :]).sum(1).float().cpu().numpy())
        q = np.concatenate(out).astype(np.float64)
        acc = q if acc is None else acc + q
        del z
    return np.clip(acc / len(seeds), 1e-6, 1.0 - 1e-6)

'''

NEW_BLEND = '''        # ---------------- MNCA (GPU 필요). 없으면 건너뛰고 비중을 되돌린다
        p_mn = None
        if _GPU and W_MNCA > 0.0:
            try:
                _Xm = np.c_[Xf.to_numpy(dtype=np.float64),
                            np.ones(len(Xf))]        # abs_regime = 1
                _ord_mn = _mnca_predict(_Xm, MNCA_SEEDS)
                _tm2 = dict(zip(ordered[ID_COL].tolist(), _ord_mn.tolist()))
                p_mn = np.array([_tm2[r] for r in test[ID_COL].tolist()],
                                dtype=np.float64)
                print(f" MNCA     mean={p_mn.mean():.4f}  "
                      f"상관={np.corrcoef(p_tabm, p_mn)[0, 1]:.4f}")
            except Exception as exc:                 # noqa: BLE001
                print(f" MNCA 실패, 건너뜀: {type(exc).__name__} {exc}")
                p_mn = None
        if p_mn is None:
            print(" MNCA 없음 -> CB 0.30 / TabM 0.70 로 되돌림")
            preds = 0.30 * p_cb + 0.70 * p_tabm
        else:
            preds = W_CB * p_cb + W_MNCA * p_mn + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}")'''

OLD_BLEND = '''        preds = W_CB * p_cb + W_HGB * p_hgb + W_TABM * p_tabm
        print(f" blended  mean={preds.mean():.4f}  "
              f"({W_CB:.2f}/{W_HGB:.2f}/{W_TABM:.2f})")'''


def main():
    z = zipfile.ZipFile(SRC)
    src = z.read("script.py").decode("utf-8")

    old_w = "W_CB, W_HGB, W_TABM = 0.3, 0.0, 0.7"
    assert old_w in src, "비중 줄을 못 찾았다"
    src = src.replace(old_w,
                      f"W_CB, W_HGB, W_TABM = {W_CB_NEW}, 0.0, {W_TM}\n"
                      f"W_MNCA = {W_MN}\n"
                      f"MNCA_SEEDS = {tuple(MNCA_SEEDS)}")

    anchor = "\ndef predict_hgb("
    assert anchor in src
    src = src.replace(anchor, GPU_BLOCK + anchor, 1)

    assert OLD_BLEND in src, "혼합 블록을 못 찾았다"
    src = src.replace(OLD_BLEND, NEW_BLEND)

    assert "fbregime_" in src and "CALIB_T" in src

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)
    drop = {"model/hgb.npz", "requirements.txt"}
    keep = [n for n in z.namelist() if n != "script.py" and n not in drop]
    for n in keep:
        p = os.path.join(OUT, n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "wb").write(z.read(n))
    io.open(os.path.join(OUT, "requirements.txt"), "w", encoding="utf-8",
            newline="\n").write("numpy>=1.24\npandas>=2.0\ntorch>=2.0\n")

    # wd=0 으로 재학습한 1군 모델로 교체
    for new, old in (("all_s42", "all_tabm_seed_42"),
                     ("regular_s42", "regular_tabm_seed_42")):
        shutil.copy(os.path.join(NPZ_T, new + ".npz"),
                    os.path.join(OUT, "model", old + ".npz"))
    for sd in MNCA_SEEDS:
        shutil.copy(os.path.join(NPZ_M, f"mnca_s{sd}.npz"),
                    os.path.join(OUT, "model", f"mnca_s{sd}.npz"))

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  {tot/1e6:.1f}MB / 10,240MB = {tot/1.024e10*100:.2f}%")
    print(f"  비중  CatBoost {W_CB_NEW} / MNCA {W_MN} / TabM {W_TM}")
    print(f"  1군 TabM 은 weight_decay 0 으로 재학습, MNCA 시드 {MNCA_SEEDS}")


if __name__ == "__main__":
    main()
