# -*- coding: utf-8 -*-
"""submit_29 = submit_27(LB 1058)에서 **1군 경로를 8시드 평균으로**.

바꾸는 것 하나
    submit_27   1군 = 0.6 x all(seed42) + 0.4 x regular(seed42)
    submit_29   1군 = 0.6 x all(8시드 평균) + 0.4 x regular(8시드 평균)

    퓨처스 경로(fbregime_*, 11.8%)와 CatBoost 비중(0.30)은 그대로 둔다.
    한 번에 여러 개를 실으면 submit_26 때처럼(다섯 개 동시 변경, 1057->1056)
    뭐가 효과인지 못 가린다.

근거 — 시드 곡선 실측 (관문 VS=2024, 시드 8개, 조합 평균)
    시드수   전체     시드편차
      1     904.9     6.45      <- submit_27 이 여기 있다
      2     912.2     6.74      +7.4
      4     912.8     3.26      +8.0
      8     912.7     0.00      +7.9
    score(k) = 914.3 - 8.3/k.  k=4 에서 거의 눕는다.

    '시드편차 6.45' 가 요점이다. 시드 8개의 단독 점수가 6.45점 폭으로 흩어져
    있는데 지금은 그중 하나를 뽑아 쓰고 있다. 평균은 그 도박을 없앤다.

왜 지인 코드로 학습했나
    리더보드가 학습 코드로 갈린다 (지인 1046/1048/1057/1058 vs 우리
    1020/1048/1049/1052). 우리 코드로 8시드를 만들면 기반이 5점 낮은 데서
    시작한다. train_conditional.py 가 --seeds 를 지원하므로 지인 코드를
    그대로 8시드로 돌렸다. Stage2 에폭도 1 그대로다.

GPU 추론 — 이게 8시드를 가능하게 한다
    numpy 손구현으로는 245,789행 1패스에 약 9분이라 예산(10분)이 꽉 찬다.
    시드를 늘릴 여지가 0 이었다. torch GPU 로 옮기니

        numpy 351.34s  ->  torch 2.18s   (60,000행, 161배)
        최대차 1.192e-07,  1e-5 초과 0개  <- 값이 같다
        청크 65536 vs 8192 최대차 0.000e+00  <- 규칙 4 안전

    245,789행 환산으로 9초다. 8시드면 72초. 예산 안에 넉넉히 들어간다.
    pip install torch 는 이 서버에서 70초였다 (설치 예산 10분).

안전장치 — torch 나 CUDA 가 없으면 시드 1개로 축소
    numpy 되돌림 경로를 8시드로 돌리면 예산을 훨씬 넘긴다. 그래서 GPU 가
    없으면 **시드 1개만** 쓴다. 그러면 최악의 경우에도 submit_27 과 같은
    동작으로 완주한다. 느려도 도는 게 낫다 — 제출 슬롯은 하루 5개다.
"""
import io
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "submit_jaemin_27.zip")
NPZ = os.path.join(ROOT, "colab", "_dl", "npz8lb")
OUT = os.path.join(ROOT, "submit_29")
SEEDS = (42, 1, 777, 2, 7, 13, 99, 2024)

GPU_BLOCK = '''
# ---------------------------------------------------------------- GPU 순전파
#
# numpy 손구현과 **같은 값**이 나오는지 확인했다 (최대차 1.192e-07, 60,000행).
# 청크 크기를 바꿔도 최대차 0.000e+00 이라 규칙 4 도 안전하다 — 행마다 독립이다.
#
# 메모리 요령: rank-1 어댑터를 가중치에 미리 접는다.
#   (nr * r_k) @ W.T = nr @ (r_k * W).T
# 그대로 쓰면 (B, k, d_in) 중간 텐서가 생겨 청크 65536 에서 6.8GB 다.
# 접어두면 사라지고 (B, k, d_block) 만 남는다.
_TORCH = None
_GPU = False
try:
    import torch as _TORCH
    _GPU = _TORCH.cuda.is_available()
except Exception:
    _TORCH = None


def _tabm_forward_torch(nums, cats, metadata, archive, chunk_size=16384):
    dev = _TORCH.device("cuda")
    t = lambda a: _TORCH.as_tensor(np.ascontiguousarray(a), device=dev)
    W0 = t(archive["num_embedding_weight"])
    B0 = t(archive["num_embedding_bias"])
    cards = list(metadata["cat_cardinalities"])
    nb = int(metadata["n_blocks"])
    BS = [t(archive[f"block_{i}_s"]) for i in range(nb)]
    BB = [t(archive[f"block_{i}_bias"]) for i in range(nb)]
    OW, OB = t(archive["output_weight"]), t(archive["output_bias"])
    nd = nums.shape[1] * W0.shape[1]
    offs = np.cumsum([0] + cards[:-1]) + nd
    r0, w0 = t(archive["block_0_r"]), t(archive["block_0_weight"])
    WK0 = r0[:, None, :nd] * w0[None, :, :nd]
    CATW = [(r0[:, None, offs[j]:offs[j] + c] * w0[None, :, offs[j]:offs[j] + c])
            for j, c in enumerate(cards)]
    WK = [None] + [t(archive[f"block_{i}_r"])[:, None, :]
                   * t(archive[f"block_{i}_weight"])[None, :, :]
                   for i in range(1, nb)]
    out = []
    with _TORCH.no_grad():
        for s in range(0, len(nums), chunk_size):
            xn = t(nums[s:s + chunk_size])
            xc = t(cats[s:s + chunk_size].astype(np.int64))
            nr = _TORCH.clamp(xn[:, :, None] * W0[None] + B0[None], min=0.0)
            nr = nr.reshape(len(xn), -1)
            h = _TORCH.einsum("bd,kod->bko", nr, WK0)
            for j in range(len(cards)):
                h = h + CATW[j][:, :, xc[:, j]].permute(2, 0, 1)
            h = _TORCH.clamp(h * BS[0][None] + BB[0][None], min=0.0)
            for blk in range(1, nb):
                h = _TORCH.einsum("bkd,kod->bko", h, WK[blk])
                h = _TORCH.clamp(h * BS[blk][None] + BB[blk][None], min=0.0)
            lg = _TORCH.einsum("bki,kio->bko", h, OW) + OB[None]
            out.append(_TORCH.sigmoid(lg[:, :, 0]).mean(dim=1).float().cpu().numpy())
            del h, nr, xn, xc
    return np.concatenate(out) if out else np.empty(0, dtype="float32")

'''


def main():
    z = zipfile.ZipFile(SRC)
    src = z.read("script.py").decode("utf-8")

    # ① numpy 판 이름을 바꾸고 갈림길을 만든다
    old_def = "def _tabm_forward(nums, cats, metadata: dict, archive, chunk_size: int = 1024):"
    assert old_def in src, "_tabm_forward 를 못 찾았다"
    src = src.replace(old_def, GPU_BLOCK + "\n" + old_def.replace(
        "_tabm_forward(", "_tabm_forward_numpy("))
    # 갈림길 함수를 numpy 판 뒤에 넣는다
    anchor = "\ndef predict_hgb("
    assert anchor in src
    src = src.replace(anchor, '''
def _tabm_forward(nums, cats, metadata: dict, archive, chunk_size: int = 1024):
    """GPU 가 있으면 torch, 없거나 실패하면 numpy. 값은 같다."""
    if _GPU:
        try:
            return _tabm_forward_torch(nums, cats, metadata, archive)
        except Exception as exc:                       # noqa: BLE001
            print(f" GPU 경로 실패, numpy 로 되돌림: {type(exc).__name__} {exc}")
    return _tabm_forward_numpy(nums, cats, metadata, archive, chunk_size)

''' + anchor, 1)

    # ② 1군 경로를 8시드 평균으로
    old_1gun = '''        meta_a, arc_a = load_bundle(resolve("model/all_tabm_seed_42.npz"))
        p_ord = predict_frame(Xf, meta_a, arc_a, chunk_size=4096)

        rows_r = np.flatnonzero(~isf_o)
        if len(rows_r):
            mt, ar = load_bundle(resolve("model/regular_tabm_seed_42.npz"))
            pr = predict_frame(Xf.iloc[rows_r], mt, ar, chunk_size=4096)
            p_ord[rows_r] = 0.6 * p_ord[rows_r] + 0.4 * pr'''
    new_1gun = '''        # GPU 가 없으면 시드 1개로 줄인다 — numpy 되돌림으로 8패스를 돌리면
        # 추론 예산(10분)을 훨씬 넘긴다. 느려도 완주하는 쪽을 택한다.
        _sds = TABM_SEEDS if _GPU else TABM_SEEDS[:1]
        print(f" TabM 시드 {len(_sds)}개 사용 (GPU={_GPU})")

        def _avg(tag, frame):
            # 전처리는 **한 번만** 한다. _tabm_transform 이 pitcher_id(792) /
            # batter_id(830) 를 파이썬 dict 로 매핑해서 호출당 15초씩 든다.
            # 8시드는 같은 전처리기(X_all 에 한 번 fit)를 공유하므로 재사용해도
            # 값이 같다. 다르면 조용히 틀리는 대신 즉시 멈춘다.
            acc = None
            base = None
            nums = cats = None
            for _sd in _sds:
                _mt, _ar = load_bundle(resolve(f"model/{tag}_s{_sd}.npz"))
                if base is None:
                    base = _mt
                    nums, cats = _tabm_transform(frame, _mt)
                else:
                    for _k in ("feature_names", "cat_cols", "num_cols",
                               "medians", "means", "stds", "missing_cols",
                               "cat_values", "num_embeddings", "k",
                               "n_blocks", "d_block"):
                        if _mt[_k] != base[_k]:
                            raise ValueError(
                                f"시드 {_sd} 의 {_k} 가 기준과 다르다")
                _q = _tabm_forward(nums, cats, _mt, _ar, 4096)
                acc = _q if acc is None else acc + _q
            return acc / len(_sds)

        p_ord = _avg("all", Xf)

        rows_r = np.flatnonzero(~isf_o)
        if len(rows_r):
            pr = _avg("regular", Xf.iloc[rows_r])
            p_ord[rows_r] = 0.6 * p_ord[rows_r] + 0.4 * pr'''
    assert old_1gun in src, "1군 블록을 못 찾았다"
    src = src.replace(old_1gun, new_1gun)

    # ③ 시드 목록 상수
    src = src.replace("W_CB, W_HGB, W_TABM =",
                      f"TABM_SEEDS = {tuple(SEEDS)}\nW_CB, W_HGB, W_TABM =", 1)

    assert "fbregime_" in src, "퓨처스 경로가 사라졌다"
    assert "CALIB_T" in src, "온도 상수가 사라졌다"

    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(os.path.join(OUT, "model"))
    io.open(os.path.join(OUT, "script.py"), "w", encoding="utf-8",
            newline="\n").write(src)

    # submit_27 에서 가져오되 seed42 단일 모델 두 개는 뺀다 (8시드로 대체)
    drop = {"model/all_tabm_seed_42.npz", "model/regular_tabm_seed_42.npz",
            "model/hgb.npz", "requirements.txt"}
    keep = [n for n in z.namelist() if n != "script.py" and n not in drop]
    for n in keep:
        p = os.path.join(OUT, n)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        io.open(p, "wb").write(z.read(n))
    io.open(os.path.join(OUT, "requirements.txt"), "w",
            encoding="utf-8", newline="\n").write(
        "numpy>=1.24\npandas>=2.0\ntorch>=2.0\n")

    n_ok = 0
    for br in ("all", "regular"):
        for sd in SEEDS:
            f = os.path.join(NPZ, f"{br}_s{sd}.npz")
            if not os.path.exists(f):
                raise SystemExit(f"없음: {f}")
            shutil.copy(f, os.path.join(OUT, "model", f"{br}_s{sd}.npz"))
            n_ok += 1

    tot = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(OUT) for f in fs)
    print(f"  {OUT}  {tot/1e6:.1f}MB / 10,240MB = {tot/1.024e10*100:.2f}%")
    print(f"  8시드 모델 {n_ok}개, submit_27 에서 그대로 {len(keep)}개")
    print(f"  HistGB(hgb.npz) 제거 — 비중 0 이라 안 쓴다")


if __name__ == "__main__":
    main()
