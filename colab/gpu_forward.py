# -*- coding: utf-8 -*-
"""TabM 순전파를 torch GPU 로 옮기고 numpy 판과 같은 값이 나오는지 검증한다.

왜 이걸 먼저 하나
    지금 추론은 numpy 손구현이고 CPU 로 245,789행에 약 9분이 걸린다.
    평가 서버에 L4 GPU 22.4GB 가 있는데 안 쓰고 있고, 그 때문에 시드를
    1개밖에 못 태운다. 시드 곡선 실측이 k=1 -> k=4 에서 +8.0 이었다.

    그런데 GPU 로 옮기면서 값이 미세하게라도 달라지면, 리더보드로 잡아둔
    시프트(-0.0070)와 비중(0.30)이 전부 어긋난다. 그래서 **속도보다 일치가
    먼저다.** 여기서 최대차를 확인하고 넘어간다.

설계 — 교체가 아니라 추가
    torch 가 없거나 CUDA 가 없거나 GPU 경로가 터지면 **numpy 로 되돌린다.**
    제출 슬롯이 하루 5개고 실패하면 한 장을 버린다. 느려도 도는 게 낫다.

        _tabm_forward()        기존 이름 유지. 안에서 갈라진다
          -> _tabm_forward_torch()   GPU 있으면
          -> _tabm_forward_numpy()   없거나 실패하면

수치 일치에 대해
    같은 float32 연산이라도 matmul 누적 순서가 달라 1e-6 수준 차이는 난다.
    그게 점수에 영향을 주는지가 기준이다 — 예측이 0.5 근처이고 Brier 를
    소수 넷째 자리에서 다투므로 1e-5 이하면 무해하다.

    규칙 4 감사는 별개다. '같은 행이면 같은 값' 이 요구사항인데, 행마다
    독립적인 순전파라 청크 구성이 바뀌어도 그 행의 출력은 자기 입력에만
    의존한다. 그래도 실제로 확인한다.
"""
import os
import sys
import time

import numpy as np

DEV = None
try:
    import torch
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except Exception:
    torch = None


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def forward_numpy(nums, cats, meta, ar, chunk=1024):
    """submit_27 의 _tabm_forward 와 동일. 기준값이다."""
    if len(nums) == 0:
        return np.empty(0, dtype="float32")
    w = ar["num_embedding_weight"]
    b = ar["num_embedding_bias"]
    num_repr = np.maximum(nums[:, :, None] * w[None] + b[None], 0.0)
    num_repr = num_repr.reshape(len(nums), -1)
    k = int(meta["k"])
    cards = list(meta["cat_cardinalities"])
    nd = num_repr.shape[1]
    offs = np.cumsum([0] + cards[:-1]) + nd
    out = []
    for s in range(0, len(num_repr), chunk):
        cn, cc = num_repr[s:s + chunk], cats[s:s + chunk]
        h = None
        for blk in range(int(meta["n_blocks"])):
            W = ar[f"block_{blk}_weight"]
            r = ar[f"block_{blk}_r"]
            sc_ = ar[f"block_{blk}_s"]
            bb = ar[f"block_{blk}_bias"]
            if blk == 0:
                h = np.matmul(cn[:, None, :] * r[None, :, :nd], W[:, :nd].T)
                for j, c in enumerate(cards):
                    o = int(offs[j])
                    idx = cc[:, j]
                    sr = r[:, o:o + c][:, idx].T
                    sw = W[:, o:o + c][:, idx].T
                    h += sr[:, :, None] * sw[:, None, :]
            else:
                h = np.matmul(h * r[None], W.T)
            h = np.maximum(h * sc_[None] + bb[None], 0.0)
        lg = np.einsum("bki,kio->bko", h, ar["output_weight"]) \
            + ar["output_bias"][None]
        out.append(_sig(lg[:, :, 0]).mean(axis=1))
    return np.concatenate(out) if out else np.empty(0, dtype="float32")


def forward_torch(nums, cats, meta, ar, chunk=16384, dev=None):
    """같은 계산을 torch 로.

    메모리 요령 — rank-1 어댑터를 가중치에 미리 접는다
        원식은  h = (nr * r_k) @ W.T  인데, 이걸 그대로 쓰면 (B, k, d_in)
        중간 텐서가 생긴다. k=32, d_in=816, B=65536 이면 6.8GB 다.

        (nr * r_k) @ W.T = nr @ (r_k * W).T  이므로 멤버별 가중치
        Wk[k] = r[k] * W 를 **한 번만** 만들어 두면 중간 텐서가 사라지고
        einsum('bd,kod->bko') 한 방으로 끝난다. Wk 는 32x256x816x4 = 27MB 다.

        결과 텐서 (B, k, d_block) 만 남아 B=16384 에서 537MB 다.

    범주 열은 numpy 판과 같은 방식으로 처리한다 — 원핫을 만들지 않고
    해당 열만 골라 더한다. 그래서 값이 같다.
    """
    dev = dev or DEV
    if len(nums) == 0:
        return np.empty(0, dtype="float32")
    t = lambda a: torch.as_tensor(np.ascontiguousarray(a), device=dev)   # noqa
    W0 = t(ar["num_embedding_weight"])
    B0 = t(ar["num_embedding_bias"])
    cards = list(meta["cat_cardinalities"])
    nb = int(meta["n_blocks"])
    BS = [t(ar[f"block_{i}_s"]) for i in range(nb)]
    BB = [t(ar[f"block_{i}_bias"]) for i in range(nb)]
    OW, OB = t(ar["output_weight"]), t(ar["output_bias"])
    nd = nums.shape[1] * W0.shape[1]
    offs = np.cumsum([0] + cards[:-1]) + nd

    # 어댑터를 접은 멤버별 가중치를 미리 만든다
    r0 = t(ar["block_0_r"])
    w0 = t(ar["block_0_weight"])
    WK0 = r0[:, None, :nd] * w0[None, :, :nd]            # (k, d_block, nd)
    # 범주 쪽은 (k, d_block, card) 조각을 그대로 쓴다 (인덱싱으로 골라 쓴다)
    CATW = [(r0[:, None, offs[j]:offs[j] + c]
             * w0[None, :, offs[j]:offs[j] + c])        # (k, d_block, card)
            for j, c in enumerate(cards)]
    WK = [None]
    for i in range(1, nb):
        WK.append(t(ar[f"block_{i}_r"])[:, None, :]
                  * t(ar[f"block_{i}_weight"])[None, :, :])

    out = []
    with torch.no_grad():
        for s in range(0, len(nums), chunk):
            xn = t(nums[s:s + chunk])
            xc = t(cats[s:s + chunk].astype(np.int64))
            nr = torch.clamp(xn[:, :, None] * W0[None] + B0[None], min=0.0)
            nr = nr.reshape(len(xn), -1)
            h = torch.einsum("bd,kod->bko", nr, WK0)
            for j in range(len(cards)):
                h = h + CATW[j][:, :, xc[:, j]].permute(2, 0, 1)
            h = torch.clamp(h * BS[0][None] + BB[0][None], min=0.0)
            for blk in range(1, nb):
                h = torch.einsum("bkd,kod->bko", h, WK[blk])
                h = torch.clamp(h * BS[blk][None] + BB[blk][None], min=0.0)
            lg = torch.einsum("bki,kio->bko", h, OW) + OB[None]
            out.append(torch.sigmoid(lg[:, :, 0]).mean(dim=1).float().cpu().numpy())
            del h, nr, xn, xc
    return np.concatenate(out) if out else np.empty(0, dtype="float32")


if __name__ == "__main__":
    import json
    mp = sys.argv[1] if len(sys.argv) > 1 else "model/all_tabm_seed_42.npz"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 60000
    z = np.load(mp, allow_pickle=False)
    # 지인 내보내기는 'metadata', 우리 내보내기는 'meta' 로 저장한다
    mk = "metadata" if "metadata" in z.files else "meta"
    meta = json.loads(str(z[mk].item()))
    if "cat_cardinalities" not in meta:
        meta["cat_cardinalities"] = meta["cards"]
    nnum = z["num_embedding_weight"].shape[0]
    cards = list(meta["cat_cardinalities"])
    print(f"  모델 {os.path.basename(mp)}   수치 {nnum}열  범주 {len(cards)}열  "
          f"k={meta['k']}  블록 {meta['n_blocks']}  d={meta['d_block']}")
    print(f"  장치 {DEV}\n")

    rng = np.random.default_rng(0)
    nums = rng.standard_normal((n, nnum)).astype(np.float32)
    cats = np.stack([rng.integers(0, c, n) for c in cards], 1).astype(np.int64)

    t0 = time.time(); pn = forward_numpy(nums, cats, meta, z, 4096)
    tn = time.time() - t0
    t0 = time.time(); pt = forward_torch(nums, cats, meta, z, 65536)
    tt = time.time() - t0

    d = np.abs(pn.astype(np.float64) - pt.astype(np.float64))
    print(f"  numpy  {tn:7.2f}s   torch  {tt:7.2f}s   가속 {tn/max(tt,1e-9):5.1f}배")
    print(f"  최대차 {d.max():.3e}   평균차 {d.mean():.3e}   "
          f"1e-5 초과 {int((d > 1e-5).sum()):,}개 / {n:,}")
    print(f"  245,789행 환산   numpy {tn*245789/n/60:5.2f}분   "
          f"torch {tt*245789/n/60:5.2f}분")

    # 청크 크기를 바꿔도 같은 값이 나오나 (규칙 4 의 전제)
    p2 = forward_torch(nums, cats, meta, z, 8192)
    print(f"  청크 65536 vs 8192 최대차 {np.abs(pt-p2).max():.3e}"
          f"   (행마다 독립이어야 한다)")
    print("\n  최대차가 1e-5 이하여야 리더보드로 잡은 시프트·비중이 그대로 유효하다.")
