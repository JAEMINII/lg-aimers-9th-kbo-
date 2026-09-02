# -*- coding: utf-8 -*-
"""flat MLP 를 제출용으로 학습하고 numpy 로 추출한다.

구성 (관문에서 확정, 두 채점 지표 모두 1위)
    전체 시즌 균등 학습 (시즌가중 없음), 6 epoch, 시드 3개 평균
    PLR 수치임베딩 + 범주임베딩 + 3층 MLP
    loss = 0.5 BCE + 0.5 Brier

    관문 전체채점   CatBoost 단독 903.2  ->  + flat MLP 0.35  937.2

왜 시즌가중을 안 주나
    CatBoost 가 이미 시즌가중 2.0 을 쓴다. MLP 에도 주면 같은 보정을 두 번 해서
    상관이 올라가고 다양성이 죽는다. 실측으로 확인됐다.
        flat(가중없음) CB상관 0.847  ->  sw4(가중4.5) 0.902
        섞은 뒤 전체채점 929.6  ->  921.1

제출은 VS=2025 로 학습한다 (2019~2024 전체가 학습 구간).
관문 검증은 VS=2024 로 따로 돌린다.
"""
import json
import os
import sys

import numpy as np
import torch

SC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SC)
DATA = "/workspace/aimers/data"
OUT = "/workspace/aimers/out"
SEEDS = (42, 1, 777)
EPOCHS = 6

import features44 as F                                        # noqa: E402
import mlp_gpu as M                                           # noqa: E402


def export(model, path):
    """state_dict 를 numpy 배열로 뽑는다.

    PLR 임베딩:  2*pi*w*x -> [cos, sin] -> per-feature Linear + ReLU
    범주:        Embedding 인덱싱
    본체:        Linear x3 + ReLU,  head -> sigmoid
    """
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in model.state_dict().items()}
    out = {}
    for k, v in sd.items():
        out[k.replace(".", "__")] = v
    np.savez_compressed(path, **out)
    return list(sd.keys())


if __name__ == "__main__":
    VS = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    tag = f"vs{VS}"
    # features44 / mlp_gpu 는 import 시점에 VS=2024 로 데이터를 만든다.
    # 제출용은 VS=2025 가 필요하므로 여기서 다시 만든다.
    d = F.build(DATA, VS=VS)
    X, y = d["X44"], d["y"]
    m_tr = d["m_tr"]
    Xn, Xc, cards = M.prep(X, m_tr, d["cat_idx"])
    M.XN = torch.from_numpy(Xn)
    M.XC = torch.from_numpy(Xc)
    M.YY = torch.from_numpy(y.astype(np.float32))
    tr_idx = np.where(m_tr)[0]
    print(f"VS={VS}  학습 {len(tr_idx):,}행   수치 {Xn.shape[1]}열  범주 {Xc.shape[1]}열",
          flush=True)

    meta = dict(n_num=int(Xn.shape[1]), cards=[int(c) for c in cards],
                seeds=list(SEEDS), epochs=EPOCHS, vs=VS,
                d_emb=24, d_hidden=[128, 256, 128], d_cat=8,
                features=d["F44"], cat_idx=[int(i) for i in d["cat_idx"]])
    os.makedirs(os.path.join(OUT, "mlp_export"), exist_ok=True)
    for sd_ in SEEDS:
        torch.manual_seed(sd_)
        m = M.MLP_PLR(Xn.shape[1], cards).to(M.DEV)
        M.train(m, tr_idx, EPOCHS, 1e-3, tag=f"seed{sd_}")
        keys = export(m, os.path.join(OUT, "mlp_export", f"{tag}_seed{sd_}.npz"))
        print(f"  seed {sd_} 저장", flush=True)
        del m
        torch.cuda.empty_cache()
    meta["state_keys"] = keys
    with open(os.path.join(OUT, "mlp_export", f"{tag}_meta.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print("state_dict 키:", keys)
