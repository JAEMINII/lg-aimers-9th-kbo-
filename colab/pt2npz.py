# -*- coding: utf-8 -*-
"""MSA .pt(payload) -> 추론용 npz. msa_s2 기준본과 전 배열 일치로 검증된 매핑."""
import json
import sys

import numpy as np
import torch

NAME_MAP = {
    "num_module.linear.weight": "num_embedding_weight",
    "num_module.linear.bias": "num_embedding_bias",
    "output.weight": "output_weight",
    "output.bias": "output_bias",
}


def convert(pt_path, npz_path):
    pay = torch.load(pt_path, map_location="cpu", weights_only=False)
    prep = pay["preprocessor"]
    sd = pay["model_state"]
    fn = prep["feature_names"]
    cc = prep["cat_cols"]
    nc = prep["num_cols"]
    out = {}
    cards = []
    for i, c in enumerate(cc):
        vals = sorted(float(v) for v in prep["cat_values"][c])
        out[f"catkey_{i}"] = np.array(vals, np.float64)
        out[f"catval_{i}"] = np.arange(1, len(vals) + 1, dtype=np.int64)
        cards.append(len(vals) + 1)
    out["cat_idx"] = np.array([fn.index(c) for c in cc], np.int64)
    out["med"] = np.array([prep["medians"][c] for c in nc], np.float64)
    out["mu"] = np.array([prep["means"][c] for c in nc], np.float64)
    out["sd"] = np.array([prep["stds"][c] for c in nc], np.float64)
    out["has_nan"] = np.array([c in set(prep["missing_cols"]) for c in nc], bool)
    for k, v in sd.items():
        if k in NAME_MAP:
            nk = NAME_MAP[k]
        elif k.startswith("backbone.blocks."):
            b, leaf = k.split(".")[2], k.split(".")[-1]
            nk = f"block_{b}_{leaf}"
        else:
            raise KeyError(k)
        out[nk] = v.numpy().astype(np.float32)
    k_ens = int(sd["output.weight"].shape[0])
    meta = {"features": fn, "cards": cards, "k": k_ens,
            "n_blocks": sum(1 for k2 in sd if k2.endswith(".0.weight")
                            and k2.startswith("backbone")),
            "d_block": int(sd["backbone.blocks.1.0.weight"].shape[0]),
            "num_embeddings": "linear_relu",
            "backbone_kind": "batch_ensemble",
            "cat_cardinalities": cards,
            "trained_with": "state6 decay15 + S2 (1군 라우팅용)"}
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(npz_path, **out)
    return npz_path


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
    print("saved", sys.argv[2])
