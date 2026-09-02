# -*- coding: utf-8 -*-
"""NumPy-only submission-time inference for the exported official TabM."""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from preprocess import ID, TARGET, build_inference_features, deserialize_history, sort_by_row_id


HERE = os.path.dirname(os.path.abspath(__file__))


def resolve(rel: str) -> str:
    candidates = [
        os.path.join(os.getcwd(), rel),
        os.path.join(HERE, rel),
        os.path.join(HERE, "..", rel),
        rel,
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"cannot resolve {rel}; checked {candidates}")


def load_bundle(path: str):
    archive = np.load(path, allow_pickle=False)
    metadata = json.loads(str(archive["metadata"].item()))
    return metadata, archive


def _tabm_transform(X: pd.DataFrame, metadata: dict):
    expected = list(metadata["feature_names"])
    missing = [col for col in expected if col not in X.columns]
    if missing:
        raise ValueError(f"missing exported features: {missing}")
    X = X[expected]

    cat_cols = list(metadata["cat_cols"])
    num_cols = list(metadata["num_cols"])
    cats = np.zeros((len(X), len(cat_cols)), dtype=np.int64)
    for j, col in enumerate(cat_cols):
        mapping = {value: i + 1 for i, value in enumerate(metadata["cat_values"][col])}
        cats[:, j] = X[col].map(mapping).fillna(0).astype("int64").to_numpy()

    numeric = X[num_cols].apply(pd.to_numeric, errors="coerce")
    parts = []
    for col in num_cols:
        values = numeric[col].astype("float64").fillna(float(metadata["medians"][col]))
        values = (values - float(metadata["means"][col])) / float(metadata["stds"][col])
        parts.append(values.to_numpy(dtype="float32"))
    for col in metadata["missing_cols"]:
        parts.append(numeric[col].isna().to_numpy(dtype="float32"))
    nums = np.column_stack(parts).astype("float32", copy=False)
    return nums, cats


def _one_hot(cats: np.ndarray, cardinalities: list[int]) -> np.ndarray:
    blocks = []
    rows = np.arange(len(cats))
    for j, cardinality in enumerate(cardinalities):
        block = np.zeros((len(cats), cardinality), dtype="float32")
        values = np.clip(cats[:, j], 0, cardinality - 1)
        block[rows, values] = 1.0
        blocks.append(block)
    return np.column_stack(blocks) if blocks else np.empty((len(cats), 0), dtype="float32")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def predict_frame(X: pd.DataFrame, metadata: dict, archive, chunk_size: int = 1024):
    """Predict already-transformed 44-feature rows independently."""
    nums, cats = _tabm_transform(X, metadata)
    if len(nums) == 0:
        return np.empty(0, dtype="float32")
    if metadata["num_embeddings"] == "linear_relu":
        weight = archive["num_embedding_weight"]
        bias = archive["num_embedding_bias"]
        num_repr = np.maximum(nums[:, :, None] * weight[None, :, :] + bias[None, :, :], 0.0)
        num_repr = num_repr.reshape(len(nums), -1)
    elif metadata["num_embeddings"] == "none":
        num_repr = nums
    else:
        raise ValueError(f"unsupported exported embedding: {metadata['num_embeddings']}")

    k = int(metadata["k"])
    cardinalities = list(metadata["cat_cardinalities"])
    num_dim = num_repr.shape[1]
    cat_offsets = np.cumsum([0] + cardinalities[:-1]) + num_dim
    if metadata["backbone_kind"] != "batch_ensemble":
        raise ValueError("optimized exporter currently requires batch_ensemble TabM")
    predictions = []
    for start in range(0, len(num_repr), chunk_size):
        end = start + chunk_size
        chunk_nums = num_repr[start:end]
        chunk_cats = cats[start:end]
        h = np.empty((len(chunk_nums), k, int(metadata["d_block"])), dtype="float32")
        for block in range(int(metadata["n_blocks"])):
            weight = archive[f"block_{block}_weight"]
            r = archive[f"block_{block}_r"]
            s = archive[f"block_{block}_s"]
            bias_block = archive[f"block_{block}_bias"]
            if block == 0:
                # Exact equivalent of one-hot -> BatchEnsemble -> Linear, without
                # materializing the 2,489-dimensional one-hot matrix.
                h = np.matmul(
                    chunk_nums[:, None, :] * r[None, :, :num_dim],
                    weight[:, :num_dim].T,
                )
                for j, cardinality in enumerate(cardinalities):
                    offset = int(cat_offsets[j])
                    indices = chunk_cats[:, j]
                    selected_r = r[:, offset:offset + cardinality][:, indices].T
                    selected_w = weight[:, offset:offset + cardinality][:, indices].T
                    h += selected_r[:, :, None] * selected_w[:, None, :]
            else:
                h = np.matmul(h * r[None, :, :], weight.T)
            h = h * s[None, :, :] + bias_block[None, :, :]
            h = np.maximum(h, 0.0)

        logits = np.einsum(
            "bki,kio->bko", h, archive["output_weight"]
        ) + archive["output_bias"][None, :, :]
        predictions.append(_sigmoid(logits[:, :, 0]).mean(axis=1))
    return np.concatenate(predictions) if predictions else np.empty(0, dtype="float32")


def main():
    model_paths = {
        "all": os.path.join(HERE, "model", "all_tabm_seed_42.npz"),
        "futures": os.path.join(HERE, "model", "futures_tabm_seed_42.npz"),
        "regular": os.path.join(HERE, "model", "regular_tabm_seed_42.npz"),
    }
    missing_models = [path for path in model_paths.values() if not os.path.exists(path)]
    if missing_models:
        raise FileNotFoundError(f"missing conditional TabM models: {missing_models}")
    with open(os.path.join(HERE, "model", "history.json"), encoding="utf-8") as f:
        history = deserialize_history(json.load(f))

    test = pd.read_csv(resolve("data/test.csv"), encoding="utf-8-sig")
    sample = pd.read_csv(resolve("data/sample_submission.csv"), encoding="utf-8-sig")
    ordered = sort_by_row_id(test)
    X = build_inference_features(ordered, history)

    meta_all, archive_all = load_bundle(model_paths["all"])
    pred_all = predict_frame(X, meta_all, archive_all)
    archive_all.close()

    game_type = ordered["game_type"].astype(str).to_numpy()
    pred_specialized = np.zeros(len(ordered), dtype="float32")
    for branch, code in (("futures", "F"), ("regular", "R")):
        mask = game_type == code
        if not mask.any():
            continue
        metadata, archive = load_bundle(model_paths[branch])
        pred_specialized[mask] = predict_frame(X.iloc[np.flatnonzero(mask)], metadata, archive)
        archive.close()
    pred = np.clip(0.6 * pred_all + 0.4 * pred_specialized, 0.0, 1.0)
    pred_map = dict(zip(ordered[ID].tolist(), pred))
    if any(row_id not in pred_map for row_id in sample[ID]):
        raise ValueError("submission IDs do not match test IDs")
    sample[TARGET] = [pred_map[row_id] for row_id in sample[ID]]
    output = os.path.join(os.getcwd(), "output", "submission.csv")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    sample.to_csv(output, index=False, encoding="utf-8")
    print(f"PASS: models={len(model_paths)} rows={len(sample):,} output={output}")
    print(f"prediction_mean={pred.mean():.6f} min={pred.min():.6f} max={pred.max():.6f}")


if __name__ == "__main__":
    main()
