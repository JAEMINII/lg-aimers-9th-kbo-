#!/usr/bin/env python3
"""Export official TabM checkpoints into a submit_jaemin_5-style ZIP bundle."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "open" / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "submit_tabm_conditional")
    parser.add_argument("--zip", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def tensor_np(value):
    return value.detach().cpu().numpy().astype(np.float32, copy=False)


def export_one(checkpoint_path: Path, output_path: Path):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = payload["config"]
    prep = payload["preprocessor"]
    state = payload["model_state"]
    k = int(config["k"])
    arch = str(config["arch_type"])
    if arch != "tabm":
        raise ValueError(f"this exporter currently targets arch_type=tabm, got {arch}")
    prefix = "backbone.blocks.0.0."
    if prefix + "r" not in state:
        raise ValueError("expected official TabM BatchEnsemble parameters were not found")

    arrays = {}
    cat_cols = list(prep["cat_cols"])
    cat_cardinalities = [len(prep["cat_values"][col]) + 1 for col in cat_cols]
    metadata = {
        "format_version": 1,
        "feature_names": list(prep["feature_names"]),
        "cat_cols": cat_cols,
        "num_cols": list(prep["num_cols"]),
        "cat_values": prep["cat_values"],
        "medians": prep["medians"],
        "means": prep["means"],
        "stds": prep["stds"],
        "missing_cols": list(prep["missing_cols"]),
        "cat_cardinalities": cat_cardinalities,
        "num_embeddings": config["num_embeddings"],
        "k": k,
        "n_blocks": int(config["n_blocks"]),
        "d_block": int(config["d_block"]),
        "backbone_kind": "batch_ensemble",
        "loss": config["loss"],
    }

    if config["num_embeddings"] == "linear_relu":
        arrays["num_embedding_weight"] = tensor_np(state["num_module.linear.weight"])
        arrays["num_embedding_bias"] = tensor_np(state["num_module.linear.bias"])
    elif config["num_embeddings"] != "none":
        raise ValueError(f"unsupported num_embeddings={config['num_embeddings']}")

    for block in range(int(config["n_blocks"])):
        block_prefix = f"backbone.blocks.{block}.0."
        arrays[f"block_{block}_weight"] = tensor_np(state[block_prefix + "weight"])
        arrays[f"block_{block}_r"] = tensor_np(state[block_prefix + "r"])
        arrays[f"block_{block}_s"] = tensor_np(state[block_prefix + "s"])
        arrays[f"block_{block}_bias"] = tensor_np(state[block_prefix + "bias"])
    arrays["output_weight"] = tensor_np(state["output.weight"])
    arrays["output_bias"] = tensor_np(state["output.bias"])
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return metadata


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


def write_requirements(path: Path):
    path.write_text("numpy>=1.24\npandas>=2.0\n", encoding="utf-8")


def make_zip(bundle_dir: Path, zip_path: Path):
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(bundle_dir.rglob("*")):
            if (
                path.is_file()
                and "__pycache__" not in path.parts
                and path.name != "export_report.json"
            ):
                archive.write(path, path.relative_to(bundle_dir).as_posix())


def main():
    args = parse_args()
    artifacts = args.artifacts_dir
    branches = {"all": "all_stage2", "futures": "futures_stage2", "regular": "regular_stage2"}
    checkpoints = {
        branch: artifacts / stage / "tabm_seed_42.pt"
        for branch, stage in branches.items()
    }
    missing = [str(path) for path in checkpoints.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Stage2 checkpoints: {missing}")
    bundle = args.output_dir
    model_dir = bundle / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HERE / "submission_script.py", bundle / "script.py")
    shutil.copy2(HERE / "preprocess.py", bundle / "preprocess.py")
    write_requirements(bundle / "requirements.txt")
    history_path = artifacts / "history.json"
    if history_path.exists():
        shutil.copy2(history_path, model_dir / "history.json")
    else:
        from train_chan_3 import preprocess

        train = pd.read_csv(args.data_dir / "train.csv", encoding="utf-8-sig")
        history = preprocess.fit_history_tables(train)
        (model_dir / "history.json").write_text(
            json.dumps(preprocess.serialize_history(history), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"Generated train-derived history: {model_dir / 'history.json'}")

    metadata = None
    exported_paths = {}
    for branch, checkpoint in checkpoints.items():
        metadata = export_one(checkpoint, model_dir / f"{branch}_tabm_seed_42.npz")
        exported_paths[branch] = model_dir / f"{branch}_tabm_seed_42.npz"
    config = {
        "model": "conditional official TabM exported to NumPy",
        "loss": metadata["loss"],
        "branches": list(checkpoints),
        "blend": {"all": 0.6, "specialized": 0.4},
        "specialized_by_game_type": {"F": "futures", "R": "regular"},
        "features": metadata["feature_names"],
        "submission_format": "script.py + preprocess.py + requirements.txt + model/*.npz",
        "test_row_independent": True,
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Check the exported package against the original PyTorch checkpoint.
    from train_chan_3 import preprocess
    from train_chan_3.official_tabm import OfficialTabMEstimator

    test = pd.read_csv(args.data_dir / "test.csv", encoding="utf-8-sig")
    history = preprocess.deserialize_history(
        json.loads((model_dir / "history.json").read_text(encoding="utf-8"))
    )
    ordered = preprocess.sort_by_row_id(test)
    X = preprocess.build_inference_features(ordered, history)
    submission_script = load_module("exported_submission_script", bundle / "script.py")
    parity_rows = {}
    for branch, checkpoint in checkpoints.items():
        estimator = OfficialTabMEstimator.load(checkpoint, device=args.device)
        mask = np.ones(len(ordered), dtype=bool) if branch == "all" else (
            ordered["game_type"].astype(str).to_numpy() == ("F" if branch == "futures" else "R")
        )
        if not mask.any():
            parity_rows[branch] = 0.0
            continue
        branch_x = X if branch == "all" else X.iloc[np.flatnonzero(mask)]
        pytorch = estimator.predict(branch_x)
        meta, archive = submission_script.load_bundle(
            str(exported_paths[branch])
        )
        exported = submission_script.predict_frame(branch_x, meta, archive)
        archive.close()
        parity_rows[branch] = float(np.max(np.abs(pytorch - exported)))
        if parity_rows[branch] > 1e-5:
            raise RuntimeError(f"PyTorch/NumPy parity failed for {branch}: {parity_rows[branch]}")

    report = {
        "status": "PASS",
        "rows_checked": len(X),
        "pytorch_numpy_max_abs": parity_rows,
        "submission_ready": True,
    }
    (bundle / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    zip_path = args.zip or bundle.with_suffix(".zip")
    make_zip(bundle, zip_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"BUNDLE: {bundle}")
    print(f"ZIP: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
