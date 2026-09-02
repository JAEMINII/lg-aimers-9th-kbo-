#!/usr/bin/env python3
"""Train all/Futures/Regular TabM branches and fine-tune each on 2024."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train_chan_3 import preprocess
from train_chan_3.official_tabm import (
    OfficialTabMEstimator,
    TabMConfig,
    TabularPreprocessor,
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "open" / "data")
    parser.add_argument("--config", type=Path, default=HERE / "selected_config.json")
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts_conditional")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--stage1-epochs", type=int, default=0)
    parser.add_argument("--stage2-epochs", type=int, default=1)
    parser.add_argument("--stage2-lr", type=float, default=0.0002)
    parser.add_argument(
        "--fine-tune-scope", choices=["head", "last_block"], default="head"
    )
    return parser.parse_args()


def fine_tune(model, X, y, *, epochs: int, learning_rate: float, scope: str, seed: int):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if epochs <= 0 or len(X) == 0:
        return []
    nums, cats = model.preprocessor.transform(X)
    target = np.asarray(y, dtype="float32").reshape(-1)
    dataset = TensorDataset(
        torch.from_numpy(nums), torch.from_numpy(cats), torch.from_numpy(target)
    )
    generator = torch.Generator().manual_seed(seed + 3000)
    loader = DataLoader(
        dataset,
        batch_size=model.config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=model.device.type == "cuda",
        generator=generator,
    )

    for parameter in model.model.parameters():
        parameter.requires_grad_(False)
    trainable = []
    if model.model.output is None:
        raise RuntimeError("TabM output layer is missing")
    trainable.extend(model.model.output.parameters())
    if scope == "last_block":
        trainable.extend(model.model.backbone.blocks[-1][0].parameters())
    for parameter in trainable:
        parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=model.config.weight_decay
    )

    history = []
    model.model.train()
    for epoch in range(epochs):
        started = time.time()
        objective_sum = brier_sum = 0.0
        seen = 0
        for bn, bc, by in loader:
            bn = bn.to(model.device, non_blocking=True)
            bc = bc.to(model.device, non_blocking=True) if bc.shape[1] else None
            by = by.to(model.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model.model(bn, bc)
            loss = model._loss(logits, by)
            loss.backward()
            if model.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable, model.config.gradient_clip)
            optimizer.step()
            n = len(by)
            objective_sum += float(loss.detach()) * n
            brier_sum += float((logits.detach().sigmoid().mean(dim=1).squeeze(-1) - by).square().sum())
            seen += n
        row = {
            "stage": "stage2",
            "epoch": epoch + 1,
            "train_objective": objective_sum / seen,
            "train_brier": brier_sum / seen,
            "seconds": time.time() - started,
            "scope": scope,
        }
        history.append(row)
        print(
            f"Stage2 | scope={scope} | epoch={epoch + 1:02d}/{epochs:02d} | "
            f"objective={row['train_objective']:.6f} | "
            f"train_brier={row['train_brier']:.6f} | "
            f"{row['seconds']:.1f}s",
            flush=True,
        )
    model.model.eval()
    return history


def main():
    args = parse_args()
    selected = read_json(args.config)
    if selected.get("status") != "validated":
        raise RuntimeError("selected_config.json must be validation-frozen")
    train_path = args.data_dir / "train.csv"
    train = preprocess.sort_by_row_id(pd.read_csv(train_path, encoding="utf-8-sig"))
    history = preprocess.fit_history_tables(train)
    X_all = preprocess.transform_features(train, history, train_mode=True)
    y_all = train[preprocess.TARGET].reset_index(drop=True)
    full_preprocessor = TabularPreprocessor(
        selected["model"]["params"].get("categorical_features", preprocess.TABM_CATEGORICAL_FEATURES),
        selected["model"]["params"].get("add_missing_indicators", True),
    ).fit(X_all)

    seeds = [int(x) for x in args.seeds.split(",") if x.strip()] if args.seeds else selected["seeds"]
    stage1_epochs = args.stage1_epochs or int(selected["full_train_epochs"])
    branch_masks = {
        "all": np.ones(len(train), dtype=bool),
        "futures": train["game_type"].astype(str).to_numpy() == "F",
        "regular": train["game_type"].astype(str).to_numpy() == "R",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "history.json", preprocess.serialize_history(history))
    write_json(args.output_dir / "branch_counts.json", {
        key: int(mask.sum()) for key, mask in branch_masks.items()
    })

    all_reports = []
    for branch, mask in branch_masks.items():
        X_branch = X_all.loc[mask].reset_index(drop=True)
        y_branch = y_all.loc[mask].reset_index(drop=True)
        for seed in seeds:
            stage1_dir = args.output_dir / f"{branch}_stage1"
            stage2_dir = args.output_dir / f"{branch}_stage2"
            params = dict(selected["model"]["params"])
            params.update(
                loss=selected["model"].get("loss", "brier"),
                epochs=stage1_epochs,
                patience=stage1_epochs + 1,
                device=args.device,
            )
            model = OfficialTabMEstimator(TabMConfig.from_dict(params), seed=int(seed))
            checkpoint = stage1_dir / f"checkpoint_seed_{seed}.pt"
            print(
                f"\nSTAGE1 | branch={branch} | rows={len(X_branch):,} | "
                f"seed={seed} | epochs={stage1_epochs} | loss={params['loss']}",
                flush=True,
            )
            model.fit(
                X_branch,
                y_branch,
                checkpoint_path=checkpoint,
                verbose=True,
                preprocessor=copy.deepcopy(full_preprocessor),
            )
            stage1_dir.mkdir(parents=True, exist_ok=True)
            model.save(stage1_dir / f"tabm_seed_{seed}.pt")

            tuned = OfficialTabMEstimator.load(
                stage1_dir / f"tabm_seed_{seed}.pt", device=args.device
            )
            recent = train["season"].to_numpy() == train["season"].max()
            recent = recent & mask
            X_recent = X_all.loc[recent].reset_index(drop=True)
            y_recent = y_all.loc[recent].reset_index(drop=True)
            print(
                f"STAGE2 | branch={branch} | rows={len(X_recent):,} | "
                f"scope={args.fine_tune_scope} | epochs={args.stage2_epochs}",
                flush=True,
            )
            stage2_history = fine_tune(
                tuned, X_recent, y_recent,
                epochs=args.stage2_epochs,
                learning_rate=args.stage2_lr,
                scope=args.fine_tune_scope,
                seed=int(seed),
            )
            stage2_dir.mkdir(parents=True, exist_ok=True)
            tuned.save(stage2_dir / f"tabm_seed_{seed}.pt")
            pd.DataFrame([
                {"branch": branch, "stage": "stage1", "seed": seed, **row}
                for row in model.history
            ] + [
                {"branch": branch, "seed": seed, **row} for row in stage2_history
            ]).to_csv(stage2_dir / "training_history.csv", index=False)
            all_reports.append({
                "branch": branch,
                "seed": int(seed),
                "stage1_rows": int(mask.sum()),
                "stage2_rows": int(recent.sum()),
                "stage1_epochs": stage1_epochs,
                "stage2_epochs": args.stage2_epochs,
                "fine_tune_scope": args.fine_tune_scope,
                "loss": params["loss"],
            })
    write_json(args.output_dir / "training_report.json", {
        "status": "PASS",
        "branches": all_reports,
        "blend": {"all": 0.6, "specialized": 0.4},
        "specialized_by_game_type": {"F": "futures", "R": "regular"},
    })
    print(json.dumps(read_json(args.output_dir / "training_report.json"), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
