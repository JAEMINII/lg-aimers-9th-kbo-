# -*- coding: utf-8 -*-
"""Thin, tested trainer around the official ``tabm`` package.

The k predictions are optimized independently (mean member loss), while
inference averages member probabilities, exactly as the official guide asks.
"""

from __future__ import annotations

import copy
import json
import random
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .preprocess import TABM_CATEGORICAL_FEATURES


@dataclass
class TabMConfig:
    loss: str = "brier"
    bce_weight: float = 0.5
    arch_type: str = "tabm"
    k: int = 32
    n_blocks: int = 3
    d_block: int = 256
    dropout: float = 0.1
    num_embeddings: str = "linear_relu"
    d_embedding: int = 16
    n_bins: int = 48
    n_frequencies: int = 48
    frequency_init_scale: float = 0.01
    learning_rate: float = 0.002
    weight_decay: float = 0.0003
    batch_size: int = 2048
    eval_batch_size: int = 8192
    epochs: int = 20
    patience: int = 5
    min_delta: float = 1e-7
    scheduler: str = "cosine"
    gradient_clip: float = 5.0
    num_workers: int = 0
    device: str = "auto"
    add_missing_indicators: bool = True
    categorical_features: tuple[str, ...] = tuple(TABM_CATEGORICAL_FEATURES)

    def __post_init__(self):
        if self.loss not in {"brier", "bce", "bce_brier"}:
            raise ValueError("loss must be brier, bce, or bce_brier")
        if self.num_embeddings not in {"none", "linear_relu", "piecewise", "periodic"}:
            raise ValueError("unsupported num_embeddings")
        if self.scheduler not in {"none", "cosine"}:
            raise ValueError("scheduler must be none or cosine")
        if self.arch_type not in {"tabm", "tabm-mini", "tabm-packed"}:
            raise ValueError("unsupported arch_type")
        if not 0.0 <= self.bce_weight <= 1.0:
            raise ValueError("bce_weight must be in [0, 1]")
        for name in ("k", "n_blocks", "d_block", "batch_size", "epochs"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")

    @classmethod
    def from_dict(cls, payload: dict):
        valid = {field.name for field in fields(cls)}
        values = {key: value for key, value in payload.items() if key in valid}
        if "categorical_features" in values:
            values["categorical_features"] = tuple(values["categorical_features"])
        return cls(**values)

    def to_dict(self):
        payload = asdict(self)
        payload["categorical_features"] = list(self.categorical_features)
        return payload


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TabularPreprocessor:
    """Fold-fitted numeric scaler and unknown-safe categorical encoder."""

    def __init__(self, categorical_features, add_missing_indicators=True):
        self.requested_cat_cols = list(categorical_features)
        self.add_missing_indicators = bool(add_missing_indicators)
        self.feature_names: list[str] = []
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.cat_values: dict[str, list] = {}
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.missing_cols: list[str] = []

    def fit(self, X: pd.DataFrame):
        self.feature_names = list(X.columns)
        self.cat_cols = [col for col in self.requested_cat_cols if col in X.columns]
        self.num_cols = [col for col in X.columns if col not in self.cat_cols]
        for col in self.cat_cols:
            values = pd.Series(X[col].dropna().unique()).sort_values().tolist()
            self.cat_values[col] = [v.item() if isinstance(v, np.generic) else v for v in values]
        numeric = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        self.missing_cols = [col for col in self.num_cols if numeric[col].isna().any()]
        for col in self.num_cols:
            series = numeric[col].astype("float64")
            median = float(series.median()) if series.notna().any() else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            self.medians[col] = median
            self.means[col] = mean
            self.stds[col] = std if np.isfinite(std) and std > 1e-8 else 1.0
        return self

    def transform(self, X: pd.DataFrame):
        if list(X.columns) != self.feature_names:
            missing = [col for col in self.feature_names if col not in X]
            extra = [col for col in X if col not in self.feature_names]
            if missing or extra:
                raise ValueError(f"feature mismatch: missing={missing}, extra={extra}")
            X = X[self.feature_names]
        numeric_parts = []
        numeric = X[self.num_cols].apply(pd.to_numeric, errors="coerce")
        for col in self.num_cols:
            values = numeric[col].astype("float64")
            filled = values.fillna(self.medians[col])
            numeric_parts.append(((filled - self.means[col]) / self.stds[col]).to_numpy())
        if self.add_missing_indicators:
            for col in self.missing_cols:
                numeric_parts.append(numeric[col].isna().to_numpy(dtype="float64"))
        nums = (
            np.column_stack(numeric_parts).astype("float32", copy=False)
            if numeric_parts else np.empty((len(X), 0), dtype="float32")
        )
        cat_parts = []
        for col in self.cat_cols:
            mapping = {value: index + 1 for index, value in enumerate(self.cat_values[col])}
            cat_parts.append(X[col].map(mapping).fillna(0).to_numpy(dtype="int64"))
        cats = (
            np.column_stack(cat_parts).astype("int64", copy=False)
            if cat_parts else np.empty((len(X), 0), dtype="int64")
        )
        return nums, cats

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    @property
    def n_num_features(self):
        return len(self.num_cols) + (
            len(self.missing_cols) if self.add_missing_indicators else 0
        )

    @property
    def cat_cardinalities(self):
        return [len(self.cat_values[col]) + 1 for col in self.cat_cols]

    def to_dict(self):
        return {
            "requested_cat_cols": self.requested_cat_cols,
            "add_missing_indicators": self.add_missing_indicators,
            "feature_names": self.feature_names,
            "cat_cols": self.cat_cols, "num_cols": self.num_cols,
            "cat_values": self.cat_values, "medians": self.medians,
            "means": self.means, "stds": self.stds, "missing_cols": self.missing_cols,
        }

    @classmethod
    def from_dict(cls, payload):
        obj = cls(payload["requested_cat_cols"], payload["add_missing_indicators"])
        for key in (
            "feature_names", "cat_cols", "num_cols", "cat_values", "medians",
            "means", "stds", "missing_cols",
        ):
            setattr(obj, key, payload[key])
        return obj


class OfficialTabMEstimator:
    def __init__(self, config: TabMConfig, seed: int = 42):
        self.config = config
        self.seed = int(seed)
        self.preprocessor: Optional[TabularPreprocessor] = None
        self.model = None
        self.device = None
        self.history: list[dict] = []
        self.best_epoch_: Optional[int] = None
        self.best_val_brier_: Optional[float] = None
        self.embedding_bins = None

    def _make_embeddings(self):
        if self.config.num_embeddings == "none" or not self.preprocessor.n_num_features:
            return None
        import rtdl_num_embeddings as rne

        n = self.preprocessor.n_num_features
        if self.config.num_embeddings == "linear_relu":
            return rne.LinearReLUEmbeddings(n, d_embedding=self.config.d_embedding)
        if self.config.num_embeddings == "periodic":
            return rne.PeriodicEmbeddings(
                n, d_embedding=self.config.d_embedding,
                n_frequencies=self.config.n_frequencies,
                frequency_init_scale=self.config.frequency_init_scale,
                activation=True, lite=True,
            )
        if self.embedding_bins is None:
            raise RuntimeError("piecewise embeddings require fitted bins")
        return rne.PiecewiseLinearEmbeddings(
            self.embedding_bins, d_embedding=self.config.d_embedding,
            activation=True, version="B",
        )

    def _build_model(self):
        from tabm import TabM

        cat_cardinalities = self.preprocessor.cat_cardinalities
        self.model = TabM.make(
            n_num_features=self.preprocessor.n_num_features,
            cat_cardinalities=cat_cardinalities or None,
            d_out=1,
            num_embeddings=self._make_embeddings(),
            arch_type=self.config.arch_type,
            k=self.config.k,
            n_blocks=self.config.n_blocks,
            d_block=self.config.d_block,
            dropout=self.config.dropout,
        ).to(self.device)

    def _loss(self, logits, target):
        import torch.nn.functional as F

        target = target[:, None, None].expand_as(logits)
        brier = (logits.sigmoid() - target).square().mean()
        if self.config.loss == "brier":
            return brier
        bce = F.binary_cross_entropy_with_logits(logits, target)
        if self.config.loss == "bce":
            return bce
        return self.config.bce_weight * bce + (1.0 - self.config.bce_weight) * brier

    def _predict_tensors(self, nums, cats, batch_size):
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.from_numpy(nums), torch.from_numpy(cats))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        chunks = []
        self.model.eval()
        with torch.inference_mode():
            for bn, bc in loader:
                logits = self.model(
                    bn.to(self.device, non_blocking=True),
                    bc.to(self.device, non_blocking=True) if bc.shape[1] else None,
                )
                chunks.append(logits.sigmoid().mean(dim=1).squeeze(-1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty(0, dtype="float32")

    def fit(
        self, X: pd.DataFrame, y, X_val: Optional[pd.DataFrame] = None,
        y_val=None, *, checkpoint_path: Optional[Path] = None, resume: bool = False,
        verbose: bool = True, preprocessor: Optional[TabularPreprocessor] = None,
    ):
        import torch
        import rtdl_num_embeddings as rne
        from torch.utils.data import DataLoader, TensorDataset

        seed_everything(self.seed)
        self.device = resolve_device(self.config.device)
        if preprocessor is None:
            self.preprocessor = TabularPreprocessor(
                self.config.categorical_features, self.config.add_missing_indicators
            )
            nums, cats = self.preprocessor.fit_transform(X)
        else:
            self.preprocessor = copy.deepcopy(preprocessor)
            nums, cats = self.preprocessor.transform(X)
        target = np.asarray(y, dtype="float32").reshape(-1)
        if len(target) != len(X):
            raise ValueError("X and y length mismatch")
        if self.config.num_embeddings == "piecewise":
            self.embedding_bins = rne.compute_bins(
                torch.from_numpy(nums), n_bins=self.config.n_bins
            )
        self._build_model()
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
            if self.config.scheduler == "cosine" else None
        )
        start_epoch = 0
        if resume and checkpoint_path and checkpoint_path.exists():
            checkpoint = _safe_torch_load(checkpoint_path, self.device)
            saved_config = checkpoint.get("config")
            if saved_config is not None and saved_config != self.config.to_dict():
                raise RuntimeError(
                    "checkpoint config differs from the current selected config; "
                    "remove the checkpoint or restore the original config"
                )
            saved_preprocessor = checkpoint.get("preprocessor")
            if saved_preprocessor is not None and saved_preprocessor != self.preprocessor.to_dict():
                raise RuntimeError("checkpoint was fitted on a different feature preprocessing state")
            self.model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            if scheduler is not None and checkpoint.get("scheduler_state"):
                scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.history = list(checkpoint.get("history", []))
            start_epoch = int(checkpoint["epoch"])
            if verbose:
                print(f"Resumed from epoch {start_epoch}: {checkpoint_path}", flush=True)

        val_arrays = None
        if X_val is not None and y_val is not None:
            val_arrays = (*self.preprocessor.transform(X_val), np.asarray(y_val, dtype="float32"))
        dataset = TensorDataset(
            torch.from_numpy(nums), torch.from_numpy(cats), torch.from_numpy(target)
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = DataLoader(
            dataset, batch_size=self.config.batch_size, shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda", generator=generator,
        )
        best_state, best_brier, bad_epochs = None, float("inf"), 0
        self.history = self.history[:start_epoch]
        for epoch in range(start_epoch, self.config.epochs):
            started = time.time()
            self.model.train()
            objective_sum = brier_sum = 0.0
            seen = 0
            for bn, bc, by in loader:
                bn = bn.to(self.device, non_blocking=True)
                bc_device = bc.to(self.device, non_blocking=True) if bc.shape[1] else None
                by = by.to(self.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(bn, bc_device)
                loss = self._loss(logits, by)
                loss.backward()
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                optimizer.step()
                batch_n = len(by)
                objective_sum += float(loss.detach()) * batch_n
                mean_prob = logits.detach().sigmoid().mean(dim=1).squeeze(-1)
                brier_sum += float((mean_prob - by).square().sum())
                seen += batch_n
            train_objective, train_brier = objective_sum / seen, brier_sum / seen
            val_brier = None
            if val_arrays is not None:
                vn, vc, vy = val_arrays
                vp = self._predict_tensors(vn, vc, self.config.eval_batch_size)
                val_brier = float(np.mean((vp - vy) ** 2))
            current_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                scheduler.step()
            row = {
                "epoch": epoch + 1, "train_objective": train_objective,
                "train_brier": train_brier, "val_brier": val_brier,
                "learning_rate": current_lr, "seconds": time.time() - started,
            }
            self.history.append(row)
            if verbose:
                val_text = "-" if val_brier is None else f"{val_brier:.6f}"
                print(
                    f"Epoch {epoch + 1:03d}/{self.config.epochs:03d} | "
                    f"objective={train_objective:.6f} | train_brier={train_brier:.6f} | "
                    f"val_brier={val_text} | lr={current_lr:.7f} | "
                    f"{row['seconds']:.1f}s", flush=True,
                )
            monitor = val_brier if val_brier is not None else train_brier
            if monitor < best_brier - self.config.min_delta:
                best_brier, bad_epochs = monitor, 0
                self.best_epoch_ = epoch + 1
                self.best_val_brier_ = val_brier
                if val_arrays is not None:
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                bad_epochs += 1
            if checkpoint_path is not None:
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "epoch": epoch + 1, "model_state": _cpu_state(self.model.state_dict()),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict() if scheduler else None,
                    "history": self.history, "config": self.config.to_dict(),
                    "preprocessor": self.preprocessor.to_dict(),
                }, checkpoint_path)
            if val_arrays is not None and bad_epochs >= self.config.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch + 1}; best={self.best_epoch_}", flush=True)
                break
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    def predict(self, X: pd.DataFrame):
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("model is not fitted")
        nums, cats = self.preprocessor.transform(X)
        return np.clip(
            self._predict_tensors(nums, cats, self.config.eval_batch_size), 0.0, 1.0
        )

    def save(self, path: Path):
        import torch

        if self.model is None:
            raise RuntimeError("model is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format_version": 1, "seed": self.seed, "config": self.config.to_dict(),
            "preprocessor": self.preprocessor.to_dict(),
            "embedding_bins": [x.cpu() for x in self.embedding_bins]
            if self.embedding_bins is not None else None,
            "model_state": _cpu_state(self.model.state_dict()), "history": self.history,
            "best_epoch": self.best_epoch_, "best_val_brier": self.best_val_brier_,
        }, path)

    @classmethod
    def load(cls, path: Path, device: str = "auto"):
        import torch

        resolved = resolve_device(device)
        payload = _safe_torch_load(Path(path), resolved)
        config = TabMConfig.from_dict(payload["config"])
        config.device = str(resolved)
        obj = cls(config, int(payload["seed"]))
        obj.device = resolved
        obj.preprocessor = TabularPreprocessor.from_dict(payload["preprocessor"])
        obj.embedding_bins = payload.get("embedding_bins")
        obj._build_model()
        obj.model.load_state_dict(payload["model_state"])
        obj.model.eval()
        obj.history = list(payload.get("history", []))
        obj.best_epoch_ = payload.get("best_epoch")
        obj.best_val_brier_ = payload.get("best_val_brier")
        return obj


def _cpu_state(state):
    return {key: value.detach().cpu() for key, value in state.items()}


def _safe_torch_load(path, device):
    import torch

    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # torch < 2.6
        return torch.load(path, map_location=device)
