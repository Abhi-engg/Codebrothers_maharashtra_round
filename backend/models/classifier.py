"""
Re:Learn — Misconception Classifier Models
Author: Senior ML & DL Training Engineer

This module provides modular classifier definitions for Model 1 (Individual Misconception Diagnosis):
1. MisconceptionBaselineClassifier:
   - High-speed TF-IDF + Logistic Regression / CalibratedClassifierCV baseline.
   - Computes calibrated class probabilities and macro metrics.
   - Runs fast on both CPU and GPU.
2. DebertaMisconceptionClassifier:
   - Pretrained DeBERTa-v3 architecture with the '@[Quote]' Partial Training strategy.
   - Partial Training: Freezes embeddings (100%) and bottom 6 encoder layers (0-5).
     Fine-tunes top 6 encoder layers (6-11) and classification head.
   - Focal Loss (and class-weighted Cross-Entropy) to tackle severe class imbalance.
   - Calibrated softmax distribution over all 17 taxonomy classes.
3. Feature Formatting & Taxonomy Utilities:
   - Standardized input concatenation: Question Text + Expected Physics Summary + Student Final Answer + Student Working Steps.
   - Taxonomy loading and class ID mapping.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Setup logger
logger = logging.getLogger("relearn.classifier")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Optional scikit-learn imports
try:
    import joblib
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn or joblib is not installed. Baseline classifier will be disabled.")

# Optional PyTorch & Transformers imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch is not installed. DeBERTa classifier and FocalLoss will be disabled.")

try:
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers is not installed. DeBERTa classifier will be disabled.")


# ============================================================================
# 1. Feature Formatting and Taxonomy Utilities
# ============================================================================

def extract_record_label(record: Optional[Dict[str, Any]], default: str = "UNCERTAIN-GUESS") -> str:
    """
    Safely extract the primary ground truth label from a dataset record.
    Gracefully handles None, missing keys, or empty structures.
    """
    if not record or not isinstance(record, dict):
        return default
    gt = record.get("ground_truth")
    if not isinstance(gt, dict):
        return default
    lbl = gt.get("primary_label")
    if lbl is None or not str(lbl).strip():
        return default
    return str(lbl).strip()


def format_misconception_input(
    record: Optional[Dict[str, Any]] = None,
    *,
    question_text: Optional[str] = None,
    expected_summary: Optional[str] = None,
    final_answer: Optional[str] = None,
    working_steps: Optional[str] = None,
) -> str:
    """
    Concatenate Question Text, Expected Physics Summary, Student Final Answer,
    and Student Working Steps into a unified diagnostic representation.

    Can be called either with a split record dictionary or explicit keyword arguments.
    Gracefully handles None values for nested dictionaries or string fields.
    """
    if record is not None and isinstance(record, dict):
        q_obj = record.get("question") or {}
        sub_obj = record.get("student_submission") or {}

        q_text = str(q_obj.get("question_text", "") or "").strip()
        summary = str(q_obj.get("expected_physics_summary", "") or q_obj.get("expected_summary", "") or "").strip()
        ans = str(sub_obj.get("final_answer", "") or sub_obj.get("student_answer", "") or "").strip()
        steps = str(sub_obj.get("working_steps", "") or sub_obj.get("student_working", "") or "").strip()
    else:
        q_text = str(question_text or "").strip()
        summary = str(expected_summary or "").strip()
        ans = str(final_answer or "").strip()
        steps = str(working_steps or "").strip()

    return (
        f"Question: {q_text}\n"
        f"Expected Physics Summary: {summary}\n"
        f"Student Final Answer: {ans}\n"
        f"Student Working Steps: {steps}"
    )


def load_taxonomy_labels(taxonomy_path: Union[str, Path]) -> List[str]:
    """
    Extract the canonical ordered list of all target classes from data/taxonomy.json.
    Includes misconception IDs across all domains and baseline classes.
    """
    path = Path(taxonomy_path)
    if not path.exists():
        raise FileNotFoundError(f"Taxonomy file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    labels: List[str] = []
    # Collect misconceptions in domains
    for domain in data.get("domains", []):
        for concept in domain.get("concepts", []):
            for misc in concept.get("misconceptions", []):
                misc_id = misc["misconception_id"]
                if misc_id not in labels:
                    labels.append(misc_id)

    # Collect baseline classes
    for base in data.get("baseline_classes", []):
        base_id = base["class_id"]
        if base_id not in labels:
            labels.append(base_id)

    return labels


# ============================================================================
# 2. Focal Loss & Class Weighting for PyTorch
# ============================================================================

if TORCH_AVAILABLE:
    class FocalLoss(nn.Module):
        """
        Multi-class Focal Loss implementation:
            FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
        
        Args:
            gamma (float): Focusing parameter. Higher gamma reduces the relative loss
                           for well-classified examples (p_t > 0.5), focusing on hard ones.
            weight (torch.Tensor, optional): Pre-computed per-class weighting factors alpha.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        def __init__(
            self,
            gamma: float = 2.0,
            weight: Optional[torch.Tensor] = None,
            reduction: str = "mean",
        ):
            super().__init__()
            self.gamma = float(gamma)
            if weight is not None:
                self.register_buffer("weight", weight.clone().detach().float())
            else:
                self.weight = None
            self.reduction = reduction

        def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            """
            Args:
                logits: FloatTensor of shape (batch_size, num_classes)
                targets: LongTensor of shape (batch_size)
            """
            log_p = F.log_softmax(logits, dim=-1)
            target_log_p = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
            target_p = target_log_p.exp()

            # Clamp (1 - p) to [0, 1] to avoid negative base NaN under float precision
            focal_factor = (1.0 - target_p).clamp(min=0.0, max=1.0) ** self.gamma
            loss = -focal_factor * target_log_p

            if self.weight is not None:
                # Ensure weight tensor is on same device
                if self.weight.device != targets.device:
                    self.weight = self.weight.to(targets.device)
                alpha_t = self.weight[targets]
                loss = alpha_t * loss

            if self.reduction == "mean":
                if self.weight is not None:
                    return loss.sum() / alpha_t.sum().clamp(min=1e-8)
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            elif self.reduction == "none":
                return loss
            else:
                raise ValueError(f"Unsupported reduction: {self.reduction}")

    def compute_balanced_class_weights(
        labels: Sequence[int],
        num_classes: int,
        device: Union[str, torch.device] = "cpu",
    ) -> torch.Tensor:
        """
        Compute inverse frequency balanced class weights:
            w_c = N / (C * count_c)
        Normalized so that the mean class weight is 1.0.
        """
        if num_classes <= 0:
            raise ValueError(f"num_classes must be positive, got {num_classes}")

        label_arr = np.array(labels, dtype=np.int64)
        total_samples = len(label_arr)
        if total_samples == 0:
            return torch.ones(num_classes, dtype=torch.float32, device=device)

        valid_labels = label_arr[(label_arr >= 0) & (label_arr < num_classes)]
        counts = np.bincount(valid_labels, minlength=num_classes)[:num_classes]
        
        weights = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = total_samples / (num_classes * counts[c])
            else:
                weights[c] = 1.0
        
        # Normalize weights so mean is 1.0
        weights_mean = float(np.mean(weights))
        if weights_mean > 0:
            weights = weights / weights_mean
        return torch.tensor(weights, dtype=torch.float32, device=device)
else:
    class FocalLoss:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required to use FocalLoss.")

    def compute_balanced_class_weights(*args, **kwargs):  # type: ignore
        raise ImportError("PyTorch is required to compute class weights.")


# ============================================================================
# 3. Model 1 Baseline Classifier (TF-IDF + Calibrated Logistic Regression)
# ============================================================================

class MisconceptionBaselineClassifier:
    """
    High-speed scikit-learn baseline for Model 1 Misconception Diagnosis.
    Extracts TF-IDF n-grams from the concatenated input text and trains a
    class-weighted Logistic Regression model with optional probability calibration
    via CalibratedClassifierCV.
    """
    def __init__(
        self,
        classes: Optional[List[str]] = None,
        ngram_range: Tuple[int, int] = (1, 3),
        max_features: int = 25000,
        sublinear_tf: bool = True,
        calibrate: bool = True,
        calibration_cv: int = 3,
        random_state: int = 42,
    ):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn and joblib are required for MisconceptionBaselineClassifier.")

        self.classes: List[str] = list(classes) if classes is not None else []
        self.label2id: Dict[str, int] = {cls: idx for idx, cls in enumerate(self.classes)}
        self.id2label: Dict[int, str] = {idx: cls for idx, cls in enumerate(self.classes)}

        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.calibrate = calibrate
        self.calibration_cv = calibration_cv
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            strip_accents="unicode",
            lowercase=True,
        )

        base_lr = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        )

        if self.calibrate:
            self.model = CalibratedClassifierCV(
                estimator=base_lr,
                method="sigmoid",
                cv=self.calibration_cv,
            )
        else:
            self.model = base_lr

        self.is_fitted: bool = False

    def _setup_labels(self, y: Sequence[str]) -> None:
        if not self.classes:
            unique_labels = sorted(list(set(y)))
            self.classes = unique_labels
            self.label2id = {cls: idx for idx, cls in enumerate(self.classes)}
            self.id2label = {idx: cls for idx, cls in enumerate(self.classes)}

    def fit(self, texts: Sequence[str], labels: Sequence[str]) -> "MisconceptionBaselineClassifier":
        """
        Fit the TF-IDF vectorizer and calibrated classifier on training texts and labels.
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels, str):
            labels = [labels]
        if len(texts) == 0:
            raise ValueError("Cannot fit Baseline Classifier on empty texts dataset.")

        logger.info(f"Fitting Baseline Classifier on {len(texts)} samples...")
        self._setup_labels(labels)

        # Convert string labels to integer indices, safely falling back to UNCERTAIN-GUESS
        default_idx = self.label2id.get("UNCERTAIN-GUESS", 0)
        y_indices = np.array([self.label2id.get(lbl, default_idx) for lbl in labels], dtype=np.int64)

        # Extract features and fit model
        X_vec = self.vectorizer.fit_transform(texts)
        self.model.fit(X_vec, y_indices)
        self.is_fitted = True
        logger.info(f"Baseline Classifier successfully fitted across {len(self.classes)} classes.")
        return self

    def predict_proba(self, texts: Union[str, Sequence[str]]) -> np.ndarray:
        """
        Predict probability distribution over all classes for input texts.
        Returns:
            np.ndarray of shape (len(texts), len(self.classes))
        """
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba.")
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.zeros((0, len(self.classes)), dtype=np.float32)

        X_vec = self.vectorizer.transform(texts)
        probs = self.model.predict_proba(X_vec)
        
        # Ensure probs align with all self.classes if model.classes_ has fewer classes
        if probs.shape[1] < len(self.classes):
            full_probs = np.zeros((len(texts), len(self.classes)), dtype=np.float32)
            model_classes = self.model.classes_
            for col_idx, class_idx in enumerate(model_classes):
                full_probs[:, class_idx] = probs[:, col_idx]
            return full_probs

        return probs

    def predict(self, texts: Union[str, Sequence[str]]) -> List[str]:
        """
        Predict class label strings for input texts.
        """
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return []

        probs = self.predict_proba(texts)
        pred_indices = np.argmax(probs, axis=1)
        return [self.id2label.get(idx, "UNCERTAIN-GUESS") for idx in pred_indices]

    def evaluate(self, texts: Sequence[str], labels: Sequence[str]) -> Dict[str, Any]:
        """
        Evaluate model performance on given texts and true labels.
        Returns dictionary of metrics including Macro-F1, Accuracy, per-class report,
        and confusion matrix.
        """
        if isinstance(texts, str):
            texts = [texts]
        if isinstance(labels, str):
            labels = [labels]

        if len(texts) == 0 or len(labels) == 0:
            return {
                "macro_f1": 0.0,
                "weighted_f1": 0.0,
                "accuracy": 0.0,
                "classification_report_str": "Empty evaluation dataset (0 samples).",
                "classification_report_dict": {},
                "confusion_matrix": [[0] * len(self.classes) for _ in range(len(self.classes))],
                "classes": self.classes,
            }

        preds = self.predict(texts)
        y_true = list(labels)

        # Macro F1 is the primary competition metric over all canonical classes
        macro_f1 = float(f1_score(y_true, preds, labels=self.classes, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, preds, labels=self.classes, average="weighted", zero_division=0))
        accuracy = float(accuracy_score(y_true, preds))

        report_dict = classification_report(
            y_true,
            preds,
            labels=self.classes,
            target_names=self.classes,
            output_dict=True,
            zero_division=0,
        )
        report_str = classification_report(
            y_true,
            preds,
            labels=self.classes,
            target_names=self.classes,
            digits=4,
            zero_division=0,
        )

        cm = confusion_matrix(y_true, preds, labels=self.classes)

        return {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": accuracy,
            "classification_report_str": report_str,
            "classification_report_dict": report_dict,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes,
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """
        Save fitted model, vectorizer, and label metadata to disk.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "classes": self.classes,
            "label2id": self.label2id,
            "id2label": self.id2label,
            "vectorizer": self.vectorizer,
            "model": self.model,
            "ngram_range": self.ngram_range,
            "max_features": self.max_features,
            "sublinear_tf": self.sublinear_tf,
            "calibrate": self.calibrate,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(payload, path)
        logger.info(f"Saved baseline model checkpoint to: {path}")

    @classmethod
    def load(cls, checkpoint_path: Union[str, Path]) -> "MisconceptionBaselineClassifier":
        """
        Load a saved baseline classifier from disk.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {path}")

        payload = joblib.load(path)
        instance = cls(
            classes=payload["classes"],
            ngram_range=payload["ngram_range"],
            max_features=payload["max_features"],
            sublinear_tf=payload["sublinear_tf"],
            calibrate=payload["calibrate"],
            random_state=payload["random_state"],
        )
        instance.label2id = payload["label2id"]
        instance.id2label = payload["id2label"]
        instance.vectorizer = payload["vectorizer"]
        instance.model = payload["model"]
        instance.is_fitted = payload["is_fitted"]
        logger.info(f"Loaded baseline model checkpoint from: {path} with {len(instance.classes)} classes.")
        return instance


# Alias for flexible naming convention
BaselineMisconceptionClassifier = MisconceptionBaselineClassifier


# ============================================================================
# 4. Model 1 DeBERTa-v3 with Partial Training Strategy (@[Quote])
# ============================================================================

if TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE:
    class DebertaMisconceptionClassifier(nn.Module):
        """
        DeBERTa-v3 model with the '@[Quote]' Partial Training strategy:
        - Input: Concatenated Question + Expected Summary + Student Answer + Working Steps
        - Architecture: microsoft/deberta-v3-base with 17 output classes
        - Partial Training:
            - Embeddings 100% frozen
            - Bottom 6 encoder layers (0-5) 100% frozen
            - Top 6 encoder layers (6-11) + pooler + classifier head fine-tuned
        - Loss: Multi-class Focal Loss (or class-weighted Cross-Entropy)
        """
        def __init__(
            self,
            model_name: str = "microsoft/deberta-v3-base",
            classes: Optional[List[str]] = None,
            freeze_embeddings: bool = True,
            freeze_layers_count: int = 6,
            loss_type: str = "focal",
            focal_gamma: float = 2.0,
            class_weights: Optional[torch.Tensor] = None,
            dropout_prob: float = 0.15,
        ):
            super().__init__()
            self.model_name = model_name
            self.classes = list(classes) if classes is not None else []
            self.num_classes = len(self.classes) if self.classes else 17
            self.label2id = {cls: idx for idx, cls in enumerate(self.classes)}
            self.id2label = {idx: cls for idx, cls in enumerate(self.classes)}

            self.freeze_embeddings = freeze_embeddings
            self.freeze_layers_count = freeze_layers_count
            self.loss_type = loss_type
            self.focal_gamma = focal_gamma
            self.class_weights = class_weights

            # Load HuggingFace Config and Base Sequence Classification Model
            config = AutoConfig.from_pretrained(
                self.model_name,
                num_labels=self.num_classes,
                id2label=self.id2label,
                label2id=self.label2id,
                hidden_dropout_prob=dropout_prob,
                attention_probs_dropout_prob=dropout_prob,
            )
            self.encoder_model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                config=config,
            )

            # Ensure model parameters are in full float32 precision
            self.encoder_model.float()

            # Apply Partial Training Strategy: Freeze embeddings and bottom layers
            self.apply_partial_training_freezing(
                freeze_embeddings=self.freeze_embeddings,
                freeze_layers_count=self.freeze_layers_count,
            )

            # Setup Loss Function
            if self.loss_type == "focal":
                self.loss_fn = FocalLoss(gamma=self.focal_gamma, weight=self.class_weights)
            elif self.loss_type == "ce":
                self.loss_fn = nn.CrossEntropyLoss(weight=self.class_weights)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}. Choose 'focal' or 'ce'.")

        def apply_partial_training_freezing(
            self,
            freeze_embeddings: bool = True,
            freeze_layers_count: int = 6,
        ) -> Dict[str, int]:
            """
            Implements the Partial Training Strategy:
            - Freezes all parameters in embeddings
            - Freezes encoder layers 0 to (freeze_layers_count - 1)
            - Keeps layers freeze_layers_count to 11, pooler, and classifier head trainable
            """
            total_params = 0
            frozen_params = 0
            trainable_params = 0

            for name, param in self.encoder_model.named_parameters():
                total_params += param.numel()
                is_frozen = False

                # Freeze embeddings
                if freeze_embeddings and ("embeddings" in name and "rel_embeddings" not in name):
                    is_frozen = True
                
                # Freeze bottom encoder layers
                elif "encoder.layer." in name:
                    try:
                        layer_idx = int(name.split("encoder.layer.")[1].split(".")[0])
                        if layer_idx < freeze_layers_count:
                            is_frozen = True
                    except (IndexError, ValueError):
                        pass

                if is_frozen:
                    param.requires_grad = False
                    frozen_params += param.numel()
                else:
                    param.requires_grad = True
                    trainable_params += param.numel()

            pct_frozen = (frozen_params / total_params * 100) if total_params > 0 else 0.0
            pct_trainable = (trainable_params / total_params * 100) if total_params > 0 else 0.0
            logger.info(
                f"Partial Training applied: {frozen_params:,} params frozen ({pct_frozen:.1f}%), "
                f"{trainable_params:,} params trainable ({pct_trainable:.1f}%)"
            )

            return {
                "total_params": total_params,
                "frozen_params": frozen_params,
                "trainable_params": trainable_params,
            }

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Forward pass through DeBERTa-v3 backbone and classification head.
            Returns dictionary containing 'logits' and optionally 'loss'.
            """
            kwargs: Dict[str, Any] = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                kwargs["token_type_ids"] = token_type_ids

            outputs = self.encoder_model(**kwargs)
            logits = outputs.logits

            result: Dict[str, torch.Tensor] = {"logits": logits}

            if labels is not None:
                loss = self.loss_fn(logits, labels)
                result["loss"] = loss

            return result

        @torch.no_grad()
        def predict_logits(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Run forward pass without gradients and return raw logits.
            """
            was_training = self.training
            self.eval()
            try:
                outputs = self.forward(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                return outputs["logits"]
            finally:
                if was_training:
                    self.train()

        @torch.no_grad()
        def predict_proba(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Run forward pass and return softmax calibrated probabilities across classes.
            """
            logits = self.predict_logits(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            return F.softmax(logits, dim=-1)

        @torch.no_grad()
        def predict(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
        ) -> List[str]:
            """
            Run forward pass and return predicted class string labels.
            """
            probs = self.predict_proba(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pred_indices = torch.argmax(probs, dim=-1).cpu().tolist()
            return [self.id2label.get(idx, "UNCERTAIN-GUESS") for idx in pred_indices]

        def save_pretrained(self, save_directory: Union[str, Path]) -> None:
            """
            Save model weights, configuration, tokenizer mapping, and taxonomy classes.
            """
            save_path = Path(save_directory)
            save_path.mkdir(parents=True, exist_ok=True)

            # Save huggingface model and config
            self.encoder_model.save_pretrained(str(save_path))

            # Save taxonomy and strategy metadata
            meta = {
                "model_name": self.model_name,
                "classes": self.classes,
                "num_classes": self.num_classes,
                "label2id": self.label2id,
                "id2label": self.id2label,
                "freeze_embeddings": self.freeze_embeddings,
                "freeze_layers_count": self.freeze_layers_count,
                "loss_type": self.loss_type,
                "focal_gamma": self.focal_gamma,
            }
            with open(save_path / "model_metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            logger.info(f"DeBERTa model checkpoint successfully saved to: {save_path}")

        @classmethod
        def from_pretrained(
            cls,
            save_directory: Union[str, Path],
            device: Union[str, torch.device] = "cpu",
        ) -> "DebertaMisconceptionClassifier":
            """
            Load fine-tuned DeBERTa misconception classifier from saved directory.
            """
            save_path = Path(save_directory)
            meta_path = save_path / "model_metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing model_metadata.json at: {save_path}")

            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            instance = cls(
                model_name=str(save_path),
                classes=meta["classes"],
                freeze_embeddings=meta.get("freeze_embeddings", True),
                freeze_layers_count=meta.get("freeze_layers_count", 6),
                loss_type=meta.get("loss_type", "focal"),
                focal_gamma=meta.get("focal_gamma", 2.0),
            )
            instance.to(device)
            instance.eval()
            logger.info(f"Loaded DeBERTa model from: {save_path} on device {device}")
            return instance

else:
    class DebertaMisconceptionClassifier:  # type: ignore
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch and Hugging Face Transformers are required for DebertaMisconceptionClassifier.")
