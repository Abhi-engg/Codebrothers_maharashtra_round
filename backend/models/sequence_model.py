"""
Re:Learn Model 2: Sequence-Based Pattern Analyzer
Tracks longitudinal student trajectories (Q1 -> Q2 -> Q3 ...) to detect:
  1. PERSISTENT_MISCONCEPTION: Cognitive traps that recur over multiple attempts.
  2. TRANSIENT_SLIP: One-off arithmetic or unit mistakes that resolve immediately.
  3. RESOLVING_TRAJECTORY: Active learning progress (Error -> Remediation/Uncertain -> Mastery).

Provides:
  - RuleBasedSequenceAnalyzer: Fast, deterministic, interpretable heuristic analyzer.
  - GRUSequenceAnalyzer: 2-layer Bidirectional GRU with attention pooling and multi-task heads.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("relearn.sequence_model")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# Canonical Pattern Classes for Model 2
SEQUENCE_PATTERN_CLASSES: List[str] = [
    "PERSISTENT_MISCONCEPTION",
    "TRANSIENT_SLIP",
    "RESOLVING_TRAJECTORY",
]

# Baseline / non-misconception single-step classes
BASELINE_NON_MISC: set[str] = {
    "CORRECT",
    "SLIP-ARITHMETIC",
    "SLIP-UNIT",
    "UNCERTAIN-GUESS",
}


def load_taxonomy_classes(taxonomy_path: Union[str, Path] = "data/taxonomy.json") -> List[str]:
    """Extract canonical misconception and baseline classes from taxonomy JSON."""
    tax_file = Path(taxonomy_path)
    if not tax_file.exists():
        return [
            "MISC-G09-MOT-01", "MISC-G09-MOT-02", "MISC-G09-GRV-01",
            "MISC-G10-OPT-01", "MISC-G10-OPT-02", "MISC-G10-OPT-03",
            "MISC-G10-ELE-01", "MISC-G10-ELE-02", "MISC-G10-ELE-03",
            "MISC-G11-KIN-01", "MISC-G11-DYN-01",
            "MISC-G12-EST-01", "MISC-G12-EMI-01",
            "CORRECT", "SLIP-ARITHMETIC", "SLIP-UNIT", "UNCERTAIN-GUESS",
        ]
    with open(tax_file, "r", encoding="utf-8") as f:
        tax = json.load(f)
    classes = []
    for domain in tax.get("domains", []):
        for concept in domain.get("concepts", []):
            for m in concept.get("misconceptions", []):
                classes.append(m["misconception_id"])
    for b in tax.get("baseline_classes", []):
        classes.append(b["class_id"])
    return classes


# ==============================================================================
# 1. RULE-BASED SEQUENCE ANALYZER (Deterministic Baseline)
# ==============================================================================

class RuleBasedSequenceAnalyzer:
    """
    Deterministic rule engine that inspects attempt histories to diagnose
    trajectory-level cognitive dynamics.
    """

    def __init__(self, taxonomy_classes: Optional[List[str]] = None) -> None:
        self.taxonomy_classes = taxonomy_classes or load_taxonomy_classes()

    def analyze_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a complete student session log.
        Returns pattern_type, persistent_misconception_id, confidence, summary, and recommendation.
        """
        steps = session.get("steps", [])
        if not steps:
            return {
                "pattern_type": "TRANSIENT_SLIP",
                "persistent_misconception_id": None,
                "confidence": 0.50,
                "evidence_summary": "Empty attempt session.",
                "recommendation": "GATHER_MORE_EVIDENCE",
            }

        diagnoses: List[str] = []
        confidences: List[float] = []
        for step in steps:
            diag = step.get("model1_individual_diagnosis", "UNCERTAIN-GUESS") or "UNCERTAIN-GUESS"
            conf = float(step.get("model1_confidence", 0.8))
            diagnoses.append(diag)
            confidences.append(conf)

        n_steps = len(diagnoses)
        misc_diagnoses = [d for d in diagnoses if d not in BASELINE_NON_MISC]
        misc_counts = Counter(misc_diagnoses)

        # 1. Check for RESOLVING_TRAJECTORY:
        # Initial error(s) followed by sustained CORRECT attempts
        if n_steps >= 3:
            first_half = diagnoses[:n_steps // 2 + 1]
            last_steps = diagnoses[-2:]
            had_prior_error = any(d not in ["CORRECT"] for d in first_half)
            ends_correct = all(d == "CORRECT" for d in last_steps)
            if had_prior_error and ends_correct:
                resolved_id = misc_diagnoses[0] if misc_diagnoses else None
                return {
                    "pattern_type": "RESOLVING_TRAJECTORY",
                    "persistent_misconception_id": resolved_id,
                    "confidence": 0.95,
                    "evidence_summary": (
                        f"Student initially struggled ({first_half[0]}), transitioned through learning, "
                        f"and resolved understanding with consecutive correct attempts ({last_steps})."
                    ),
                    "recommendation": "CONFIRM_MASTERY_WITH_ADVANCED_TRANSFER",
                }

        # 2. Check for PERSISTENT_MISCONCEPTION:
        # Same conceptual misconception appears >= 2 times or in >= 50% of attempts
        if misc_counts:
            most_common_misc, count = misc_counts.most_common(1)[0]
            if count >= 2 or (count >= 1 and n_steps <= 2):
                return {
                    "pattern_type": "PERSISTENT_MISCONCEPTION",
                    "persistent_misconception_id": most_common_misc,
                    "confidence": min(0.99, 0.70 + (count / n_steps) * 0.28),
                    "evidence_summary": (
                        f"Cognitive trap {most_common_misc} recurred in {count}/{n_steps} "
                        f"attempts ({', '.join(diagnoses)})."
                    ),
                    "recommendation": "TRIGGER_TARGETED_PEDAGOGICAL_INTERVENTION",
                }

        # 3. Check for TRANSIENT_SLIP:
        # Isolated slip (e.g. arithmetic/unit error) flanked by correct/uncertain attempts
        slip_count = sum(1 for d in diagnoses if d in ["SLIP-ARITHMETIC", "SLIP-UNIT"])
        if slip_count >= 1 and not misc_counts:
            return {
                "pattern_type": "TRANSIENT_SLIP",
                "persistent_misconception_id": None,
                "confidence": 0.90,
                "evidence_summary": (
                    f"Student experienced an isolated execution slip ({', '.join(diagnoses)}) "
                    f"without recurring conceptual misconception."
                ),
                "recommendation": "ENCOURAGE_CAREFUL_CALCULATION_NO_REMEDIATION",
            }

        # Default fallback:
        return {
            "pattern_type": "TRANSIENT_SLIP",
            "persistent_misconception_id": None,
            "confidence": 0.70,
            "evidence_summary": f"Trajectory showed non-systematic variance: {', '.join(diagnoses)}",
            "recommendation": "MONITOR_NEXT_ATTEMPT",
        }


# ==============================================================================
# 2. NEURAL SEQUENCE MODEL (2-Layer Bidirectional GRU)
# ==============================================================================

if TORCH_AVAILABLE:

    class GRUSequenceNet(nn.Module):
        """
        Bidirectional GRU network mapping sequence of step embeddings to:
          1. Pattern Type logits (3 classes)
          2. Persistent Misconception ID logits (num_taxonomy_classes + 1 for 'NONE')
        """

        def __init__(
            self,
            num_diagnosis_classes: int,
            embedding_dim: int = 32,
            hidden_dim: int = 64,
            num_layers: int = 2,
            dropout: float = 0.1,
            num_pattern_classes: int = len(SEQUENCE_PATTERN_CLASSES),
        ) -> None:
            super().__init__()
            self.num_diagnosis_classes = num_diagnosis_classes
            self.num_pattern_classes = num_pattern_classes

            # Diagnosis embedding (+1 for padding / unknown)
            self.diag_embedding = nn.Embedding(
                num_embeddings=num_diagnosis_classes + 1,
                embedding_dim=embedding_dim,
                padding_idx=num_diagnosis_classes,
            )

            # Step feature dimension: embedding_dim + confidence (1) + normalized_step_pos (1)
            in_features = embedding_dim + 2

            self.gru = nn.GRU(
                input_size=in_features,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

            # Attention pooling layer to aggregate bidirectional step outputs
            self.attention_weights = nn.Linear(hidden_dim * 2, 1)

            # Multi-task classification heads
            self.pattern_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_pattern_classes),
            )

            self.misc_classifier = nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_diagnosis_classes + 1),  # Last idx = NONE
            )

        def forward(
            self,
            diag_indices: torch.Tensor,     # (batch, seq_len)
            confidences: torch.Tensor,      # (batch, seq_len)
            step_positions: torch.Tensor,   # (batch, seq_len)
            mask: torch.Tensor,             # (batch, seq_len) bool (True for valid steps)
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            # Embed diagnoses
            diag_emb = self.diag_embedding(diag_indices)  # (batch, seq_len, emb_dim)

            # Concatenate continuous step features
            features = torch.cat(
                [diag_emb, confidences.unsqueeze(-1), step_positions.unsqueeze(-1)],
                dim=-1,
            )  # (batch, seq_len, in_features)

            gru_out, _ = self.gru(features)  # (batch, seq_len, hidden_dim * 2)

            # Masked Attention Pooling
            attn_scores = self.attention_weights(gru_out).squeeze(-1)  # (batch, seq_len)
            attn_scores = attn_scores.masked_fill(~mask, -1e9)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, seq_len, 1)

            pooled = torch.sum(gru_out * attn_weights, dim=1)  # (batch, hidden_dim * 2)

            pattern_logits = self.pattern_classifier(pooled)
            misc_logits = self.misc_classifier(pooled)

            return pattern_logits, misc_logits


class GRUSequenceAnalyzer:
    """
    High-level wrapper around GRUSequenceNet for session batching, training,
    inference, and serialization.
    """

    def __init__(
        self,
        taxonomy_classes: Optional[List[str]] = None,
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        device: Optional[str] = None,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for GRUSequenceAnalyzer.")

        self.taxonomy_classes = taxonomy_classes or load_taxonomy_classes()
        self.diag2idx = {c: i for i, c in enumerate(self.taxonomy_classes)}
        self.idx2diag = {i: c for i, c in enumerate(self.taxonomy_classes)}
        self.padding_idx = len(self.taxonomy_classes)
        self.none_misc_idx = len(self.taxonomy_classes)

        self.pattern_classes = SEQUENCE_PATTERN_CLASSES
        self.pattern2idx = {p: i for i, p in enumerate(self.pattern_classes)}
        self.idx2pattern = {i: p for i, p in enumerate(self.pattern_classes)}

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = GRUSequenceNet(
            num_diagnosis_classes=len(self.taxonomy_classes),
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        ).to(self.device)

    def encode_session(self, session: Dict[str, Any], max_len: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert a single session dictionary to tensors."""
        steps = session.get("steps", [])
        seq_len = min(len(steps), max_len)

        diag_indices = [self.padding_idx] * max_len
        confs = [0.0] * max_len
        step_pos = [0.0] * max_len
        mask = [False] * max_len

        for i in range(seq_len):
            step = steps[i]
            diag = step.get("model1_individual_diagnosis", "UNCERTAIN-GUESS")
            diag_idx = self.diag2idx.get(diag, self.diag2idx.get("UNCERTAIN-GUESS", 0))
            conf = float(step.get("model1_confidence", 0.8))

            diag_indices[i] = diag_idx
            confs[i] = conf
            step_pos[i] = (i + 1) / max(1, len(steps))
            mask[i] = True

        return (
            torch.tensor(diag_indices, dtype=torch.long),
            torch.tensor(confs, dtype=torch.float32),
            torch.tensor(step_pos, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool),
        )

    def predict_session(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """Inference for a single student attempt session."""
        self.model.eval()
        diag_idx, conf, pos, mask = self.encode_session(session)

        diag_idx = diag_idx.unsqueeze(0).to(self.device)
        conf = conf.unsqueeze(0).to(self.device)
        pos = pos.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            pattern_logits, misc_logits = self.model(diag_idx, conf, pos, mask)
            pattern_probs = F.softmax(pattern_logits, dim=-1).squeeze(0).cpu().tolist()
            pred_pattern_idx = int(torch.argmax(pattern_logits, dim=-1).item())
            pred_misc_idx = int(torch.argmax(misc_logits, dim=-1).item())

        pattern_type = self.idx2pattern[pred_pattern_idx]
        pattern_confidence = pattern_probs[pred_pattern_idx]

        persistent_misc_id = (
            self.idx2diag[pred_misc_idx] if pred_misc_idx < self.none_misc_idx else None
        )
        if pattern_type != "PERSISTENT_MISCONCEPTION":
            persistent_misc_id = None

        recommendation_map = {
            "PERSISTENT_MISCONCEPTION": "TRIGGER_TARGETED_PEDAGOGICAL_INTERVENTION",
            "TRANSIENT_SLIP": "ENCOURAGE_CAREFUL_CALCULATION_NO_REMEDIATION",
            "RESOLVING_TRAJECTORY": "CONFIRM_MASTERY_WITH_ADVANCED_TRANSFER",
        }

        return {
            "pattern_type": pattern_type,
            "persistent_misconception_id": persistent_misc_id,
            "confidence": round(float(pattern_confidence), 4),
            "pattern_probabilities": {
                p: round(float(prob), 4) for p, prob in zip(self.pattern_classes, pattern_probs)
            },
            "recommendation": recommendation_map.get(pattern_type, "MONITOR_NEXT_ATTEMPT"),
        }

    def save(self, directory: Union[str, Path]) -> None:
        """Save GRU model weights, config, and vocabulary mappings."""
        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), out_dir / "gru_weights.pt")

        metadata = {
            "taxonomy_classes": self.taxonomy_classes,
            "pattern_classes": self.pattern_classes,
            "num_diagnosis_classes": len(self.taxonomy_classes),
            "num_pattern_classes": len(self.pattern_classes),
        }
        with open(out_dir / "sequence_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info("Saved GRUSequenceAnalyzer checkpoint to %s", out_dir)

    @classmethod
    def load(cls, directory: Union[str, Path], device: Optional[str] = None) -> GRUSequenceAnalyzer:
        """Load saved GRUSequenceAnalyzer checkpoint."""
        in_dir = Path(directory)
        with open(in_dir / "sequence_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)

        analyzer = cls(
            taxonomy_classes=meta["taxonomy_classes"],
            device=device,
        )
        weights_path = in_dir / "gru_weights.pt"
        analyzer.model.load_state_dict(
            torch.load(weights_path, map_location=analyzer.device, weights_only=True)
        )
        analyzer.model.eval()
        logger.info("Loaded GRUSequenceAnalyzer checkpoint from %s on %s", in_dir, analyzer.device)
        return analyzer
