"""
Global memory for continual topic model.

Dynamic-K design: G_t ∈ R^{K_t × V} where K_t changes over time.

  K_t = K_{t-1} - |removed| + |novel|

Because K_t is always the exact count of active global topics, the local VAE
at timestamp t is created with n_topics = K_{t-1}, so residual learning

    L_t = G_{t-1} + ΔL_t

is always applicable — both sides are (K_{t-1} × V).

After curation:
  - Removed topics are dropped from the list.
  - Novel topics are appended.
  - beta_logits is exactly (K_t, V) — no zero rows, no padding.
"""

import json
import numpy as np
from pathlib import Path
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from src.topic_utils import (
    Topic, embed_topics_from_beta, extract_topics, cosine_similarity_matrix,
    get_word_embedding_matrix,
)


# ── Sparse routing helpers ────────────────────────────────────────────────────

def _entmax15(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """1.5-entmax (Tsallis α=1.5) — produces truly sparse distributions.

    Applies independently along *axis*.  Pure-numpy, fully differentiable
    in the forward pass (the backward pass is not needed here because we
    operate on numpy gate outputs, not torch autograd tensors).

    Reference: Peters et al., "Sparse Sequence-to-Sequence Models", 2019.
    """
    x = np.asarray(x, dtype=np.float64)
    # Move target axis to the end
    x = np.moveaxis(x, axis, -1)
    shape = x.shape
    x_flat = x.reshape(-1, shape[-1])

    out = np.zeros_like(x_flat)
    for i in range(x_flat.shape[0]):
        out[i] = _entmax15_1d(x_flat[i])
    out = out.reshape(shape)
    return np.moveaxis(out, -1, axis)


def _entmax15_1d(z: np.ndarray) -> np.ndarray:
    """1.5-entmax for a single 1-D vector."""
    z = z - z.max()
    sorted_z = np.sort(z)[::-1]
    n = len(z)
    cumsum = np.cumsum(sorted_z)
    rho = np.arange(1, n + 1, dtype=np.float64)
    mean_val = (cumsum - 1.0) / rho
    # Threshold: last index where sorted_z > mean
    support = sorted_z > mean_val
    k = int(support.sum())
    if k == 0:
        k = 1
    tau = (cumsum[k - 1] - 1.0) / k
    p = np.maximum(z - tau, 0.0)
    # Normalize (entmax 1.5 uses square root normalisation)
    p = p ** 2
    total = p.sum()
    if total > 0:
        p = p / total
    return p


def _row_entmax15(x: np.ndarray) -> np.ndarray:
    """Apply 1.5-entmax row-wise to a 2-D matrix."""
    return _entmax15(x, axis=1)


@dataclass
class GlobalUpdate:
    """Record of what changed in one timestamp update."""
    timestamp: int
    n_retained: int = 0
    n_removed: int = 0
    n_novel: int = 0
    retained_ids: list[int] = field(default_factory=list)
    removed_ids: list[int] = field(default_factory=list)
    novel_ids: list[int] = field(default_factory=list)


class GlobalMemory:
    """Manages the evolving global topic set with dynamic K.

    Attributes:
        topics      : list of Topic objects (length K_t — every entry is active)
        beta_logits : (K_t, vocab_size) topic-word logit matrix (no zero rows)
        history     : list of GlobalUpdate records
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_topics: int = 500,
        vocab: Optional[list[str]] = None,
    ):
        self.topics: list[Topic] = []
        self.beta_logits: Optional[np.ndarray] = None   # (K_t, vocab_size)
        self.history: list[GlobalUpdate] = []
        self.embedding_model = embedding_model
        self.max_topics = max_topics
        self.vocab = vocab
        self._next_id = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def K(self) -> int:
        """Number of active global topics."""
        return len(self.topics)

    @property
    def n_topics(self) -> int:
        """Alias for K."""
        return len(self.topics)

    @property
    def active_topics(self) -> list[Topic]:
        """All topics (every entry is active in dynamic-K design)."""
        return self.topics

    # ── Initialisation ─────────────────────────────────────────────────────

    def initialize_from_local(
        self,
        local_topics: list[Topic],
        local_beta: np.ndarray,
        timestamp: int = 0,
        vocab: Optional[list[str]] = None,
    ):
        """Initialize global memory from the first timestamp's local topics.

        All K_0 local topics become global topics; beta_logits = (K_0, V).
        """
        K = len(local_topics)
        self.topics = []
        for t in local_topics:
            new_topic = deepcopy(t)
            new_topic.id = self._next_id
            new_topic.source = "global"
            new_topic.metadata["origin_timestamp"] = timestamp
            new_topic.metadata["origin_source"] = t.source
            self.topics.append(new_topic)
            self._next_id += 1

        self.beta_logits = local_beta.copy()   # (K_0, V)

        # Ensure embeddings: alpha_k = sum_v beta_kv * e_v
        vocab = vocab or self.vocab
        if vocab is not None and not _all_topics_have_embeddings(self.topics):
            self.topics = embed_topics_from_beta(
                self.topics, self.beta_logits, vocab, self.embedding_model
            )

        update = GlobalUpdate(
            timestamp=timestamp,
            n_novel=K,
            novel_ids=[t.id for t in self.topics],
        )
        self.history.append(update)
        print(f"  Global memory initialized: K_0 = {K} topics from T{timestamp}")

    # ── Update ─────────────────────────────────────────────────────────────

    def update(
        self,
        retained_indices: list[int],
        retained_refined_words: dict[int, list[str]],
        novel_topics: list[Topic],
        novel_beta_rows: Optional[np.ndarray],
        local_beta: np.ndarray,
        timestamp: int,
    ) -> GlobalUpdate:
        """Update global memory after curation.

        retained_indices      : indices into self.topics to keep
        retained_refined_words: {idx: [refined_words]} optional word refinement
        novel_topics          : new Topic objects to append
        novel_beta_rows       : (n_novel, V) logit rows for novel topics
        local_beta            : (n_local, V) full local beta (fallback source)
        timestamp             : current timestamp index

        Result: self.topics becomes length K_t = |retained| + |novel|,
                self.beta_logits becomes (K_t, V) — exactly, no zero rows.
        """
        retained_set = set(retained_indices)
        removed_ids = []
        retained_ids = []

        new_topics: list[Topic] = []
        new_beta_rows: list[np.ndarray] = []

        # Keep retained topics
        for idx, topic in enumerate(self.topics):
            if idx in retained_set:
                t = deepcopy(topic)
                if idx in retained_refined_words and retained_refined_words[idx]:
                    t.words = retained_refined_words[idx][:len(t.words)]
                new_topics.append(t)
                new_beta_rows.append(self.beta_logits[idx])
                retained_ids.append(t.id)
            else:
                removed_ids.append(topic.id)

        # Append novel topics
        novel_ids = []
        for i, t in enumerate(novel_topics):
            if len(new_topics) >= self.max_topics:
                print(f"  Warning: max_topics={self.max_topics} reached, "
                      f"skipping novel topic {i}")
                break

            if novel_beta_rows is not None and i < len(novel_beta_rows):
                beta_row = novel_beta_rows[i]
            elif t.id < local_beta.shape[0]:
                beta_row = local_beta[t.id]
            else:
                beta_row = np.zeros(local_beta.shape[1])

            new_t = deepcopy(t)
            new_t.id = self._next_id
            new_t.source = "global"
            new_t.metadata["origin_timestamp"] = timestamp
            new_t.metadata["origin_source"] = t.source
            new_topics.append(new_t)
            new_beta_rows.append(beta_row)
            novel_ids.append(self._next_id)
            self._next_id += 1

        # Commit — K_t is now dynamic
        self.topics = new_topics
        self.beta_logits = (
            np.stack(new_beta_rows) if new_beta_rows
            else np.zeros((0, local_beta.shape[1]))
        )

        # Re-embed all topics: alpha_k = sum_v beta_kv * e_v
        if self.vocab is not None and not _all_topics_have_embeddings(self.topics):
            self.topics = embed_topics_from_beta(
                self.topics, self.beta_logits, self.vocab, self.embedding_model
            )

        update = GlobalUpdate(
            timestamp=timestamp,
            n_retained=len(retained_ids),
            n_removed=len(removed_ids),
            n_novel=len(novel_ids),
            retained_ids=retained_ids,
            removed_ids=removed_ids,
            novel_ids=novel_ids,
        )
        self.history.append(update)

        print(f"  Global memory updated at T{timestamp}: "
              f"{len(retained_ids)} retained, {len(removed_ids)} removed, "
              f"{len(novel_ids)} novel → K_{timestamp} = {self.K}")
              
        return update

    # ── Tensor access ─────────────────────────────────────────────────────

    def soft_update(
        self,
        local_topics: list[Topic],
        local_beta: np.ndarray,
        retain_gates: np.ndarray,
        novelty_gates: np.ndarray,
        vocab: list[str],
        top_m: int,
        timestamp: int,
        tau_assign: float = 0.1,
        tau_replace: float = 0.1,
        novelty_lambda: float = 0.3,
        eps: float = 1e-8,
        # ── Fix 1: Sparse routing ──
        routing: str = "entmax",          # "softmax" | "entmax"
        sim_mask_threshold: float = 0.0,  # mask sims below this before routing
        tau_anneal_rate: float = 0.0,     # per-step tau decay (0 = no anneal)
        # ── Fix 2: Multi-prototype novelty ──
        multi_novelty: bool = True,
        novelty_threshold: float = 0.5,   # gate threshold for "high novelty"
        max_novelty_clusters: int = 5,
        # ── Fix 3: Diversity regularisation ──
        diversity_weight: float = 0.01,
        # ── Fix 4: EMA anchors ──
        anchor_gamma: float = 0.95,
        anchor_weight: float = 0.005,
    ) -> dict:
        """Fixed-K soft update with survival and novelty gates.

        Incorporates four structural fixes:
          1. Sparse routing (entmax / masked softmax) for slot assignments
          2. Multi-prototype novelty routing (cluster novel locals → distinct slots)
          3. Diversity repulsion regularisation between global embeddings
          4. EMA semantic anchors to prevent long-term drift
        """
        if self.beta_logits is None or self.K == 0:
            self.initialize_from_local(local_topics, local_beta, timestamp, vocab=vocab)
            # Bootstrap EMA anchors from initial embeddings
            alpha0 = _topic_embedding_matrix(self.topics)
            if alpha0 is not None:
                self._ema_anchors = alpha0.copy()
            return {
                "mean_survival": 0.0,
                "mean_novelty": 1.0,
                "effective_novel_mass": float(len(local_topics)),
                "max_replacement_weight": 0.0,
                "diversity_reg": 0.0,
                "anchor_reg": 0.0,
            }

        K = self.K
        J = len(local_topics)
        if J == 0:
            return {
                "mean_survival": float(np.mean(retain_gates)) if K else 0.0,
                "mean_novelty": 0.0,
                "effective_novel_mass": 0.0,
                "max_replacement_weight": 0.0,
                "diversity_reg": 0.0,
                "anchor_reg": 0.0,
            }

        retain = np.clip(np.asarray(retain_gates, dtype=np.float64), eps, 1.0)
        novelty = np.clip(np.asarray(novelty_gates, dtype=np.float64), 0.0, 1.0)
        old_beta = np.asarray(self.beta_logits, dtype=np.float64)
        local_beta = np.asarray(local_beta, dtype=np.float64)

        if retain.shape[0] != K:
            raise ValueError(f"retain_gates length {retain.shape[0]} != K {K}")
        if novelty.shape[0] != J:
            raise ValueError(f"novelty_gates length {novelty.shape[0]} != J {J}")
        if local_beta.shape[0] != J:
            raise ValueError("local_beta rows must match local_topics")

        # ────────────────────────────────────────────────────────────────
        # Fix 1: Sparse routing — replace dense softmax with entmax
        # ────────────────────────────────────────────────────────────────
        sim_matrix = cosine_similarity_matrix(local_topics, self.topics)  # (J, K)

        # Optional temperature annealing
        effective_tau = tau_assign
        if tau_anneal_rate > 0 and timestamp > 0:
            effective_tau = tau_assign * (1.0 + tau_anneal_rate * timestamp)

        scaled_sims = sim_matrix / max(effective_tau, eps)

        # Optional similarity masking: zero out weak connections
        if sim_mask_threshold > 0:
            mask = sim_matrix >= sim_mask_threshold
            scaled_sims = np.where(mask, scaled_sims, -1e9)

        if routing == "entmax":
            assign_base = _row_entmax15(scaled_sims)
        else:
            assign_base = _row_softmax(scaled_sims)

        assignments = (1.0 - novelty)[:, None] * assign_base  # (J, K)

        assigned_mass = assignments.sum(axis=0)  # (K,)
        denom = retain + assigned_mass + eps
        assimilated_beta = (
            retain[:, None] * old_beta
            + assignments.T @ local_beta
        ) / denom[:, None]

        # ────────────────────────────────────────────────────────────────
        # Fix 2: Multi-prototype novelty routing
        # ────────────────────────────────────────────────────────────────
        novel_mass = float(novelty.sum())

        old_alpha = _topic_embedding_matrix(self.topics)   # (K, D) or None
        local_alpha = _topic_embedding_matrix(local_topics)  # (J, D) or None

        if novel_mass > eps and multi_novelty and local_alpha is not None:
            # Identify high-novelty local topics
            high_novel_mask = novelty >= novelty_threshold
            high_novel_idx = np.where(high_novel_mask)[0]

            if len(high_novel_idx) >= 2:
                # Cluster high-novelty topics into distinct groups
                novel_embs = local_alpha[high_novel_idx]   # (n_novel, D)
                novel_weights = novelty[high_novel_idx]
                n_clusters = min(max_novelty_clusters, len(high_novel_idx))
                cluster_labels = _simple_kmeans(novel_embs, n_clusters)

                # Build per-cluster prototypes (beta and alpha)
                prototypes_beta = []
                prototypes_alpha = []
                for c in range(n_clusters):
                    c_mask = cluster_labels == c
                    if not c_mask.any():
                        continue
                    c_idx = high_novel_idx[c_mask]
                    c_weights = novelty[c_idx]
                    c_total = c_weights.sum() + eps
                    proto_beta = (c_weights[:, None] * local_beta[c_idx]).sum(axis=0) / c_total
                    proto_alpha = (c_weights[:, None] * local_alpha[c_idx]).sum(axis=0) / c_total
                    proto_alpha = proto_alpha / (np.linalg.norm(proto_alpha) + eps)
                    prototypes_beta.append(proto_beta)
                    prototypes_alpha.append(proto_alpha)

                # Route prototypes to distinct weak slots via sparse bipartite matching
                if prototypes_alpha and old_alpha is not None:
                    weakness = 1.0 - retain  # higher = weaker slot
                    proto_alpha_mat = np.stack(prototypes_alpha)  # (C, D)
                    proto_beta_mat = np.stack(prototypes_beta)    # (C, V)

                    # Affinity: weakness * (1 - similarity to existing slot)
                    # so novel concepts go to weak AND dissimilar slots
                    slot_sim = proto_alpha_mat @ old_alpha.T  # (C, K)
                    affinity = weakness[None, :] * (1.0 - slot_sim)  # (C, K)
                    
                    # Greedy assignment: each prototype claims its best slot
                    slot_mix = np.zeros(K, dtype=np.float64)
                    slot_novelty_beta = np.zeros_like(old_beta)
                    slot_novelty_alpha = np.zeros_like(old_alpha)
                    claimed = set()
                    sorted_protos = np.argsort([-w.sum() for w in [novelty[high_novel_idx[cluster_labels == c]] for c in range(len(prototypes_alpha))]])
                    for pi in sorted_protos:
                        # Pick best unclaimed slot
                        scores = affinity[pi].copy()
                        for s in claimed:
                            scores[s] = -1e9
                        best_slot = int(np.argmax(scores))
                        claimed.add(best_slot)
                        replace_w = novelty_lambda * weakness[best_slot]
                        slot_mix[best_slot] = replace_w
                        slot_novelty_beta[best_slot] = proto_beta_mat[pi]
                        slot_novelty_alpha[best_slot] = proto_alpha_mat[pi]

                    # Also handle low-novelty locals with original single-prototype
                    low_novel_mask = (~high_novel_mask) & (novelty > eps)
                    low_novel_idx = np.where(low_novel_mask)[0]
                    if len(low_novel_idx) > 0:
                        low_mass = novelty[low_novel_idx].sum()
                        low_beta = (novelty[low_novel_idx, None] * local_beta[low_novel_idx]).sum(axis=0) / low_mass
                        low_alpha_vec = (novelty[low_novel_idx, None] * local_alpha[low_novel_idx]).sum(axis=0) / low_mass
                        low_alpha_vec = low_alpha_vec / (np.linalg.norm(low_alpha_vec) + eps)
                        # Route to weakest unclaimed slot
                        residual_weakness = weakness.copy()
                        for s in claimed:
                            residual_weakness[s] = -1e9
                        if residual_weakness.max() > 0:
                            best_low = int(np.argmax(residual_weakness))
                            low_w = novelty_lambda * weakness[best_low] * 0.5
                            slot_mix[best_low] = max(slot_mix[best_low], low_w)
                            slot_novelty_beta[best_low] = low_beta
                            slot_novelty_alpha[best_low] = low_alpha_vec

                    # Assemble new embeddings
                    has_novelty_routing = True
                    replace_weights = slot_mix / (novelty_lambda + eps)
                else:
                    has_novelty_routing = False
            else:
                has_novelty_routing = False
        else:
            has_novelty_routing = False

        # Fallback: original single-prototype novelty (when multi disabled or < 2 novel)
        if not has_novelty_routing:
            if novel_mass > eps:
                novelty_beta = (novelty[:, None] * local_beta).sum(axis=0) / novel_mass
                replace_weights = _softmax((1.0 - retain) / max(tau_replace, eps))
                slot_mix = novelty_lambda * replace_weights
                if local_alpha is not None:
                    novelty_alpha_vec = (novelty[:, None] * local_alpha).sum(axis=0) / novel_mass
                else:
                    novelty_alpha_vec = None
            else:
                novelty_beta = np.zeros(old_beta.shape[1], dtype=np.float64)
                replace_weights = np.zeros(K, dtype=np.float64)
                slot_mix = np.zeros(K, dtype=np.float64)
                novelty_alpha_vec = None

            # Build per-slot novelty arrays for uniform interface
            slot_novelty_beta = np.tile(novelty_beta, (K, 1))
            if old_alpha is not None and novelty_alpha_vec is not None:
                slot_novelty_alpha = np.tile(novelty_alpha_vec, (K, 1))
            else:
                slot_novelty_alpha = None

        # ── Assemble final beta and alpha ─────────────────────────────────
        if old_alpha is not None and local_alpha is not None:
            assimilated_alpha = (
                retain[:, None] * old_alpha
                + assignments.T @ local_alpha
            ) / denom[:, None]

            new_alpha = (
                (1.0 - slot_mix[:, None]) * assimilated_alpha
                + slot_mix[:, None] * slot_novelty_alpha
            )

            # ────────────────────────────────────────────────────────────
            # Fix 3: Diversity repulsion regularisation
            # ────────────────────────────────────────────────────────────
            diversity_reg = 0.0
            if diversity_weight > 0 and K > 1:
                normed = new_alpha / (np.linalg.norm(new_alpha, axis=1, keepdims=True) + eps)
                cos_sim = normed @ normed.T  # (K, K)
                # Zero out diagonal
                np.fill_diagonal(cos_sim, 0.0)
                diversity_reg = float((cos_sim ** 2).sum() / (K * (K - 1)))
                # Apply repulsion: push embeddings apart
                # Gradient of cos²(a_k, a_k') w.r.t. a_k ∝ cos_sim * a_k'
                repulsion_grad = (cos_sim @ normed) * 2.0 / (K * (K - 1))
                new_alpha = new_alpha - diversity_weight * repulsion_grad

            # ────────────────────────────────────────────────────────────
            # Fix 4: EMA semantic anchors
            # ────────────────────────────────────────────────────────────
            anchor_reg = 0.0
            if not hasattr(self, '_ema_anchors') or self._ema_anchors is None:
                self._ema_anchors = old_alpha.copy()

            if anchor_weight > 0 and self._ema_anchors is not None:
                anchors = self._ema_anchors
                # Compute drift penalty: weighted cosine distance
                normed_new = new_alpha / (np.linalg.norm(new_alpha, axis=1, keepdims=True) + eps)
                normed_anch = anchors / (np.linalg.norm(anchors, axis=1, keepdims=True) + eps)
                cos_to_anchor = (normed_new * normed_anch).sum(axis=1)  # (K,)
                drift = 1.0 - cos_to_anchor  # cosine distance
                anchor_reg = float(np.mean(retain * drift))
                # Pull embeddings toward anchors proportional to retain gate
                drift_grad = normed_new - normed_anch
                new_alpha = new_alpha - anchor_weight * retain[:, None] * drift_grad

            # Re-normalise
            new_alpha = new_alpha / (
                np.linalg.norm(new_alpha, axis=1, keepdims=True) + eps
            )

            # Update EMA anchors: hat{α}_k ← γ * hat{α}_k + (1-γ) * α_k
            self._ema_anchors = anchor_gamma * self._ema_anchors + (1.0 - anchor_gamma) * new_alpha

            new_beta = _beta_from_alpha(new_alpha, vocab, self.embedding_model)
        else:
            diversity_reg = 0.0
            anchor_reg = 0.0
            new_alpha = None
            new_beta = (
                (1.0 - slot_mix[:, None]) * assimilated_beta
                + slot_mix[:, None] * slot_novelty_beta
            )
            new_beta = new_beta / (new_beta.sum(axis=1, keepdims=True) + eps)

        refreshed = extract_topics(new_beta, vocab, top_m=top_m, source="global")
        for idx, topic in enumerate(refreshed):
            topic.id = self.topics[idx].id
            topic.metadata = deepcopy(self.topics[idx].metadata)
            topic.metadata["last_soft_update"] = timestamp
            topic.metadata["survival_gate"] = float(retain[idx])
            topic.metadata["replacement_weight"] = float(replace_weights[idx] if idx < len(replace_weights) else 0.0)

        if new_alpha is not None:
            for topic, emb in zip(refreshed, new_alpha):
                topic.embedding = emb.astype(np.float32)
                topic.metadata["embedding_source"] = "soft_updated_alpha"
            self.topics = refreshed
        else:
            self.topics = embed_topics_from_beta(
                refreshed, new_beta, vocab, self.embedding_model
            )
        self.beta_logits = new_beta.astype(np.float32)

        update = GlobalUpdate(
            timestamp=timestamp,
            n_retained=K,
            n_removed=0,
            n_novel=0,
            retained_ids=[t.id for t in self.topics],
            removed_ids=[],
            novel_ids=[],
        )
        self.history.append(update)

        stats = {
            "mean_survival": float(retain.mean()),
            "mean_novelty": float(novelty.mean()),
            "effective_novel_mass": novel_mass,
            "max_replacement_weight": float(replace_weights.max()) if K else 0.0,
            "diversity_reg": diversity_reg,
            "anchor_reg": anchor_reg,
            "routing": routing,
            "multi_novelty_used": has_novelty_routing if multi_novelty else False,
        }
        print(
            f"  Soft memory updated at T{timestamp}: K={self.K}, "
            f"mean survival={stats['mean_survival']:.3f}, "
            f"novel mass={stats['effective_novel_mass']:.3f}, "
            f"div_reg={diversity_reg:.4f}, anchor_reg={anchor_reg:.4f}"
        )
        return stats

    def get_beta_tensor(self, device: str = "cpu"):
        """Return the (K_t, vocab_size) logit matrix as a torch tensor.

        Shape always matches self.n_topics so the VAE can apply:
            L_t = G_{t-1} + ΔL_t   (both K_{t-1} × V)
        """
        import torch
        if self.beta_logits is None or self.K == 0:
            return None
        beta = np.asarray(self.beta_logits, dtype=np.float32)
        return torch.tensor(np.log(beta + 1e-12), dtype=torch.float32, device=device)

    # ── Summaries ──────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        """Human-readable summary of active topics."""
        lines = [f"Global Memory: K = {self.K} active topics"]
        for i, t in enumerate(self.topics):
            lines.append(f"  [{i:3d} | id {t.id:3d}] {t.to_string()}")
        return "\n".join(lines)

    def get_evolution_summary(self) -> str:
        """Summary of topic evolution across timestamps."""
        lines = ["Topic Evolution:"]
        n_active = 0
        for update in self.history:
            n_active = n_active - update.n_removed + update.n_novel
            lines.append(
                f"  T{update.timestamp}: +{update.n_novel} novel, "
                f"-{update.n_removed} removed, ={update.n_retained} retained "
                f"→ K_{update.timestamp} = {n_active}"
            )
        return "\n".join(lines)

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, path: str):
        """Save global memory to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)


        # Save (K_t, V) logit matrix — no zero rows, no padding
        if self.beta_logits is not None:
            np.save(str(path / "beta_logits.npy"), self.beta_logits)

        # Save topic list (every entry is active)
        topics_data = []
        for t in self.topics:
            topics_data.append({
                "id": t.id,
                "words": t.words,
                "word_weights": t.word_weights,
                "source": t.source,
                "metadata": t.metadata,
            })
        with open(path / "topics.json", "w") as f:
            json.dump(topics_data, f, indent=2)

        # Save history
        history_data = []
        for h in self.history:
            history_data.append({
                "timestamp": h.timestamp,
                "n_retained": h.n_retained,
                "n_removed": h.n_removed,
                "n_novel": h.n_novel,
                "retained_ids": h.retained_ids,
                "removed_ids": h.removed_ids,
                "novel_ids": h.novel_ids,
            })
        with open(path / "history.json", "w") as f:
            json.dump(history_data, f, indent=2)

    def load(self, path: str, vocab: Optional[list[str]] = None):
        """Load global memory from disk."""
        path = Path(path)

        # Load (K_t, V) logit matrix
        beta_path = path / "beta_logits.npy"
        if beta_path.exists():
            self.beta_logits = np.load(str(beta_path))

        # Load topic list
        with open(path / "topics.json") as f:
            topics_data = json.load(f)
        self.topics = []
        for td in topics_data:
            self.topics.append(Topic(
                id=td["id"],
                words=td["words"],
                word_weights=td["word_weights"],
                source=td["source"],
                metadata=td["metadata"],
            ))

        self._next_id = max(t.id for t in self.topics) + 1 if self.topics else 0

        vocab = vocab or self.vocab
        if vocab is not None and self.beta_logits is not None:
            self.topics = embed_topics_from_beta(
                self.topics, self.beta_logits, vocab, self.embedding_model
            )

        # Load history
        with open(path / "history.json") as f:
            history_data = json.load(f)
        self.history = [GlobalUpdate(**h) for h in history_data]


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum() + 1e-12)


def _row_softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (exp_x.sum(axis=1, keepdims=True) + 1e-12)


def _all_topics_have_embeddings(topics: list[Topic]) -> bool:
    return all(t.embedding is not None for t in topics)


def _topic_embedding_matrix(topics: list[Topic]) -> Optional[np.ndarray]:
    if not _all_topics_have_embeddings(topics):
        return None
    alpha = np.stack([t.embedding for t in topics]).astype(np.float64)
    return alpha / (np.linalg.norm(alpha, axis=1, keepdims=True) + 1e-12)


def _beta_from_alpha(
    alpha: np.ndarray,
    vocab: list[str],
    embedding_model: str,
) -> np.ndarray:
    word_embeddings = get_word_embedding_matrix(vocab, embedding_model).astype(np.float64)
    logits = alpha @ word_embeddings.T
    return _row_softmax(logits).astype(np.float32)


def _simple_kmeans(
    embeddings: np.ndarray,
    n_clusters: int,
    max_iter: int = 20,
) -> np.ndarray:
    """Lightweight k-means on L2-normalised embeddings (cosine k-means).

    Returns integer cluster labels of shape (n,).
    """
    n = embeddings.shape[0]
    if n_clusters >= n:
        return np.arange(n)

    # Initialise centroids with k-means++ style
    rng = np.random.RandomState(42)
    centroids_idx = [rng.randint(n)]
    for _ in range(1, n_clusters):
        dists = np.min(
            [np.linalg.norm(embeddings - embeddings[ci], axis=1) for ci in centroids_idx],
            axis=0,
        )
        probs = dists ** 2
        probs = probs / (probs.sum() + 1e-12)
        centroids_idx.append(rng.choice(n, p=probs))

    centroids = embeddings[centroids_idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign
        sims = embeddings @ centroids.T  # cosine similarity
        new_labels = sims.argmax(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        # Update centroids
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centroids[c] = embeddings[mask].mean(axis=0)
                centroids[c] /= np.linalg.norm(centroids[c]) + 1e-12
    return labels
