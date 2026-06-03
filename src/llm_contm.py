"""
LLM Embedding-based Continual Neural Topic Model (LLM-CoNTM).

Architecture:
  - DVAE Encoder: BoW → Log-Normal (Dirichlet approximation) → topic proportions θ
  - LLM Embedding Decoder: Uses frozen word embeddings + learnable topic embeddings
  - GRU-like Gating: Fuses global and local topic expressions non-linearly

Custom Losses:
  - ELBO (Reconstruction + KL Divergence)
  - Contrastive Decoder loss for topic-word alignment

Streaming:
  - LLM-driven Adaptive Plasticity (Semantic Surprise Index - Timestamp Level)
  - Semantic Memory Buffer for catastrophic forgetting prevention
  - Global Topic Inference only (Local offsets are purely for adaptation)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# =============================================================================
# 1. MODEL ARCHITECTURE
# =============================================================================

class DVAEEncoder(nn.Module):
    """DVAE Encoder: BoW → (μ, log σ²) using Log-Normal approximation to Dirichlet."""

    def __init__(self, vocab_size: int, n_topics: int, hidden_dim: int = 256,
                 dropout: float = 0.2):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, n_topics)
        self.fc_logvar = nn.Linear(hidden_dim, n_topics)

        # BatchNorm on output (stabilizes training)
        self.bn_mu = nn.BatchNorm1d(n_topics, affine=False)

    def forward(self, x: torch.Tensor):
        h = self.drop(x)
        h = F.softplus(self.bn1(self.fc1(h)))
        h = F.softplus(self.bn2(self.fc2(h)))
        mu = self.bn_mu(self.fc_mu(h))
        logvar = self.fc_logvar(h)
        return mu, logvar


class NonLinearGatingModule(nn.Module):
    """GRU-like non-linear gating to fuse global and local topic embeddings."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.gate_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t_global: torch.Tensor, t_local_residual: torch.Tensor) -> torch.Tensor:
        # Gate quyết định xem ta nên "cộng" bao nhiêu phần trăm của residual vào global
        gate_weight = torch.sigmoid(self.gate_linear(t_local_residual))
        # Residual connection: Giữ nguyên global, chỉ cộng thêm phần offset đã được gate
        t_fused = t_global + (gate_weight * t_local_residual)
        return t_fused


class LLMEmbeddingDecoder(nn.Module):
    """Decoder using frozen LLM word embeddings and learnable topic embeddings."""
    
    def __init__(self, n_topics: int, embed_dim: int, temperature: float = 0.7):
        super().__init__()
        self.n_topics = n_topics
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Global topic embeddings — frozen prior
        self.register_buffer(
            "topic_embeddings_global",
            torch.randn(n_topics, embed_dim) * 0.02,
        )
        # Trainable local topic offsets
        self.topic_offsets_local = nn.Parameter(
            torch.zeros(n_topics, embed_dim)
        )
        # Non-linear gating module
        self.gate = NonLinearGatingModule(embed_dim)

    def get_fused_topic_embeddings(self) -> torch.Tensor:
        """Return the gated fusion of global + local topic embeddings: (K, D)."""
        return self.gate(self.topic_embeddings_global, self.topic_offsets_local)

    def get_beta(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute local/fused topic-word distribution β: (K, V)."""
        t_fused = self.get_fused_topic_embeddings()
        logits = torch.mm(t_fused, word_embeddings.t()) / self.temperature
        return F.softmax(logits, dim=-1)

    def get_global_beta(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pure GLOBAL topic-word distribution β: (K, V) for clean inference."""
        logits = torch.mm(self.topic_embeddings_global, word_embeddings.t()) / self.temperature
        return F.softmax(logits, dim=-1)

    def forward(self, theta: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        # 1. Beta từ fused embeddings cho training
        beta = self.get_beta(word_embeddings)
        # 2. Tổ hợp tuyến tính (Convex Combination): P(w|d) = Theta * Beta
        word_probs = torch.mm(theta, beta)
        # 3. Log-probability cho Negative Log-Likelihood Loss
        log_recon = torch.log(word_probs + 1e-12)
        return log_recon


class LLMEncoderDecoderCoNTM(nn.Module):
    """Full LLM-CoNTM model: DVAE Encoder + LLM Embedding Decoder."""

    def __init__(
        self,
        vocab_size: int,
        n_topics: int,
        embed_dim: int,
        enc_hidden: int = 256,
        dropout: float = 0.2,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_topics = n_topics
        self.embed_dim = embed_dim

        self.encoder = DVAEEncoder(vocab_size, n_topics, enc_hidden, dropout)
        self.decoder = LLMEmbeddingDecoder(n_topics, embed_dim, temperature)

        self.register_buffer("prior_mu", torch.zeros(n_topics))
        self.register_buffer("prior_logvar", torch.zeros(n_topics))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return F.softmax(z, dim=-1)

    def forward(
        self,
        bow: torch.Tensor,
        word_embeddings: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> dict:
        bow_norm = bow / (bow.sum(dim=1, keepdim=True) + 1e-12)
        mu, logvar = self.encoder(bow_norm)
        theta = self.reparameterize(mu, logvar)
        log_recon = self.decoder(theta, word_embeddings)

        recon_loss = -(bow * log_recon).sum(dim=1).mean()
        kl_loss = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()

        return {
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "loss": recon_loss + kl_weight * kl_loss,
            "theta": theta,
            "mu": mu,
            "logvar": logvar,
            "log_recon": log_recon,
        }

    @torch.no_grad()
    def ema_update_global(self, rho_star: float):
        """Update frozen global prior via EMA after a full timestamp training."""
        fused = self.decoder.get_fused_topic_embeddings().detach()
        self.decoder.topic_embeddings_global.copy_(
            (1.0 - rho_star) * self.decoder.topic_embeddings_global + rho_star * fused
        )

    def reset_local_offsets(self):
        """Zero-out local offsets at the start of each new timestamp."""
        nn.init.zeros_(self.decoder.topic_offsets_local)

    def get_topic_word_dist(self, word_embeddings: torch.Tensor) -> np.ndarray:
        """Return β: (K, V) local topic-word distribution (for training insights)."""
        with torch.no_grad():
            return self.decoder.get_beta(word_embeddings).cpu().numpy()

    def get_global_topic_word_dist(self, word_embeddings: torch.Tensor) -> np.ndarray:
        """Return β: (K, V) purely global topic-word distribution for clean Export/Inference."""
        with torch.no_grad():
            return self.decoder.get_global_beta(word_embeddings).cpu().numpy()


# =============================================================================
# 2. CUSTOM LOSS FUNCTION
# =============================================================================

class CoNTMTotalLoss(nn.Module):
    """
    Combined loss for LLM-CoNTM:
        L_total = L_ELBO + λ_contrastive * L_contrastive_dec
    """

    def __init__(
        self,
        temperature: float = 0.7,
        lambda_contrastive: float = 0.5,
        top_n_words: int = 20,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda_contrastive = lambda_contrastive
        self.top_n_words = top_n_words

    def compute_elbo(self, recon_loss: torch.Tensor, kl_loss: torch.Tensor,
                     kl_weight: float = 1.0) -> torch.Tensor:
        return recon_loss + kl_weight * kl_loss

    def compute_contrastive_dec(
        self,
        fused_topic_embeddings: torch.Tensor,
        word_embeddings: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        K = fused_topic_embeddings.shape[0]
        V = word_embeddings.shape[0]

        sim = torch.mm(fused_topic_embeddings, word_embeddings.t()) / self.temperature

        top_n = min(self.top_n_words, V)
        _, top_indices = beta.topk(top_n, dim=-1)

        log_denom = torch.logsumexp(sim, dim=-1)
        top_sim = sim.gather(1, top_indices)
        log_numer = torch.logsumexp(top_sim, dim=-1)

        loss = -(log_numer - log_denom).mean()
        return loss

    def forward(
        self,
        model_output: dict,
        word_embeddings: torch.Tensor,
        kl_weight: float = 1.0,
    ) -> dict:
        recon_loss = model_output["recon_loss"]
        kl_loss = model_output["kl_loss"]

        # 1. ELBO
        elbo = self.compute_elbo(recon_loss, kl_loss, kl_weight)

        # 2. Contrastive decoder loss
        contrastive_loss = torch.tensor(0.0, device=recon_loss.device)
        if self.lambda_contrastive > 0:
            decoder = model_output.get("_decoder_ref", None)
            if decoder is not None:
                # Fused_emb_grad keeps the computation graph attached for backprop
                fused_emb_grad = decoder.get_fused_topic_embeddings()
                # Beta is detached to serve as fixed targets for top-N words
                beta_detached = decoder.get_beta(word_embeddings).detach()
                
                contrastive_loss = self.compute_contrastive_dec(
                    fused_emb_grad, word_embeddings, beta_detached
                )

        total_loss = elbo + self.lambda_contrastive * contrastive_loss

        return {
            "total_loss": total_loss,
            "elbo": elbo,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "contrastive_loss": contrastive_loss,
        }


# =============================================================================
# 3. SEMANTIC MEMORY BUFFER
# =============================================================================

class SemanticMemoryBuffer:
    """Semantic memory buffer for replay-based catastrophic forgetting prevention."""

    def __init__(self, max_size: int = 1000, embed_dim: int = 384):
        self.max_size = max_size
        self.embed_dim = embed_dim

        self._bow: list[np.ndarray] = []
        self._embeddings: list[np.ndarray] = []
        self._uniqueness_scores: list[float] = []

    @property
    def size(self) -> int:
        return len(self._bow)

    @property
    def is_empty(self) -> bool:
        return self.size == 0

    def _compute_uniqueness(self, embedding: np.ndarray) -> float:
        if self.size == 0:
            return 1.0
        buffer_emb = np.stack(self._embeddings)
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        buf_norm = buffer_emb / (np.linalg.norm(buffer_emb, axis=1, keepdims=True) + 1e-12)
        similarities = emb_norm @ buf_norm.T
        max_sim = float(similarities.max())
        return 1.0 - max_sim

    def add_batch(self, bow_batch: np.ndarray, embeddings_batch: np.ndarray):
        n_docs = bow_batch.shape[0]
        for i in range(n_docs):
            bow_i = bow_batch[i]
            emb_i = embeddings_batch[i]
            uniqueness = self._compute_uniqueness(emb_i)

            if self.size < self.max_size:
                self._bow.append(bow_i.copy())
                self._embeddings.append(emb_i.copy())
                self._uniqueness_scores.append(uniqueness)
            else:
                min_idx = int(np.argmin(self._uniqueness_scores))
                if uniqueness > self._uniqueness_scores[min_idx]:
                    self._bow[min_idx] = bow_i.copy()
                    self._embeddings[min_idx] = emb_i.copy()
                    self._uniqueness_scores[min_idx] = uniqueness

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        if self.is_empty:
            raise ValueError("Cannot sample from empty buffer.")
        n = min(n, self.size)
        scores = np.array(self._uniqueness_scores)
        probs = scores / (scores.sum() + 1e-12)
        indices = np.random.choice(self.size, size=n, replace=False, p=probs)

        bow_sample = np.stack([self._bow[i] for i in indices])
        emb_sample = np.stack([self._embeddings[i] for i in indices])
        return bow_sample, emb_sample


# =============================================================================
# 4. STREAMING TRAINING & METRICS
# =============================================================================

def compute_surprise_index(
    current_timestamp_mean: np.ndarray,
    previous_timestamp_mean: Optional[np.ndarray],
) -> float:
    """Compute Semantic Surprise Index η_t globally at the end of a timestamp."""
    if previous_timestamp_mean is None:
        return 0.5

    current_norm = current_timestamp_mean / (np.linalg.norm(current_timestamp_mean) + 1e-12)
    prev_norm = previous_timestamp_mean / (np.linalg.norm(previous_timestamp_mean) + 1e-12)

    cosine_sim = float(np.dot(current_norm, prev_norm))
    eta = 1.0 - cosine_sim
    return max(0.0, eta)


def compute_adaptive_lr_scale(
    t: int,
    eta_timestamp: float,
    gamma: float = 2.0,
    tau_0: float = 1.0,
    kappa: float = 0.7,
) -> float:
    """Compute adaptive learning rate multiplier ρ_t* per timestamp."""
    rho_t = 1.0 / ((tau_0 + t) ** kappa)
    rho_star = rho_t * (1.0 + gamma * eta_timestamp)
    return min(rho_star, 0.95)


def train_streaming_step(
    model: LLMEncoderDecoderCoNTM,
    criterion: CoNTMTotalLoss,
    optimizer: torch.optim.Optimizer,
    bow_batch: torch.Tensor,
    word_embeddings: torch.Tensor,
    llm_doc_embeddings: Optional[torch.Tensor],
    memory_buffer: SemanticMemoryBuffer,
    kl_weight: float = 1.0,
    replay_ratio: float = 0.2,
    device: str = "cuda",
) -> dict:
    """Execute a single streaming training step inside a timestamp.
    
    NOTE: 
    - Surprise index and adaptive rate are NO LONGER computed here. 
    - EMA updates happen outside this function (at the timestamp level).
    """
    model.train()

    # Default placeholders since calculation shifted to timestamp-level
    eta_t = 0.0
    rho_star = 0.0

    current_emb_np = None
    if llm_doc_embeddings is not None:
        current_emb_np = llm_doc_embeddings.detach().cpu().numpy()

    # ─── Step A: Interleave with memory buffer ────────────────────────────
    if not memory_buffer.is_empty and replay_ratio > 0:
        n_replay = max(1, int(bow_batch.shape[0] * replay_ratio / (1.0 - replay_ratio)))
        n_replay = min(n_replay, memory_buffer.size)
        replay_bow_np, _ = memory_buffer.sample(n_replay)

        replay_bow = torch.from_numpy(replay_bow_np).float().to(device)
        bow_combined = torch.cat([bow_batch, replay_bow], dim=0)
    else:
        bow_combined = bow_batch

    # ─── Step B: Forward pass ─────────────────────────────────────────────
    model_output = model(bow_combined, word_embeddings, kl_weight)
    model_output["_decoder_ref"] = model.decoder

    # ─── Step C: Compute total loss (No llm_doc_embeddings anymore) ───────
    loss_dict = criterion(
        model_output=model_output,
        word_embeddings=word_embeddings,
        kl_weight=kl_weight,
    )

    # ─── Step D: Backward pass and parameter update ───────────────────────
    optimizer.zero_grad()
    loss_dict["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    # ─── Step E: Update memory buffer ─────────────────────────────────────
    if current_emb_np is not None:
        bow_np = bow_batch.detach().cpu().numpy()
        memory_buffer.add_batch(bow_np, current_emb_np)

    return {
        "total_loss": loss_dict["total_loss"].item(),
        "elbo": loss_dict["elbo"].item(),
        "recon_loss": loss_dict["recon_loss"].item(),
        "kl_loss": loss_dict["kl_loss"].item(),
        "contrastive_loss": loss_dict["contrastive_loss"].item(),
        "surprise_index": eta_t,       # Always 0.0 inside step
        "adaptive_rate": rho_star,     # Always 0.0 inside step
        "new_mean_embedding": None,    # Ignored, caller calculates across all batches
    }