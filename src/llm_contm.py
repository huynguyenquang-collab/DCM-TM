"""
LLM Embedding-based Continual Neural Topic Model (LLM-CoNTM).

Architecture:
  - DVAE Encoder: BoW → Log-Normal (Dirichlet approximation) → topic proportions θ
  - LLM Embedding Decoder: Uses frozen word embeddings + learnable topic embeddings
  - GRU-like Gating: Fuses global and local topic expressions non-linearly

Custom Losses:
  - ELBO (Reconstruction + KL Divergence)
  - Encoder Distillation from LLM document embeddings
  - Contrastive Decoder loss for topic-word alignment

Streaming:
  - LLM-driven Adaptive Plasticity (Semantic Surprise Index)
  - Semantic Memory Buffer for catastrophic forgetting prevention
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
    """DVAE Encoder: BoW → (μ, log σ²) using Log-Normal approximation to Dirichlet.

    Takes a normalized Bag-of-Words vector and outputs parameters for a
    Log-Normal distribution (reparameterizable Dirichlet approximation,
    following ProdLDA/AVITM style).
    """

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
        """
        Args:
            x: (batch, vocab_size) normalized BoW vector.
        Returns:
            mu: (batch, K) mean of log-normal.
            logvar: (batch, K) log-variance of log-normal.
        """
        h = self.drop(x)
        h = F.softplus(self.bn1(self.fc1(h)))
        h = F.softplus(self.bn2(self.fc2(h)))
        mu = self.bn_mu(self.fc_mu(h))
        logvar = self.fc_logvar(h)
        return mu, logvar


class NonLinearGatingModule(nn.Module):
    """GRU-like non-linear gating to fuse global and local topic embeddings.

    Gate formula:
        z = sigmoid(W_z [T_global ; ΔT_local] + b_z)
        T_fused = z ⊙ T_global + (1 - z) ⊙ ΔT_local
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        # Gate: takes concatenation of global (D) and local offset (D) → D
        self.gate_linear = nn.Linear(embed_dim, embed_dim)

    def forward(self, t_global: torch.Tensor, t_local_residual: torch.Tensor) -> torch.Tensor:
        # Chỉ cần dùng t_local để tính toán độ mở của gate
        # Gate quyết định xem ta nên "cộng" bao nhiêu phần trăm của residual vào global
        gate_weight = torch.sigmoid(self.gate_linear(t_local_residual))
        
        # Residual connection: Giữ nguyên global, chỉ cộng thêm phần offset đã được gate
        t_fused = t_global + (gate_weight * t_local_residual)
        return t_fused

class LLMEmbeddingDecoder(nn.Module):
    """Decoder using frozen LLM word embeddings and learnable topic embeddings.
    Strictly aligns with Embedded Topic Model (ETM) mathematical foundations.
    """
    def __init__(self, n_topics: int, embed_dim: int, temperature: float = 0.7):
        super().__init__()
        self.n_topics = n_topics
        self.embed_dim = embed_dim
        self.temperature = temperature

        # Global topic embeddings — treated as a frozen EMA prior (like a Target
        # Network in DQN / Mean Teacher). The optimizer never touches this;
        # it is updated exclusively via an explicit EMA call after each timestamp.
        self.register_buffer(
            "topic_embeddings_global",
            torch.randn(n_topics, embed_dim) * 0.02,
        )
        # Trainable local topic offsets: (K, D) — the only parameters the
        # optimizer updates for each timestamp.
        self.topic_offsets_local = nn.Parameter(
            torch.zeros(n_topics, embed_dim)
        )
        # Non-linear gating module
        self.gate = NonLinearGatingModule(embed_dim)
        
        # ĐÃ XÓA: self.bn = nn.BatchNorm1d(...) vì nó phá vỡ probability simplex của Theta

    def get_fused_topic_embeddings(self) -> torch.Tensor:
        """Return the gated fusion of global + local topic embeddings: (K, D)."""
        return self.gate(self.topic_embeddings_global, self.topic_offsets_local)

    def get_beta(self, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute topic-word distribution β: (K, V)."""
        t_fused = self.get_fused_topic_embeddings()
        # (K, V) = (K, D) @ (D, V) / τ
        logits = torch.mm(t_fused, word_embeddings.t()) / self.temperature
        beta = F.softmax(logits, dim=-1)
        return beta

    def forward(self, theta: torch.Tensor, word_embeddings: torch.Tensor) -> torch.Tensor:
        """Decode topic proportions to word distribution using ETM linear combination."""
        # 1. Lấy phân phối xác suất từ vựng của từng chủ đề: beta có shape (K, V)
        beta = self.get_beta(word_embeddings)
        
        # 2. Tổ hợp tuyến tính (Convex Combination): P(w|d) = Theta * Beta
        # Theta (Batch, K) @ Beta (K, V) -> (Batch, V)
        # Vì Theta và Beta đều đã được Softmax (tổng = 1), word_probs chắc chắn là phân phối chuẩn.
        word_probs = torch.mm(theta, beta)
        
        # 3. Chuyển sang log-probability để tính Negative Log-Likelihood Loss
        log_recon = torch.log(word_probs + 1e-12)

        return log_recon


class LLMEncoderDecoderCoNTM(nn.Module):
    """Full LLM-CoNTM model: DVAE Encoder + LLM Embedding Decoder.

    Global topic embeddings act as a frozen EMA prior (Target Network style).
    Only topic_offsets_local and encoder weights are updated by the optimizer.
    After each timestamp, call ema_update_global() once to absorb local learning.
    """

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

        # Standard normal prior for log-normal KL
        self.register_buffer("prior_mu", torch.zeros(n_topics))
        self.register_buffer("prior_logvar", torch.zeros(n_topics))

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = μ + σ·ε, θ = softmax(z)."""
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
        """Forward pass returning losses and latent variables.

        Args:
            bow: (batch, V) raw BoW counts.
            word_embeddings: (V, D) frozen word embeddings.
            kl_weight: KL annealing coefficient.
        """
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
        """Update frozen global prior via EMA after a full timestamp training.

        T_global ← (1 − ρ*) · T_global + ρ* · T_fused
        Called once per timestamp, NOT inside the batch loop.

        Args:
            rho_star: Adaptive EMA rate (output of compute_adaptive_lr_scale).
        """
        fused = self.decoder.get_fused_topic_embeddings().detach()
        self.decoder.topic_embeddings_global.copy_(
            (1.0 - rho_star) * self.decoder.topic_embeddings_global + rho_star * fused
        )

    def reset_local_offsets(self):
        """Zero-out local offsets at the start of each new timestamp."""
        nn.init.zeros_(self.decoder.topic_offsets_local)

    def get_topic_word_dist(self, word_embeddings: torch.Tensor) -> np.ndarray:
        """Return β: (K, V) topic-word distribution as numpy array."""
        with torch.no_grad():
            return self.decoder.get_beta(word_embeddings).cpu().numpy()


# =============================================================================
# 2. CUSTOM LOSS FUNCTION
# =============================================================================


class CoNTMTotalLoss(nn.Module):
    """Combined loss for LLM-CoNTM:
        L_total = L_ELBO + λ₁ * L_distill_enc + λ₂ * L_contrastive_dec

    Components:
      1. L_ELBO: Reconstruction + KL (from model forward pass)
      2. L_distill_enc: KL(softmax(W_s · LLM(d)) || q(θ|d)) — encoder distillation
      3. L_contrastive_dec: Contrastive topic-word alignment loss
    """

    def __init__(
        self,
        n_topics: int,
        llm_doc_dim: int,
        temperature: float = 0.7,
        lambda_distill: float = 1.0,
        lambda_contrastive: float = 0.5,
        top_n_words: int = 20,
    ):
        super().__init__()
        self.n_topics = n_topics
        self.temperature = temperature
        self.lambda_distill = lambda_distill
        self.lambda_contrastive = lambda_contrastive
        self.top_n_words = top_n_words

        # Projection from LLM doc embedding space → topic simplex
        self.distill_proj = nn.Linear(llm_doc_dim, n_topics)

    def compute_elbo(self, recon_loss: torch.Tensor, kl_loss: torch.Tensor,
                     kl_weight: float = 1.0) -> torch.Tensor:
        """Standard DVAE ELBO = Reconstruction + weighted KL."""
        return recon_loss + kl_weight * kl_loss

    def compute_distill_enc(
        self,
        llm_doc_embeddings: torch.Tensor,
        theta: torch.Tensor,
    ) -> torch.Tensor:
        """Encoder distillation loss.

        Aligns model's predicted topic distribution with dimension-reduced
        LLM document embedding via KL divergence.

        Args:
            llm_doc_embeddings: (batch, llm_doc_dim) LLM embeddings of documents.
            theta: (batch, K) model's predicted topic proportions.
        Returns:
            Scalar distillation loss.
        """
        # Project LLM embeddings to topic space and softmax
        # (batch, K)
        teacher_logits = self.distill_proj(llm_doc_embeddings)
        teacher_dist = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL(teacher || student) where student = theta
        # Use log for numerical stability
        log_theta = torch.log(theta + 1e-12)
        log_teacher = torch.log(teacher_dist + 1e-12)

        # KL(P || Q) = sum P * (log P - log Q)
        kl = (teacher_dist * (log_teacher - log_theta)).sum(dim=-1).mean()
        return kl

    def compute_contrastive_dec(
        self,
        fused_topic_embeddings: torch.Tensor,
        word_embeddings: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """Contrastive decoder loss.

        Pushes topic embeddings closer to top-N activated words and
        further from irrelevant words.

        L = -sum_k log( sum_{w in TopW(k)} exp(T_k^T e_w / τ) /
                        sum_{w' in V} exp(T_k^T e_{w'} / τ) )

        Args:
            fused_topic_embeddings: (K, D) fused topic embeddings.
            word_embeddings: (V, D) word embeddings.
            beta: (K, V) current topic-word distribution.
        Returns:
            Scalar contrastive loss.
        """
        K = fused_topic_embeddings.shape[0]
        V = word_embeddings.shape[0]

        # Similarity scores: (K, V) = (K, D) @ (D, V) / τ
        sim = torch.mm(fused_topic_embeddings, word_embeddings.t()) / self.temperature

        # Get top-N word indices for each topic from β
        # (K, top_n)
        top_n = min(self.top_n_words, V)
        _, top_indices = beta.topk(top_n, dim=-1)

        # Compute log-sum-exp over all words (denominator): (K,)
        log_denom = torch.logsumexp(sim, dim=-1)

        # Compute log-sum-exp over top-N words (numerator): (K,)
        # Gather top-N similarities
        top_sim = sim.gather(1, top_indices)  # (K, top_n)
        log_numer = torch.logsumexp(top_sim, dim=-1)

        # Loss = -mean_k (log_numer - log_denom)
        loss = -(log_numer - log_denom).mean()
        return loss

    def forward(
        self,
        model_output: dict,
        word_embeddings: torch.Tensor,
        llm_doc_embeddings: Optional[torch.Tensor] = None,
        kl_weight: float = 1.0,
    ) -> dict:
        """Compute total loss.

        Args:
            model_output: dict from LLMEncoderDecoderCoNTM.forward().
            word_embeddings: (V, D) frozen word embeddings.
            llm_doc_embeddings: (batch, llm_doc_dim) optional LLM doc embeddings.
            kl_weight: KL annealing weight.
        Returns:
            Dictionary with total_loss and component losses.
        """
        recon_loss = model_output["recon_loss"]
        kl_loss = model_output["kl_loss"]
        theta = model_output["theta"]

        # 1. ELBO
        elbo = self.compute_elbo(recon_loss, kl_loss, kl_weight)

        # 2. Distillation (only if LLM embeddings available)
        distill_loss = torch.tensor(0.0, device=recon_loss.device)
        if llm_doc_embeddings is not None and self.lambda_distill > 0:
            distill_loss = self.compute_distill_enc(llm_doc_embeddings, theta)

        # 3. Contrastive decoder loss
        contrastive_loss = torch.tensor(0.0, device=recon_loss.device)
        if self.lambda_contrastive > 0:
            with torch.no_grad():
                # Get beta for selecting top words (detach from grad)
                fused_emb = model_output.get("_fused_topic_embeddings", None)

            # Recompute fused embeddings with grad for contrastive loss
            # Access decoder's fused topic embeddings
            # (We pass the model reference through model_output for convenience)
            decoder = model_output.get("_decoder_ref", None)
            if decoder is not None:
                fused_emb_grad = decoder.get_fused_topic_embeddings()
                beta_detached = decoder.get_beta(word_embeddings).detach()
                contrastive_loss = self.compute_contrastive_dec(
                    fused_emb_grad, word_embeddings, beta_detached
                )

        total_loss = (
            elbo
            + self.lambda_distill * distill_loss
            + self.lambda_contrastive * contrastive_loss
        )

        return {
            "total_loss": total_loss,
            "elbo": elbo,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "distill_loss": distill_loss,
            "contrastive_loss": contrastive_loss,
        }


# =============================================================================
# 3. SEMANTIC MEMORY BUFFER
# =============================================================================


class SemanticMemoryBuffer:
    """Semantic memory buffer for replay-based catastrophic forgetting prevention.

    Stores representative documents based on semantic uniqueness computed
    from LLM embeddings. Implements diversity-based reservoir sampling.

    Usage:
        buffer = SemanticMemoryBuffer(max_size=1000, embed_dim=384)
        buffer.add_batch(bow_batch, embeddings_batch)
        replay_bow, replay_emb = buffer.sample(n=64)
    """

    def __init__(self, max_size: int = 1000, embed_dim: int = 384):
        self.max_size = max_size
        self.embed_dim = embed_dim

        # Storage
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
        """Compute semantic uniqueness of a document relative to buffer contents.

        Uniqueness = 1 - max cosine similarity to any document in buffer.
        High uniqueness means the document is dissimilar from everything stored.
        """
        if self.size == 0:
            return 1.0

        # (N, D) buffer embeddings
        buffer_emb = np.stack(self._embeddings)
        # Cosine similarity: embedding @ buffer^T (both L2 normalized)
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        buf_norm = buffer_emb / (np.linalg.norm(buffer_emb, axis=1, keepdims=True) + 1e-12)
        similarities = emb_norm @ buf_norm.T
        max_sim = float(similarities.max())
        return 1.0 - max_sim

    def add_batch(
        self,
        bow_batch: np.ndarray,
        embeddings_batch: np.ndarray,
    ):
        """Add a batch of documents to the buffer based on semantic uniqueness.

        Documents with higher uniqueness scores are prioritized.
        When buffer is full, least unique documents are replaced.

        Args:
            bow_batch: (N, V) BoW vectors.
            embeddings_batch: (N, D) LLM document embeddings.
        """
        n_docs = bow_batch.shape[0]

        for i in range(n_docs):
            bow_i = bow_batch[i]
            emb_i = embeddings_batch[i]
            uniqueness = self._compute_uniqueness(emb_i)

            if self.size < self.max_size:
                # Buffer not full — add directly
                self._bow.append(bow_i.copy())
                self._embeddings.append(emb_i.copy())
                self._uniqueness_scores.append(uniqueness)
            else:
                # Buffer full — replace least unique if new doc is more unique
                min_idx = int(np.argmin(self._uniqueness_scores))
                if uniqueness > self._uniqueness_scores[min_idx]:
                    self._bow[min_idx] = bow_i.copy()
                    self._embeddings[min_idx] = emb_i.copy()
                    self._uniqueness_scores[min_idx] = uniqueness

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample n documents from buffer, weighted by uniqueness.

        Args:
            n: Number of documents to sample.
        Returns:
            (bow_sample, embedding_sample) arrays.
        """
        if self.is_empty:
            raise ValueError("Cannot sample from empty buffer.")

        n = min(n, self.size)

        # Sample proportional to uniqueness scores
        scores = np.array(self._uniqueness_scores)
        probs = scores / (scores.sum() + 1e-12)

        indices = np.random.choice(self.size, size=n, replace=False, p=probs)

        bow_sample = np.stack([self._bow[i] for i in indices])
        emb_sample = np.stack([self._embeddings[i] for i in indices])
        return bow_sample, emb_sample

    def get_mean_embedding(self) -> Optional[np.ndarray]:
        """Return mean embedding of all buffer contents (for surprise computation)."""
        if self.is_empty:
            return None
        emb_stack = np.stack(self._embeddings)
        mean_emb = emb_stack.mean(axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-12)
        return mean_emb


# =============================================================================
# 4. STREAMING TRAINING LOOP
# =============================================================================


def compute_surprise_index(
    current_embeddings: np.ndarray,
    previous_mean_embedding: Optional[np.ndarray],
) -> float:
    """Compute Semantic Surprise Index η_t.

    η_t = 1 - CosineSimilarity(H_t, H_{t-1})

    Args:
        current_embeddings: (N, D) LLM embeddings of current batch.
        previous_mean_embedding: (D,) mean embedding of previous batch/buffer.
    Returns:
        Surprise index in [0, 2].
    """
    if previous_mean_embedding is None:
        return 0.5  # Default moderate surprise for first batch

    # Mean embedding of current batch
    current_mean = current_embeddings.mean(axis=0)
    current_mean = current_mean / (np.linalg.norm(current_mean) + 1e-12)

    prev_norm = previous_mean_embedding / (
        np.linalg.norm(previous_mean_embedding) + 1e-12
    )

    cosine_sim = float(np.dot(current_mean, prev_norm))
    eta = 1.0 - cosine_sim
    return max(0.0, eta)  # Clip to non-negative


def compute_adaptive_lr_scale(
    t: int,
    eta_t: float,
    gamma: float = 2.0,
    tau_0: float = 1.0,
    kappa: float = 0.7,
) -> float:
    """Compute adaptive learning rate multiplier ρ_t*.

    ρ_t = 1 / (τ₀ + t)^κ   (baseline decay)
    ρ_t* = ρ_t * (1 + γ * η_t)  (surprise-amplified)

    Args:
        t: Current streaming step index.
        eta_t: Semantic surprise index.
        gamma: Amplification factor for surprise.
        tau_0: Base offset for decay.
        kappa: Decay exponent.
    Returns:
        ρ_t* adaptive scale factor.
    """
    rho_t = 1.0 / ((tau_0 + t) ** kappa)
    rho_star = rho_t * (1.0 + gamma * eta_t)
    return min(rho_star, 0.95)  # <--- THÊM min(...) VÀO ĐÂY ĐỂ ĐẢM BẢO AN TOÀN TOÁN HỌC


def update_global_embeddings_ema(
    global_embeddings: torch.Tensor,
    local_fused_embeddings: torch.Tensor,
    rho_star: float,
) -> torch.Tensor:
    """Update global topic embeddings via EMA with adaptive rate.

    T_global_new = (1 - ρ*) * T_global + ρ* * T_local_fused

    Args:
        global_embeddings: (K, D) current global topic embeddings.
        local_fused_embeddings: (K, D) fused embeddings from current time slice.
        rho_star: adaptive learning rate.
    Returns:
        Updated global embeddings.
    """
    return (1.0 - rho_star) * global_embeddings + rho_star * local_fused_embeddings


def train_streaming_step(
    model: LLMEncoderDecoderCoNTM,
    criterion: CoNTMTotalLoss,
    optimizer: torch.optim.Optimizer,
    bow_batch: torch.Tensor,
    word_embeddings: torch.Tensor,
    llm_doc_embeddings: Optional[torch.Tensor],
    memory_buffer: SemanticMemoryBuffer,
    streaming_step: int,
    prev_mean_embedding: Optional[np.ndarray],
    kl_weight: float = 1.0,
    gamma: float = 2.0,
    tau_0: float = 1.0,
    kappa: float = 0.7,
    replay_ratio: float = 0.2,
    device: str = "cuda",
) -> dict:
    """Execute a single streaming training step.

    This function:
      1. Interleaves current data with replay data from memory buffer
      2. Computes Semantic Surprise Index
      3. Runs forward pass and computes total loss
      4. Applies adaptive plasticity to update global embeddings via EMA
      5. Updates memory buffer with new documents

    Args:
        model: The LLM-CoNTM model.
        criterion: CoNTMTotalLoss module.
        optimizer: Optimizer for model parameters.
        bow_batch: (N, V) current batch BoW vectors.
        word_embeddings: (V, D) frozen word embeddings tensor (on device).
        llm_doc_embeddings: (N, D_llm) LLM document embeddings, or None.
        memory_buffer: SemanticMemoryBuffer instance.
        streaming_step: Current step index t.
        prev_mean_embedding: Mean embedding from previous step (for surprise).
        kl_weight: Current KL annealing weight.
        gamma: Surprise amplification factor.
        tau_0: Decay base offset.
        kappa: Decay exponent.
        replay_ratio: Fraction of batch from buffer (default 20%).
        device: Computation device.
    Returns:
        Dictionary with loss values, surprise index, adaptive rate, and
        updated previous mean embedding.
    """
    model.train()

    # ─── Step A: Compute Semantic Surprise ────────────────────────────────
    current_emb_np = None
    if llm_doc_embeddings is not None:
        current_emb_np = llm_doc_embeddings.detach().cpu().numpy()

    eta_t = compute_surprise_index(current_emb_np, prev_mean_embedding)
    rho_star = compute_adaptive_lr_scale(streaming_step, eta_t, gamma, tau_0, kappa)

    # ─── Step B: Interleave with memory buffer ────────────────────────────
    if not memory_buffer.is_empty and replay_ratio > 0:
        n_replay = max(1, int(bow_batch.shape[0] * replay_ratio / (1.0 - replay_ratio)))
        n_replay = min(n_replay, memory_buffer.size)
        replay_bow_np, replay_emb_np = memory_buffer.sample(n_replay)

        replay_bow = torch.from_numpy(replay_bow_np).float().to(device)
        bow_combined = torch.cat([bow_batch, replay_bow], dim=0)

        if llm_doc_embeddings is not None:
            replay_emb = torch.from_numpy(replay_emb_np).float().to(device)
            llm_combined = torch.cat([llm_doc_embeddings, replay_emb], dim=0)
        else:
            llm_combined = None
    else:
        bow_combined = bow_batch
        llm_combined = llm_doc_embeddings

    # ─── Step C: Forward pass ─────────────────────────────────────────────
    model_output = model(bow_combined, word_embeddings, kl_weight)
    # Attach decoder reference for contrastive loss computation
    model_output["_decoder_ref"] = model.decoder

    # ─── Step D: Compute total loss ───────────────────────────────────────
    loss_dict = criterion(
        model_output=model_output,
        word_embeddings=word_embeddings,
        llm_doc_embeddings=llm_combined,
        kl_weight=kl_weight,
    )

    # ─── Step E: Backward pass and parameter update ───────────────────────
    optimizer.zero_grad()
    loss_dict["total_loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()

    # ─── Step F: Memory buffer update ───────────────────────────────────────
    # NOTE: EMA update of topic_embeddings_global is intentionally NOT done
    # here. Global is a frozen prior within a timestamp (like a Target Network).
    # The caller (run_llm_contm.py) performs a single EMA update after the
    # entire timestamp training loop completes.

    # ─── Step G: Update memory buffer ─────────────────────────────────────
    if current_emb_np is not None:
        bow_np = bow_batch.detach().cpu().numpy()
        memory_buffer.add_batch(bow_np, current_emb_np)

    # ─── Step H: Compute new mean embedding for next step ─────────────────
    new_mean_embedding = None
    if current_emb_np is not None:
        new_mean_embedding = current_emb_np.mean(axis=0)
        new_mean_embedding = new_mean_embedding / (
            np.linalg.norm(new_mean_embedding) + 1e-12
        )

    return {
        "total_loss": loss_dict["total_loss"].item(),
        "elbo": loss_dict["elbo"].item(),
        "recon_loss": loss_dict["recon_loss"].item(),
        "kl_loss": loss_dict["kl_loss"].item(),
        "distill_loss": loss_dict["distill_loss"].item(),
        "contrastive_loss": loss_dict["contrastive_loss"].item(),
        "surprise_index": eta_t,
        "adaptive_rate": rho_star,
        "new_mean_embedding": new_mean_embedding,
    }
