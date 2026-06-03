"""
LLM-CoNTM Runner — Streaming training with evaluation.

Trains the LLM Embedding-based Continual Neural Topic Model on the NIPS corpus
using the streaming paradigm with adaptive plasticity and semantic memory.

Usage:
    python run_llm_contm.py
    python run_llm_contm.py --n-topics 50 --epochs 80 --device cuda
"""

import argparse
import json
import time
import numpy as np
import scipy.sparse as sp
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

# Import các thành phần cốt lõi từ src
from src.llm_contm import (
    LLMEncoderDecoderCoNTM,
    CoNTMTotalLoss,
    SemanticMemoryBuffer,
    train_streaming_step,
    compute_adaptive_lr_scale,
    compute_surprise_index,  # <--- ĐÃ THÊM HÀM NÀY
)
from src.topic_utils import (
    topic_diversity, topic_coherence_pmi, extract_topics,
)
from src.global_memory import GlobalMemory


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_CONFIG = {
    "data_dir": "data/NIPS",     # Thư mục chứa các file preprocessed (.npz, .npy, .json)
    "output_dir": "outputs_llm_contm",
    "n_topics": 50,
    "embed_dim": 384,           # Sẽ tự động ghi đè theo shape của doc_embeddings.npy
    "enc_hidden": 256,
    "dropout": 0.2,
    "temperature": 1.0,
    "lr": 0.002,
    "weight_decay": 1e-6,
    "batch_size": 256,
    "epochs_per_timestamp": 80,
    "kl_warmup_epochs": 20,
    "embedding_model": "all-MiniLM-L6-v2",
    # Loss weights
    "lambda_contrastive": 0.1,
    "top_n_contrastive": 20,
    # Streaming / adaptive plasticity
    "gamma": 2.0,
    "tau_0": 1.0,
    "kappa": 0.7,
    "replay_ratio": 0.2,
    "buffer_size": 500,
    # Evaluation
    "top_m_words": 15,
    "top_n_coherence": 10,
}


# =============================================================================
# DATASET CLASS FOR STREAMING DATA
# =============================================================================

class NIPSStreamingDataset(Dataset):
    """Dataset quản lý việc phân mảnh dữ liệu theo từng timestamp và split cụ thể."""
    def __init__(self, bow_matrix, doc_embeddings, metadata, target_ts, split="train"):
        # Lọc ra danh sách các vị trí index thuộc về timestamp và tập split mong muốn
        self.indices = [
            i for i, meta in enumerate(metadata) 
            if meta["ts"] == target_ts and meta["split"] == split
        ]
        # Cắt lấy submatrix tương ứng dạng CSR để tăng tốc độ truy xuất truy cập ngẫu nhiên
        self.bow_matrix = bow_matrix[self.indices].tocsr()
        self.doc_embeddings = doc_embeddings[self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # Chuyển đổi vector BoW thưa (sparse row) thành mảng dense float32
        bow_vec = np.array(self.bow_matrix[idx].todense()).squeeze()
        doc_emb = self.doc_embeddings[idx]
        return torch.tensor(bow_vec).float(), torch.tensor(doc_emb).float()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def check_environment():
    """Kiểm tra tính sẵn sàng của thiết bị huấn luyện CUDA."""
    print("=" * 70)
    print("ENVIRONMENT CHECK")
    print("=" * 70)

    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[OK] CUDA available: {device_name} ({mem_gb:.1f} GB)")
        device = "cuda"
    else:
        print("[WARN] CUDA not available — using CPU (training will be slow)")
        device = "cpu"

    print("=" * 70)
    return device

def train_timestamp(
    model: LLMEncoderDecoderCoNTM,
    criterion: CoNTMTotalLoss,
    train_loader: DataLoader,
    word_embeddings_tensor: torch.Tensor,
    memory_buffer: SemanticMemoryBuffer,
    streaming_step: int,
    prev_mean_embedding,
    config: dict,
    device: str,
) -> dict:
    """Huấn luyện mô hình trên một mốc thời gian (timestamp) cụ thể — Đã bỏ Early Stopping."""
    epochs = config["epochs_per_timestamp"]
    kl_warmup = config["kl_warmup_epochs"]

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    best_loss = float("inf")
    best_state = None
    history = []
    
    # Dùng để gom toàn bộ embedding của timestamp ở epoch cuối cùng
    all_timestamp_embs = []

    # Vòng lặp sẽ luôn chạy đủ số epochs cấu hình (ví dụ: 50 hoặc 80)
    for epoch in range(1, epochs + 1):
        kl_weight = min(1.0, epoch / kl_warmup) if kl_warmup > 0 else 1.0
        epoch_losses = []

        for bow_batch, llm_batch in train_loader:
            bow_batch = bow_batch.to(device)
            llm_batch = llm_batch.to(device)

            step_result = train_streaming_step(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                bow_batch=bow_batch,
                word_embeddings=word_embeddings_tensor,
                llm_doc_embeddings=llm_batch,
                memory_buffer=memory_buffer,
                kl_weight=kl_weight,
                replay_ratio=config["replay_ratio"],
                device=device,
            )
            epoch_losses.append(step_result["total_loss"])
            
            # Chỉ gom embedding ở Epoch cuối cùng để đại diện cho toàn bộ Timestamp này
            if epoch == epochs:
                all_timestamp_embs.append(llm_batch.detach().cpu().numpy())

        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)

        # CHỈ theo dõi checkpoint tốt nhất sau khi KL trọng số đã đạt 1.0 (hết warmup)
        if epoch >= kl_warmup and avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        # Log định kỳ mỗi 10 epochs hoặc epoch đầu tiên
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | Loss {avg_loss:.4f}"
                  f" | η={step_result['surprise_index']:.3f}"
                  f" | ρ*={step_result['adaptive_rate']:.4f}")

        history.append({
            "epoch": epoch,
            "loss": avg_loss,
            "surprise": step_result["surprise_index"],
            "adaptive_rate": step_result["adaptive_rate"],
        })

    # Khôi phục trạng thái có loss tối ưu nhất đạt được trong suốt quá trình chạy full epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    # Tính toán vector trung bình đại diện cho toàn bộ Timestamp
    current_timestamp_mean = None
    if all_timestamp_embs:
        current_timestamp_mean = np.concatenate(all_timestamp_embs, axis=0).mean(axis=0)

    return {
        "final_epoch": epochs,
        "best_loss": best_loss,
        "history": history,
        "timestamp_mean_embedding": current_timestamp_mean,
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(config: dict):
    """Chạy toàn bộ pipeline huấn luyện tuần tự LLM-CoNTM Streaming."""
    device = check_environment()

    # 1. LOAD TRỰC TIẾP TỪ CÁC FILE ARTIFACTS ĐÃ QUA TIỀN XỬ LÝ
    print(f"\nLoading preprocessed artifacts from {config['data_dir']}...")
    data_path = Path(config["data_dir"])
    
    train_bow = sp.load_npz(data_path / "train_bow.npz")
    doc_embeddings = np.load(data_path / "doc_embeddings.npy")
    vocab_embeddings = np.load(data_path / "vocab_embeddings.npy")
    
    with open(data_path / "vocab.txt", "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f.readlines()]
    with open(data_path / "doc_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    vocab_size = len(vocab)
    embed_dim = doc_embeddings.shape[1]  # Ghi đè số chiều tự động từ file embedding thực tế
    config["embed_dim"] = embed_dim

    # Trích xuất danh sách các mốc thời gian độc nhất (timestamps) từ metadata
    timestamps = sorted(list(set(m["ts"] for m in metadata)))
    print(f"Corpus: {vocab_size} vocab, {len(timestamps)} timestamps, Embedding Dim: {embed_dim}")

    # Đưa ma trận embedding của từ vựng lên thiết bị huấn luyện (Frozen anchor)
    word_embeddings_tensor = torch.from_numpy(vocab_embeddings).float().to(device)

    # Khởi tạo thư mục chứa kết quả đầu ra
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Khởi tạo kiến trúc mô hình cốt lõi
    print(f"\nInitializing LLM-CoNTM model...")
    model = LLMEncoderDecoderCoNTM(
        vocab_size=vocab_size,
        n_topics=config["n_topics"],
        embed_dim=embed_dim,
        enc_hidden=config["enc_hidden"],
        dropout=config["dropout"],
        temperature=config["temperature"],
    ).to(device)

    # Khởi tạo tiêu chuẩn Loss (ELBO + Contrastive)
    criterion = CoNTMTotalLoss(
        temperature=config["temperature"],
        lambda_contrastive=config["lambda_contrastive"],
        top_n_words=config["top_n_contrastive"],
    ).to(device)

    # Khởi tạo bộ nhớ đệm ngữ nghĩa (Semantic Memory Buffer) chống quên
    memory_buffer = SemanticMemoryBuffer(
        max_size=config["buffer_size"],
        embed_dim=embed_dim,
    )

    # Khởi tạo Global Memory phục vụ tương thích với module đánh giá
    global_memory = GlobalMemory(
        embedding_model=config["embedding_model"],
        max_topics=config["n_topics"] * 2,
        vocab=vocab,
    )

    all_results = []
    prev_mean_embedding = None

    print(f"\n{'='*70}")
    print("LLM-CoNTM STREAMING PIPELINE")
    print(f"{'='*70}")
    print(f"Topics: {config['n_topics']}, Embed dim: {embed_dim}")
    print(f"Temperature: {config['temperature']}")
    print(f"λ_contrastive: {config['lambda_contrastive']}")
    print(f"Buffer size: {config['buffer_size']}, Replay ratio: {config['replay_ratio']}")
    print(f"{'='*70}\n")

    # Lặp tuần tự qua từng mốc thời gian của luồng stream dữ liệu
    for step_idx, ts in enumerate(timestamps):
        t_start = time.time()
        
        # Đếm số lượng tài liệu thuộc timestamp này để log
        n_docs = sum(1 for m in metadata if m["ts"] == ts and m["split"] == "train")

        print(f"\n{'─'*70}")
        print(f"TIMESTAMP T{ts} — {n_docs} documents [step {step_idx}]")
        print(f"{'─'*70}")

        # Reset độ lệch cục bộ khi dịch chuyển sang mốc thời gian mới
        if step_idx > 0:
            model.reset_local_offsets()

        # 2. KHỞI TẠO DATALOADER VỚI DROP_LAST THAO TÁC TRÊN TỪNG MỐC THỜI GIAN
        train_dataset = NIPSStreamingDataset(
            bow_matrix=train_bow,
            doc_embeddings=doc_embeddings,
            metadata=metadata,
            target_ts=ts,
            split="train"
        )

        if len(train_dataset) == 0:
            print(f"  [WARN] No training documents found for T{ts}, skipping...")
            continue

        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True, 
            drop_last=(len(train_dataset) > config["batch_size"])
        )

        # Tiến hành huấn luyện trên timestamp hiện tại
        print(f"  Training ({config['epochs_per_timestamp']} max epochs)...")
        train_info = train_timestamp(
            model=model,
            criterion=criterion,
            train_loader=train_loader,
            word_embeddings_tensor=word_embeddings_tensor,
            memory_buffer=memory_buffer,
            streaming_step=step_idx,
            prev_mean_embedding=prev_mean_embedding,
            config=config,
            device=device,
        )

        # Lấy vector trung bình vừa tính được của cả Timestamp hiện tại
        current_ts_mean = train_info["timestamp_mean_embedding"]

        # ── EMA update of global prior (Target Network style) ──────────────
        # Tính toán Surprise thật sự giữa Timestamp này và Timestamp trước đó
        eta_timestamp = compute_surprise_index(
            current_timestamp_mean=current_ts_mean,
            previous_timestamp_mean=prev_mean_embedding
        )

        # Tính toán rho_star dựa trên Surprise thực tế cấp độ Timestamp
        rho_star = compute_adaptive_lr_scale(
            t=step_idx, 
            eta_timestamp=eta_timestamp,
            gamma=config["gamma"], 
            tau_0=config["tau_0"], 
            kappa=config["kappa"]
        )
        
        # Tiến hành EMA update vào Global Memory dài hạn
        model.ema_update_global(rho_star)
        
        print(f"  EMA global update: η_ts={eta_timestamp:.4f}, ρ*={rho_star:.4f}")

        # Cập nhật vector của timestamp này làm "quá khứ" cho timestamp kế tiếp
        prev_mean_embedding = current_ts_mean

        # Trích xuất phân phối chủ đề (Topics Matrix) CHỈ LẤY GLOBAL BEHAVIOR
        beta = model.get_global_topic_word_dist(word_embeddings_tensor)
        local_topics = extract_topics(beta, vocab, top_m=config["top_m_words"], source=f"global_T{ts}")
        diversity = topic_diversity(local_topics)

        print(f"  Global diversity: {diversity:.4f}")
        print(f"  Best loss: {train_info['best_loss']:.4f}")
        print(f"  Wall time: {time.time() - t_start:.1f}s")

        # Cập nhật thông tin vào Global Memory phục vụ hệ thống đánh giá
        beta_logits = np.log(beta + 1e-12)
        if step_idx == 0:
            global_memory.initialize_from_local(local_topics, beta_logits, ts)
        else:
            global_memory.beta_logits = beta_logits.copy()

            from src.topic_utils import embed_topics_from_beta

            current_topics = embed_topics_from_beta(
                local_topics,
                beta,
                vocab,
                config["embedding_model"]
            )

            global_memory.topics = current_topics

        # Lưu kết quả checkpoint của mốc thời gian này
        ts_dir = output_dir / f"T{ts}"
        ts_dir.mkdir(exist_ok=True)

        result_entry = {
            "timestamp": int(ts),
            "label": str(ts),
            "n_docs": n_docs,
            "diversity": diversity,
            "best_loss": train_info["best_loss"],
            "final_epoch": train_info["final_epoch"],
            "wall_time": time.time() - t_start,
        }
        all_results.append(result_entry)

        with open(ts_dir / "result.json", "w") as f:
            json.dump(result_entry, f, indent=2)

        mem_dir = ts_dir / "global_memory"
        mem_dir.mkdir(exist_ok=True)
        np.save(str(mem_dir / "beta_logits.npy"), beta_logits)
        topics_json = [{"id": t.id, "words": t.words, "source": t.source} for t in global_memory.topics]
        with open(mem_dir / "topics.json", "w") as f:
            json.dump(topics_json, f, indent=2)
        with open(mem_dir / "history.json", "w") as f:
            json.dump([], f)

    # ─── ĐÁNH GIÁ TỔNG KẾT TOÀN BỘ CHUỖI THỜI GIAN ──────────────────────
    print(f"\n{'='*70}")
    print("SAVING FINAL RESULTS & GLOBAL EVALUATION")
    print(f"{'='*70}")

    final_mem_dir = output_dir / "final_global_memory"
    final_mem_dir.mkdir(exist_ok=True)
    np.save(str(final_mem_dir / "beta_logits.npy"), global_memory.beta_logits)
    topics_json = [{"id": t.id, "words": t.words, "source": t.source} for t in global_memory.topics]
    with open(final_mem_dir / "topics.json", "w") as f:
        json.dump(topics_json, f, indent=2)
    with open(final_mem_dir / "history.json", "w") as f:
        json.dump([], f)

    with open(output_dir / "final_topics.txt", "w") as f:
        for t in global_memory.topics:
            f.write(f"Topic {t.id}: {', '.join(t.words)}\n")

    with open(output_dir / "results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nFinal topics saved to {output_dir / 'final_topics.txt'}")
    print(f"Results summary: {output_dir / 'results.json'}")

    final_diversity = topic_diversity(global_memory.topics)
    print(f"Final Topic Diversity: {final_diversity:.4f}")

    # Tính toán độ mạch lạc Coherence NPMI trên toàn bộ dữ liệu BoW hợp nhất
    coherence = topic_coherence_pmi(
        global_memory.topics, train_bow, vocab, top_n=config["top_n_coherence"]
    )
    print(f"Final Topic Coherence (NPMI): {coherence:.4f}")

    with open(output_dir / "evaluation_results.txt", "w") as f:
        f.write(f"Number of final topics: {len(global_memory.topics)}\n")
        f.write(f"Topic Diversity: {final_diversity:.4f}\n")
        f.write(f"Topic Coherence (NPMI): {coherence:.4f}\n")

    print(f"\nDone! All outputs successfully saved to {output_dir}/")
    return global_memory, all_results


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="LLM-CoNTM Streaming Training Pipeline")
    parser.add_argument("--data-dir", default="data/NIPS_processed")
    parser.add_argument("--output-dir", default="outputs_llm_contm")
    parser.add_argument("--n-topics", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lambda-contrastive", type=float, default=0.5)
    parser.add_argument("--buffer-size", type=int, default=500)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["data_dir"] = args.data_dir
    config["output_dir"] = args.output_dir
    config["n_topics"] = args.n_topics
    config["epochs_per_timestamp"] = args.epochs
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["temperature"] = args.temperature
    config["lambda_contrastive"] = args.lambda_contrastive
    config["buffer_size"] = args.buffer_size

    run_pipeline(config)


if __name__ == "__main__":
    main()