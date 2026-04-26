import argparse
import sys
from pathlib import Path
import scipy.sparse as sp

from src.data_loader import load_corpus
from src.global_memory import GlobalMemory
from src.topic_utils import topic_diversity, topic_coherence_pmi


def evaluate_topics(output_dir: str, data_dir: str, top_n: int = 10):
    output_path = Path(output_dir)
    data_path = Path(data_dir)

    if not output_path.exists():
        print(f"Error: Output directory '{output_dir}' does not exist.")
        print("Please run the pipeline first to generate outputs.")
        sys.exit(1)

    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        sys.exit(1)

    final_mem_path = output_path / "final_global_memory"
    if not final_mem_path.exists():
        print(f"Error: Could not find '{final_mem_path}'.")
        print("The pipeline might not have completed successfully.")
        sys.exit(1)

    print(f"Loading corpus from {data_dir}...")
    corpus = load_corpus(data_dir)

    print(f"Loading final topics from {final_mem_path}...")
    global_memory = GlobalMemory()
    # We do not strictly need the vocab to be passed to load() if embeddings aren't needed,
    # but we pass it anyway.
    global_memory.load(str(final_mem_path), vocab=corpus.vocab)
    
    topics = global_memory.topics
    if not topics:
        print("No topics found in global memory.")
        sys.exit(1)

    print(f"Evaluating {len(topics)} final topics...")

    # Compute Topic Diversity
    diversity = topic_diversity(topics)
    print(f"Topic Diversity: {diversity:.4f}")

    # Compute Topic Coherence (NPMI)
    print(f"Computing Topic Coherence (NPMI) on the full training corpus (top_n={top_n})...")
    # Stack all timestamp bow matrices to get a single bow matrix for the entire corpus
    bow_matrices = [corpus.train_data[ts].bow for ts in corpus.timestamps]
    full_bow_matrix = sp.vstack(bow_matrices)
    
    coherence = topic_coherence_pmi(topics, full_bow_matrix, corpus.vocab, top_n=top_n)
    print(f"Topic Coherence (NPMI): {coherence:.4f}")

    # Optional: Save evaluation results
    eval_results_path = output_path / "evaluation_results.txt"
    with open(eval_results_path, "w") as f:
        f.write(f"Number of final topics: {len(topics)}\n")
        f.write(f"Topic Diversity: {diversity:.4f}\n")
        f.write(f"Topic Coherence (NPMI): {coherence:.4f}\n")
    
    print(f"\nEvaluation results saved to {eval_results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate final topics from pipeline output.")
    parser.add_argument("--output-dir", default="outputs",
                        help="Path to the outputs directory")
    parser.add_argument("--data-dir", default="data/NIPS",
                        help="Path to the data directory (for coherence)")
    parser.add_argument("--top-n", type=int, default=10,
                        help="Number of top words to use for PMI coherence")
    
    args = parser.parse_args()
    
    evaluate_topics(args.output_dir, args.data_dir, args.top_n)


if __name__ == "__main__":
    main()
