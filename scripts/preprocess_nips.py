"""
Advanced Preprocessing Pipeline for LLM-Augmented CoNTM
Dataset: NIPS corpus
Steps:
1. Parse CSV, split into paragraphs, filter by min_words.
2. Tokenize using fast spaCy multiprocessing.
3. Chronological Train/Test Split (Strictly 90/10 sequential per time-bin).
4. Build BoW matrices & global vocabulary.
5. Extract LLM Embeddings for Documents and Vocabulary using Sentence-Transformers.
6. Export artifacts (NPZ, NPY, JSON).
"""

import csv
import sys
import os
import re
import argparse
import numpy as np
from collections import Counter, defaultdict
from tqdm import tqdm
from scipy import sparse
import json
import torch

# Increase CSV field size limit for large texts
csv.field_size_limit(sys.maxsize)

# ── Year → Timestamp Mapping ─────────────────────────────────────────────
YEAR_BINS = [
    (1987, 1989), (1990, 1992), (1993, 1995), (1996, 1998), (1999, 2001),
    (2002, 2004), (2005, 2007), (2008, 2010), (2011, 2013), (2014, 2016),
    (2017, 2019),
]

def year_to_timestamp(year: int) -> int:
    for i, (lo, hi) in enumerate(YEAR_BINS):
        if lo <= year <= hi:
            return i
    return -1

def split_into_paragraphs(text: str, min_words: int = 15) -> list:
    # 1. Fix hyphenated line breaks: "al-\ngorithm" → "algorithm"
    text = re.sub(r"-\s*\n\s*", "", text)

    # 2. Fix non-hyphenated mid-word line breaks from PDF parsing:
    #    "distribu\ntion" → "distribution"
    #    Pattern: lowercase letter at end of line, lowercase at start of next.
    text = re.sub(r"([a-z])[ \t]*\n[ \t]*([a-z])", r"\1\2", text)

    # 3. Fix space-separated word fragments from PDF layout parsing:
    #    "alterna tives" → "alternatives", "distribu tions" → "distributions"
    #    Detects: word-stem + spurious space + known English suffix/ending.
    _BROKEN_SUFFIX = re.compile(
        r"([a-z]{2,}) "
        r"(tion[s]?|tive[s]?|ment[s]?|ness(?:[a-z]*)?|ful|less|"
        r"ous|ious|eous|able[s]?|ible[s]?|ance[s]?|ence[s]?|"
        r"ity|ities|ism[s]?|ist[s]?|ize[ds]?|izing|ify(?:ing|ied)?|"
        r"ate[ds]?|ating|ing[s]?|edly|ingly|ably|ibly|"
        r"tic[s]?|ical(?:ly)?|ive[s]?|ory|ories|ure[s]?|"
        r"ers|ors|ward[s]?|wise)\b"
    )
    # Apply twice to handle rare double-space breaks ("distribu ti ons")
    text = _BROKEN_SUFFIX.sub(r"\1\2", text)
    text = _BROKEN_SUFFIX.sub(r"\1\2", text)

    # 4. Split on blank lines into paragraphs
    raw_paras = re.split(r"\n\s*\n", text)
    paras = []
    for p in raw_paras:
        p = p.strip()
        p = re.sub(r"\s+", " ", p)
        if len(p.split()) >= min_words:
            paras.append(p)
    return paras

def tokenize_spacy(texts: list, batch_size: int = 1000) -> list:
    import spacy
    # KHÔNG dùng blank nữa, phải dùng core_web_sm để có POS Tagger
    # Chạy lệnh này ở terminal trước: python -m spacy download en_core_web_sm
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    nlp.max_length = 2_000_000

    # Chỉ định các từ loại được phép giữ lại (Loại bỏ PROPN - Tên riêng)
    ALLOWED_POS = {"NOUN", "VERB", "ADJ", "ADV"}

    # Common PDF-parsing artifact fragments that slip through as valid alpha tokens.
    # These are bare suffixes/prefixes that are never meaningful as standalone words.
    FRAGMENT_BLOCKLIST = {
        "tion", "tions", "tion", "tive", "tives", "ment", "ments",
        "ness", "ful", "less", "ous", "ious", "eous", "able", "ible",
        "ance", "ence", "ances", "ences", "ity", "ities", "ism", "isms",
        "ist", "ists", "ize", "izes", "ized", "izer", "izing",
        "ify", "ified", "ifier", "ifying", "ify",
        "ate", "ated", "ates", "ating", "ation",
        "ing", "ings", "edly", "ingly", "ably", "ibly",
        "tic", "tics", "stic", "ical", "ically",
        "ive", "ives", "ory", "ories", "ory",
        "ers", "ors", "ure", "ures", "ive",
    }

    n_cpus = min(os.cpu_count() or 1, 16)
    print(f"  Using spaCy POS Tagger with {n_cpus} processes...")

    # Hide CUDA from forked spaCy worker processes — they are CPU-only and
    # each worker would otherwise attempt (and fail) to init the CUDA driver,
    # producing a noisy but harmless warning. We restore the variable after.
    _saved_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    all_tokens = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_cpus), total=len(texts), desc="Tokenizing"):
        tokens = []
        for tok in doc:
            t = tok.lemma_.lower().strip() # Dùng lemma (từ gốc) thay vì text thô
            
            # Lọc: Không phải stopword, không phải dấu câu, đúng từ loại cho phép
            if (not tok.is_stop and not tok.is_punct and not tok.is_space):
                # Loại bỏ các từ vỡ, từ chỉ có 1-2 chữ cái, hoặc chứa số/kí tự lạ
                # Cũng loại bỏ các suffix fragment như 'tion', 'tive', 'ment'...
                if len(t) >= 3 and t.isalpha() and t not in FRAGMENT_BLOCKLIST:
                    tokens.append(t)
        all_tokens.append(tokens)

    # Restore CUDA visibility for the main process after workers are done
    if _saved_cuda is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = _saved_cuda

    return all_tokens
def main():
    parser = argparse.ArgumentParser(description="Preprocess NIPS for LLM-CoNTM")
    parser.add_argument("--input", default="data/NIPS_raw/papers.csv")
    parser.add_argument("--output-dir", default="data/NIPS_processed")
    parser.add_argument("--min-para-words", type=int, default=15)
    parser.add_argument("--min-df", type=float, default=0.0005)
    parser.add_argument("--max-df", type=float, default=0.95)
    parser.add_argument("--llm-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Step 1: Read & Chunk ──────────────────────────────────────────────
    print("\nStep 1: Reading and Splitting Paragraphs...")
    raw_documents = [] # list of dicts
    
    with open(args.input, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Parsing CSV"):
            year = int(row["year"])
            ts = year_to_timestamp(year)
            if ts < 0: continue
            
            full_text = row.get("full_text", "")
            if not full_text: continue
            
            paras = split_into_paragraphs(full_text, min_words=args.min_para_words)
            for p in paras:
                raw_documents.append({"year": year, "ts": ts, "raw_text": p})

    # Sort documents strictly by year to ensure chronological order
    raw_documents.sort(key=lambda x: x["year"])
    
    # ── Step 2: Tokenize ──────────────────────────────────────────────────
    print("\nStep 2: Tokenizing...")
    texts_to_tokenize = [d["raw_text"] for d in raw_documents]
    tokenized_docs = tokenize_spacy(texts_to_tokenize)
    
    # Filter empty after tokenization
    valid_docs = []
    for i, tokens in enumerate(tokenized_docs):
        if len(tokens) > 0:
            doc_info = raw_documents[i].copy()
            doc_info["processed_text"] = " ".join(tokens)
            valid_docs.append(doc_info)
            
    print(f"  Valid documents after tokenization: {len(valid_docs)}")

    # ── Step 3: Chronological Train/Test Split ────────────────────────────
    print("\nStep 3: Temporal Train/Test Split (Sequential 90/10)...")
    docs_by_ts = defaultdict(list)
    for doc in valid_docs:
        docs_by_ts[doc["ts"]].append(doc)
        
    train_docs, test_docs = [], []
    for ts in sorted(docs_by_ts.keys()):
        ts_docs = docs_by_ts[ts]
        split_idx = int(len(ts_docs) * 0.9) # Strict cutoff (Past = Train, Future = Test)
        
        for d in ts_docs[:split_idx]:
            d["split"] = "train"
            train_docs.append(d)
        for d in ts_docs[split_idx:]:
            d["split"] = "test"
            test_docs.append(d)

    print(f"  Train set: {len(train_docs)} docs")
    print(f"  Test set: {len(test_docs)} docs")
    
    all_final_docs = train_docs + test_docs # Ordered naturally by ts

    # ── Step 4: Build Global BoW & Vocabulary ─────────────────────────────
    print(f"\nStep 4: Building BoW (min_df={args.min_df}, max_df={args.max_df})...")
    from sklearn.feature_extraction.text import CountVectorizer

    # We fit ONLY on train set to prevent vocab leakage from the test set
    vectorizer = CountVectorizer(min_df=args.min_df, max_df=args.max_df, token_pattern=r"(?u)\b[a-zA-Z]{3,}\b")
    train_texts = [d["processed_text"] for d in train_docs]
    
    vectorizer.fit(train_texts)
    vocab = vectorizer.get_feature_names_out()
    print(f"  Global Vocabulary size: {len(vocab)}")

    # Transform both
    train_bow = vectorizer.transform(train_texts)
    test_bow = vectorizer.transform([d["processed_text"] for d in test_docs])

    # ── Step 5: Extract LLM Embeddings ────────────────────────────────────
    print(f"\nStep 5: Extracting LLM Embeddings using '{args.llm_model}'...")
    from sentence_transformers import SentenceTransformer
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device detected: {device}")
    model = SentenceTransformer(args.llm_model, device=device)
    
    # 5a. Document Embeddings (Using RAW text for better LLM context, not tokenized text)
    all_raw_texts = [d["raw_text"] for d in all_final_docs]
    print("  Encoding Documents...")
    doc_embeddings = model.encode(all_raw_texts, batch_size=256, show_progress_bar=True, convert_to_numpy=True)
    
    # 5b. Vocabulary Embeddings
    print("  Encoding Vocabulary...")
    vocab_embeddings = model.encode(vocab.tolist(), batch_size=512, show_progress_bar=True, convert_to_numpy=True)

    # ── Step 6: Save Everything ───────────────────────────────────────────
    print(f"\nStep 6: Saving artifacts to {args.output_dir}/...")
    
    # Save Sparse Matrices
    sparse.save_npz(os.path.join(args.output_dir, "train_bow.npz"), train_bow)
    sparse.save_npz(os.path.join(args.output_dir, "test_bow.npz"), test_bow)
    
    # Save Embeddings
    np.save(os.path.join(args.output_dir, "doc_embeddings.npy"), doc_embeddings)
    np.save(os.path.join(args.output_dir, "vocab_embeddings.npy"), vocab_embeddings)
    
    # Save Vocab
    with open(os.path.join(args.output_dir, "vocab.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(vocab))
        
    # Save Document Metadata (so the PyTorch Dataloader knows timestamps and splits)
    metadata = [
        {"ts": d["ts"], "year": d["year"], "split": d["split"]} 
        for d in all_final_docs
    ]
    with open(os.path.join(args.output_dir, "doc_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE - READY FOR LLM-CoNTM")
    print("=" * 60)

if __name__ == "__main__":
    main()