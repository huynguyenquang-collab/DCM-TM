"""
Continual Topic Model — Main entry point.

Usage:
    python main.py                           # Use default config
    python main.py --config configs/custom.yaml
    python main.py --config configs/default.yaml --epochs 50 --n-topics 30
"""

import argparse
import yaml
import sys
from pathlib import Path

from src.data_loader import load_corpus
from src.pipeline import ContinualTopicPipeline


def load_config(config_path: str) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config values from command-line arguments."""
    if args.n_topics is not None:
        config["vae"]["n_topics"] = args.n_topics
    if args.epochs is not None:
        config["vae"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["vae"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["vae"]["lr"] = args.lr
    if args.device is not None:
        config["pipeline"]["device"] = args.device
    if args.output_dir is not None:
        config["pipeline"]["output_dir"] = args.output_dir
    if args.llm_provider is not None:
        config["llm"]["provider"] = args.llm_provider
    if args.no_residual:
        config["vae"]["use_residual"] = False
    return config


def main():
    parser = argparse.ArgumentParser(description="Continual Topic Model")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to config YAML file")
    parser.add_argument("--data-dir", default=None,
                        help="Override data directory")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    parser.add_argument("--n-topics", type=int, default=None,
                        help="Number of topics per timestamp")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Max training epochs per timestamp")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Training batch size")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate")
    parser.add_argument("--device", default=None,
                        help="Device: cpu, cuda, auto")
    parser.add_argument("--llm-provider", default=None,
                        help="LLM provider: nim, gemini, none")
    parser.add_argument("--no-residual", action="store_true",
                        help="Disable residual delta learning")
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = override_config(config, args)

    # Override data dir if specified
    if args.data_dir:
        config["data"]["dir"] = args.data_dir

    # Print config
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False, indent=2))

    # Load corpus
    data_dir = config["data"]["dir"]
    print(f"Loading corpus from {data_dir}...")
    corpus = load_corpus(data_dir)

    # Run pipeline
    pipeline = ContinualTopicPipeline(config, corpus)
    pipeline.run()


if __name__ == "__main__":
    main()
