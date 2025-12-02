import argparse
import json
import math
import os
import random
from typing import List


def load_samples(path: str) -> List[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input dataset {path} not found.")
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f if line.strip()]
    if not lines:
        raise ValueError(f"No samples found in {path}.")
    return lines


def write_shard(output_path: str, samples: List[str]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in samples:
            f.write(line + "\n")


def split_dataset(input_path: str, output_dir: str, num_clients: int, seed: int, shuffle: bool, prefix: str):
    samples = load_samples(input_path)

    if shuffle:
        random.seed(seed)
        random.shuffle(samples)

    shard_size = math.ceil(len(samples) / num_clients)
    summary = []

    for client_idx in range(num_clients):
        start = client_idx * shard_size
        end = min(start + shard_size, len(samples))
        shard_samples = samples[start:end]
        if not shard_samples:
            break
        shard_name = f"{prefix}_client{client_idx + 1}.jsonl"
        output_path = os.path.join(output_dir, shard_name)
        write_shard(output_path, shard_samples)
        summary.append((output_path, len(shard_samples)))

    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Split pretrain jsonl dataset into client-specific shards."
    )
    parser.add_argument("--input_path", type=str, required=True, help="Path to the original pretrain jsonl file.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory where client shards will be saved.")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients to split the dataset into.")
    parser.add_argument("--seed", type=int, default=67, help="Random seed used when shuffling.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle samples before splitting.")
    parser.add_argument("--prefix", type=str, default="pretrain_shard", help="Output filename prefix.")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = split_dataset(
        input_path=args.input_path,
        output_dir=args.output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
        shuffle=args.shuffle,
        prefix=args.prefix,
    )
    print("Dataset shards created:")
    for path, count in summary:
        print(f"  {path}: {count} samples")


if __name__ == "__main__":
    main()
