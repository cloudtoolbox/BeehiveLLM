import os
import random
import sys
from functools import partial
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

from BeehiveLLM.model.LLMModel import LLMConfig, LLMForCausal


def setup_seed(seed: int, device: str = "auto"):
    """
    Seed RNGs based on the chosen device type. Keeps cudnn deterministic only when CUDA is used.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Resolve device type
    resolved = device
    if device == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
        elif torch.backends.mps.is_available():
            resolved = "mps"
        else:
            resolved = "cpu"

    if resolved == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif resolved == "mps" and torch.backends.mps.is_available():
        # Use torch.mps if present without shadowing global torch
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

def Logger(content):
    print(content)


# Build the global model; each round the server updates this structure's parameters and syncs them to every client.
def build_model(args: argparse.Namespace) -> LLMForCausal:
    config = LLMConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=args.max_position_embeddings,
        use_swiglu=True,
        use_moe=bool(args.use_moe),
        flash_attn=bool(args.flash_attn),
    )
    model = LLMForCausal(config)
    model.eval()  # Server only provides weights; no training runs on the server

    if args.init_weight != "none":
        moe_suffix = "_moe" if config.use_moe else ""
        ckpt_path = os.path.join(
            args.checkpoint_dir,
            f"{args.init_weight}_{config.hidden_size}{moe_suffix}.pth",
        )
        if os.path.exists(ckpt_path):
            Logger(f"[Server] Loading initial weights from {ckpt_path}")
            state_dict = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
        else:
            Logger(f"[Server] Weight {ckpt_path} not found, using random init.")
    else:
        Logger("[Server] Starting from random initialization.")

    return model

# Grab weights from the PyTorch model and convert to Flower Parameters
def initial_parameters(args: argparse.Namespace):
    model = build_model(args)
    params = [value.detach().cpu().numpy() for _, value in model.state_dict().items()]
    return ndarrays_to_parameters(params)

# Per-round client config (learning rate, local epochs, etc.)
def fit_config(args: argparse.Namespace, server_round: int):
    return {
        "round": server_round,
        "local_epochs": args.local_epochs,
        "learning_rate": args.learning_rate,
        "accumulation_steps": args.accumulation_steps,
        "max_seq_len": args.max_seq_len,
    }

# Aggregate client metrics with sample-count weighting and return global metrics; metrics is a list of (metric_dict, num_examples)
def weighted_metrics(metrics):
    total_examples = sum(num_examples for _, num_examples in metrics)
    if total_examples == 0:
        return {}
    aggregated = {}
    for metric_dict, num_examples in metrics:
        for key, value in metric_dict.items():
            aggregated.setdefault(key, 0.0)
            aggregated[key] += value * num_examples
    return {key: value / total_examples for key, value in aggregated.items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Beehive Flower server for LLM pretraining")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--num_clients", type=int, default=2, help="Expected number of clients")
    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=512)

    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--flash_attn", type=int, default=0)
    parser.add_argument("--init_weight", type=str, default="none")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--seed", type=int, default=67)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_seed(args.seed)

    init_params = initial_parameters(args)
    fit_config_fn = partial(fit_config, args)

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0, # Use all available clients each round
        fraction_evaluate=0.0, # Disable evaluation for now; training aggregation only
        # Require all clients online to start a round; fits the current "two M1 Pro both participate" setup
        min_available_clients=args.num_clients,
        min_fit_clients=args.num_clients,
        # Called at each round start; returns a dict passed as config to every client via client.fit(parameters, config)
        on_fit_config_fn=fit_config_fn,
        # Weighted aggregation of client metrics
        evaluate_metrics_aggregation_fn=weighted_metrics,
        initial_parameters=init_params,# Initial weights computed above
    )

    Logger(
        f"[Server] Starting Beehive Flower FedAvg at {args.server_address}, "
        f"rounds={args.rounds}, num_clients={args.num_clients}"
    )

    fl.server.start_server(# Actually start the gRPC server and wait for clients.
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024, # 1GB
    )


if __name__ == "__main__":
    main()
