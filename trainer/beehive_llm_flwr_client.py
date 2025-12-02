import gc
import math
import os
import random
import sys
from collections import OrderedDict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import argparse
import time
from contextlib import nullcontext

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import flwr as fl
from transformers import AutoTokenizer

from BeehiveLLM.dataset.dataset import PretrainDataset
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
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "manual_seed"):
                torch.mps.manual_seed(seed)  # type: ignore[attr-defined]
        except Exception:
            pass

def Logger(content):
    print(content)

   
def clear_memory(device):
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        try:
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except:
            pass
 
class DummyScaler:
    def __init__(self, enabled: bool = False):
        self._enabled = enabled

    def is_enabled(self) -> bool:
        return self._enabled

    def scale(self, loss):
        return loss

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass

    def unscale_(self, optimizer):
        pass

def _checkpoint_paths(lm_config, weight: str, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    moe_path = '_moe' if getattr(lm_config, "use_moe", False) else ''
    base = f'{weight}_{lm_config.hidden_size}{moe_path}'
    ckp_path = os.path.join(save_dir, f'{base}.pth')
    resume_path = os.path.join(save_dir, f'{base}_resume.pth')
    return ckp_path, resume_path

def save_checkpoint(lm_config, weight='full_sft', model=None, optimizer=None, epoch=0, step=0, save_dir='../checkpoints', **kwargs):
    """Save lightweight model weights and a fuller resume checkpoint (model + optimizer + extras)."""
    if model is None or optimizer is None:
        raise ValueError("model and optimizer must be provided to save a checkpoint.")

    ckp_path, resume_path = _checkpoint_paths(lm_config, weight, save_dir)
    from torch.nn.parallel import DistributedDataParallel
    state_dict = model.module.state_dict() if isinstance(model, DistributedDataParallel) else model.state_dict()

    # Save half-precision weights to a standalone file
    ckp_tmp = ckp_path + '.tmp'
    torch.save({k: v.half() for k, v in state_dict.items()}, ckp_tmp)
    os.replace(ckp_tmp, ckp_path)

    # Build resume payload with optimizer and any extra state_dict-capable kwargs
    resume_data = {
        'model': state_dict,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'world_size':  1,
    }
    for key, value in kwargs.items():
        if value is None:
            continue
        if hasattr(value, 'state_dict'):
            if isinstance(value, DistributedDataParallel):
                resume_data[key] = value.module.state_dict()
            else:
                resume_data[key] = value.state_dict()
        else:
            resume_data[key] = value

    resume_tmp = resume_path + '.tmp'
    torch.save(resume_data, resume_tmp)
    os.replace(resume_tmp, resume_path)

def load_checkpoint(lm_config, weight='full_sft', save_dir='../checkpoints'):
    """Load a previously saved resume checkpoint."""
    _, resume_path = _checkpoint_paths(lm_config, weight, save_dir)
    if not os.path.exists(resume_path):
        return None

    ckp_data = torch.load(resume_path, map_location='cpu')
    saved_ws = ckp_data.get('world_size', 1)
    current_ws =  1
    if saved_ws != current_ws:
        ckp_data['step'] = ckp_data.get('step', 0) * saved_ws // current_ws
        Logger(f'World size changed ({saved_ws}→{current_ws}); adjusted step to {ckp_data["step"]}')
    return ckp_data


def get_lr(current_step, total_steps, lr):
    """
    Cosine decay with a short linear warmup. Keeps LR >= 10% of base to avoid vanishing.
    """
    total_steps = max(1, int(total_steps))
    warmup_steps = max(1, int(0.05 * total_steps))
    step = min(max(0, current_step), total_steps)

    if step < warmup_steps:
        return lr * step / warmup_steps  # linear warmup 0 -> lr

    decay_steps = total_steps - warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps)
    min_scale = 0.1
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lr * (min_scale + (1 - min_scale) * cosine)


class LLMFlowerClient(fl.client.NumPyClient):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = self._resolve_device(args.device)
        setup_seed(args.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        self.config = LLMConfig(
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

        self.model = LLMForCausal(self.config).to(self.device)

        if args.from_weight != "none":
            moe_suffix = "_moe" if self.config.use_moe else ""
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"{args.from_weight}_{self.config.hidden_size}{moe_suffix}.pth",
            )
            if os.path.exists(ckpt_path):
                Logger(f"[Client] Loading weights from {ckpt_path}")
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=False)
            else:
                Logger(f"[Client] Weight {ckpt_path} not found, start from scratch.")

        self.loss_fct = nn.CrossEntropyLoss(reduction="none")
        self.dataset = PretrainDataset(
            data_path=args.data_path,
            tokenizer=self.tokenizer,
            max_length=args.max_seq_len,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=("cuda" in self.device),
            drop_last=True,
        )
        self.steps_per_epoch = max(1, len(self.loader))

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

        self.autocast_ctx, self.scaler = self._setup_precision()
        self.gradient_accumulation = max(1, args.accumulation_steps)

        Logger(
            f"[Client] Ready on {self.device} | data={len(self.dataset)} samples | "
            f"steps_per_epoch={self.steps_per_epoch}"
        )

    def _resolve_device(self, user_choice: str) -> str:
        if user_choice != "auto":
            return user_choice
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _setup_precision(self):
        if "cuda" in self.device and torch.cuda.is_available():
            dtype = torch.bfloat16 if self.args.dtype == "bfloat16" else torch.float16
            return (
                torch.amp.autocast(device_type="cuda", dtype=dtype),
                torch.amp.GradScaler(enabled=self.args.dtype == "float16"),
            )
        if "mps" in self.device and torch.backends.mps.is_available():
            return torch.amp.autocast(device_type="mps", dtype=torch.float16), DummyScaler()
        return nullcontext(), DummyScaler()

    # Flower API -------------------------------------------------------------
    def get_parameters(self, config):
        # Return model parameters as numpy arrays for Flower
        return [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        # Load parameters received from the server back into the model
        state_dict = OrderedDict()
        for (name, tensor), ndarray in zip(self.model.state_dict().items(), parameters):
            state_dict[name] = torch.tensor(ndarray).to(self.device)
        self.model.load_state_dict(state_dict, strict=False)

    def fit(self, parameters, config):
        # Train locally using server-provided parameters and return the updated weights

        if parameters:
            self.set_parameters(parameters)
        round_idx = int(config.get("round", 1))
        round_idx = int(config.get("round", 1))
        local_epochs = int(config.get("local_epochs", self.args.local_epochs))
        lr = float(config.get("learning_rate", self.args.learning_rate))
        accumulation = int(config.get("accumulation_steps", self.gradient_accumulation))
        total_rounds = config.get("total_rounds", self.args.total_rounds)

        last_loss = self._train_round(round_idx, total_rounds, local_epochs, lr, accumulation)
        metrics = {"loss": float(last_loss), "round": round_idx}
        return self.get_parameters(config), len(self.dataset), metrics

    def evaluate(self, parameters, config):
        if parameters:
            self.set_parameters(parameters)
        eval_loss = self._evaluate_loss(max_batches=self.args.eval_batches)
        metrics = {"loss": float(eval_loss)}
        return float(eval_loss), len(self.dataset), metrics

    # Training helpers ------------------------------------------------------
    def _train_round(self, round_idx, total_rounds, local_epochs, base_lr, accumulation_steps):
        self.model.train()
        total_steps = max(1, total_rounds * max(1, self.steps_per_epoch) * max(1, local_epochs))
        last_loss = 0.0

        for epoch in range(local_epochs):
            start_time = time.time()
            for step, (X, Y, loss_mask) in enumerate(self.loader, start=1):
                step_in_epoch=step
                global_step = (round_idx - 1) * self.steps_per_epoch * local_epochs + epoch * self.steps_per_epoch + step_in_epoch
                lr = get_lr(global_step, total_steps, base_lr)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                X = X.to(self.device)
                Y = Y.to(self.device)
                loss_mask = loss_mask.to(self.device)

                with self.autocast_ctx:
                    logits = self.model(X).logits
                    loss = self.loss_fct(
                        logits.view(-1, logits.size(-1)),
                        Y.view(-1),
                    )
                loss = (loss * loss_mask.view(-1)).sum() / (loss_mask.sum() + 1e-8)

                scaled_loss = loss / accumulation_steps
                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                if step % accumulation_steps == 0 or step == self.steps_per_epoch:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    if self.args.grad_clip and self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)

                last_loss = loss.item()

                if (step_in_epoch % self.args.log_interval == 0) or (step_in_epoch == self.steps_per_epoch):
                    spend_time = time.time() - start_time
                    # Log running metrics every log_interval steps
                    
                    spend_time_min = spend_time / 60.0
                    eta_sec = spend_time / step_in_epoch  * self.steps_per_epoch - spend_time
                    eta_min = max(0.0, eta_sec / 60.0)
                    

                    Logger(
                       f"[Client] round={round_idx} epoch={epoch + 1}/{local_epochs} "
                       f"step={step_in_epoch}/{self.steps_per_epoch}) "
                       f"loss:{last_loss:.6f} "
                       f"lr:{lr:.6f} "
                       f"elapsed:{spend_time_min:.2f}min "
                       f"eta:{eta_min:.2f}min"
                   )

            Logger(
                f"[Client] round={round_idx} epoch={epoch + 1}/{local_epochs} "
                f"loss={last_loss:.6f} time={time.time() - start_time:.2f}s"
            )

        if self.args.save_local_checkpoints:
            self._save_local_checkpoint(round_idx)

        return last_loss

    def _evaluate_loss(self, max_batches=2):
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0.0
        with torch.no_grad():
            for batch_idx, (X, Y, loss_mask) in enumerate(self.loader, start=1):
                X = X.to(self.device)
                Y = Y.to(self.device)
                loss_mask = loss_mask.to(self.device)

                logits = self.model(X).logits
                loss = self.loss_fct(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1),
                )
                total_loss += (loss * loss_mask.view(-1)).sum().item()
                total_tokens += loss_mask.sum().item()
                if batch_idx >= max_batches:
                    break
        self.model.train()
        if total_tokens == 0:
            return 0.0
        return total_loss / total_tokens

    def _save_local_checkpoint(self, round_idx: int):
        os.makedirs(self.args.checkpoint_dir, exist_ok=True)
        save_checkpoint(
            self.config,
            weight=f"{self.args.save_weight}_round{round_idx}",
            model=self.model,
            optimizer=self.optimizer,
            save_dir=self.args.checkpoint_dir,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Beehive Flower client for LLM pretraining")
    parser.add_argument("--server_address", type=str, default="127.0.0.1:8080")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=1)

    parser.add_argument("--local_epochs", type=int, default=1)
    parser.add_argument("--total_rounds", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=2)
    parser.add_argument("--save_local_checkpoints", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="../checkpoints")
    parser.add_argument("--save_weight", type=str, default="beehive_llm_client")
    parser.add_argument("--from_weight", type=str, default="none")

    parser.add_argument("--vocab_size", type=int, default=6400)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_kv_heads", type=int, default=2)
    parser.add_argument("--intermediate_size", type=int, default=None)
    parser.add_argument("--max_position_embeddings", type=int, default=32768)
    parser.add_argument("--use_moe", type=int, default=0)
    parser.add_argument("--flash_attn", type=int, default=0)
    return parser.parse_args()


def main():
    import platform
    import multiprocessing as mp
    # Choose spawn start_method on platforms where fork can break Flower/gRPC
    #   - macOS (Intel/Apple Silicon) prefers spawn for gRPC stability
    #   - Windows always uses spawn
    system = platform.system().lower()
    if system in ["darwin", "windows"]:
        try:
            mp.set_start_method("spawn", force=True)
            print("Using spawn start method for multiprocessing")
        except RuntimeError:
            # Ignore if a start method was already set
            pass
    args = parse_args()
    client = LLMFlowerClient(args)
    Logger(f"[Client] Connecting to {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, 
                                 client=client,
                                 grpc_max_message_length=1024 * 1024 * 1024)  # 1GB gRPC message limit
    


if __name__ == "__main__":
    main()
