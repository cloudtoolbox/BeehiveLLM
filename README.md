# BeehiveLLM Federated Pretraining

Mission: make it practical to train small, task-focused LLMs in a garage workshop setting with low-cost hardware; this is a trial path for people to see if they can effectively build a useful model on their own data.

Federated training setup for a lightweight causal language model using [Flower](https://flower.dev/). The repo includes:
- `trainer/beehive_llm_flwr_server.py`: orchestrates FedAvg rounds and seeds the global weights.
- `trainer/beehive_llm_flwr_client.py`: runs local pretraining on user data and returns updates.
- `model/LLMModel.py`: compact transformer with GQA, SwiGLU FFN, optional MoE toggle, YaRN-scaled RoPE, and flash-attention fast path.

## Features
- Federated pretraining with configurable client counts, rounds, and per-round hyperparameters.
- Transformer core supporting grouped-query attention, long-context RoPE, mixed precision, gradient accumulation, and optional flash attention.
- Pluggable tokenizer + JSONL dataset loader (`dataset/dataset.py`) for quick local experiments.
- Checkpoint utilities to resume or warm start clients and distribute initial weights from the server.

## Requirements
- Python ≥ 3.10, PyTorch with CUDA/MPS support recommended for performance.
- `flwr`, `transformers`, `numpy`, `pandas`, `scikit-learn`.
- A tokenizer directory compatible with `AutoTokenizer` (e.g., a HF tokenizer folder).

Install base dependencies (edit CUDA wheel as needed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # or cpu/mps build
pip install flwr transformers pandas scikit-learn
```

## Data Preparation
- Expected format: newline-delimited JSON with a `text` field.
```json
{"text": "Example sentence goes here."}
{"text": "Another training sample."}
```
- Point clients to the file via `--data_path` and to a tokenizer folder via `--tokenizer_path`.

- Recommended sources:
  - NanoChat: small open-source conversational/pretrain corpus; export to JSONL with a `text` field.
  - MiniMind: the upstream MiniMind pretraining JSONL (e.g., `dataset/pretrain_hq.jsonl`) works directly with this loader.

- Dataset splitter: use `dataset/split_pretrain_dataset.py` to shard a JSONL file into per-client pieces. Example:
```bash
python dataset/split_pretrain_dataset.py \
  --input_path dataset/pretrain.jsonl \
  --output_dir dataset/shards \
  --num_clients 4 \
  --shuffle \
  --prefix beehive
```

## Quickstart
1) Set Python path so the package resolves:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

2) Launch the Flower server (runs FedAvg and serves initial weights):
```bash
python trainer/beehive_llm_flwr_server.py \
  --server_address 0.0.0.0:8080 \
  --rounds 5 \
  --num_clients 2 
  
```

3) Start each client in a separate shell (adjust paths/devices):
```bash
python trainer/beehive_llm_flwr_client.py \
  --server_address 127.0.0.1:8080 \
  --device auto \
  --data_path <Your JOSNL file> \
  --tokenizer_path <Your tokenizer folder with HF config file> \
  --batch_size <Your setting such as 4 if home computer>\
  --local_epochs < First time to try, you can choose 1> \
  --total_rounds 5 \
  --learning_rate 5e-4 \
  --accumulation_steps 8 \
  --max_seq_len < based on your data such as 512> \
  --checkpoint_dir <output folder for checkpoints> \

```

When all expected clients are connected (`--num_clients` on the server), rounds begin automatically. Metrics are logged per client and aggregated on the server.

> Single-machine note: if you only have one computer, run the server and a single client on that machine and set `--num_clients 1` on the server (and any address like `127.0.0.1:8080` that both can reach). You can still federate across multiple local processes if desired.

## Apple Silicon (M1/M2/M3/M4) multi-Mac setup
- Works well for lightweight runs; each Mac runs one Flower client, and one hosts the server.
- Use `--device auto` (picks `mps`) or force `--device mps` on clients; keep `--dtype bfloat16` or `float16` for speed.
- Keep batches modest (e.g., `--batch_size 4-8`, `--accumulation_steps 8-16`) to fit M-series memory.
- Server: `--num_clients` must equal the number of Macs you launch; all must be online for each round.
- Networking: pick a reachable host (e.g., `--server_address 192.168.x.x:8080`) and ensure firewalls allow the port.
- Multiprocessing is already set to `spawn` on macOS to avoid gRPC issues; no extra flags needed.

## Configuration Highlights
- Server: `--rounds`, `--num_clients`, `--learning_rate`, `--local_epochs`, `--accumulation_steps`, `--init_weight`, `--checkpoint_dir`.
- Client: `--device auto|cuda|mps|cpu`, `--batch_size`, `--max_seq_len`, `--eval_batches`, `--grad_clip`, `--dtype float16|bfloat16`, `--save_local_checkpoints`, `--save_weight`, `--from_weight`.
- Model shape: `--vocab_size`, `--hidden_size`, `--num_layers`, `--num_heads`, `--num_kv_heads` (GQA), `--intermediate_size` (defaults inferred), `--max_position_embeddings`, `--use_moe`, `--flash_attn`.

## Checkpoints
- Server can broadcast an initial weight file named like `<init_weight>_<hidden_size>[_moe].pth` from `--checkpoint_dir`.
- Clients can warm start via `--from_weight` and save per-round checkpoints when `--save_local_checkpoints` is set.

## Tips
- Keep `--num_clients` aligned with the number of active clients; server waits for all before each round.
- For long contexts, ensure GPU memory matches `--max_position_embeddings` and sequence length.
- If using Apple Silicon or Windows, multiprocessing is forced to `spawn` to keep Flower/gRPC stable.
- If you hit GPU/MPS OOM, reduce `--batch_size` and increase `--accumulation_steps` to keep the effective batch similar without exceeding memory.
