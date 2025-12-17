#!/usr/bin/env python3
"""
Minimal SHAFT example to run ST's permutation-equivariant BERT on CrypTen.

Loads encrypted weights produced by ST (encrypt_Bert.py), ensures permutation
keys are available, converts the PyTorch model to CrypTen, encrypts it, and
runs a demo forward pass. Uses SHAFT's MultiProcessLauncher to spawn multiple
parties (default 2).
"""

import argparse
import os
import shutil
import sys

SHAFT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SHAFT_ROOT not in sys.path:
    sys.path.insert(0, SHAFT_ROOT)

import crypten as ct
import crypten.communicator as comm
import torch
import torch.nn as nn


def _add_paths():
    """Ensure we can import MultiProcessLauncher and ST modules."""
    here = os.path.abspath(os.path.dirname(__file__))
    examples_dir = os.path.abspath(os.path.join(here, ".."))
    tc_dir = os.path.join(examples_dir, "text-classification")
    if tc_dir not in sys.path:
        sys.path.append(tc_dir)


_add_paths()

from multiprocess_launcher import MultiProcessLauncher  # noqa: E402


def ensure_st_on_path(st_root: str) -> str:
    st_root = os.path.abspath(st_root)
    if st_root not in sys.path:
        sys.path.insert(0, st_root)
    return st_root


def ensure_keys(src_dir: str, dst_dir: str = "keys"):
    """Copy permutation keys into working dir if missing."""
    needed = ("key_m.pt", "unkey_m.pt")
    os.makedirs(dst_dir, exist_ok=True)
    for name in needed:
        dst_path = os.path.join(dst_dir, name)
        if os.path.exists(dst_path):
            continue
        src_path = os.path.join(src_dir, name)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Expected key file missing: {src_path}")
        shutil.copyfile(src_path, dst_path)


class BertWrapper(nn.Module):
    """Expose only tensor inputs for CrypTen tracing."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def build_torch_model(weights_path: str, device: str):
    from mytransformers.models.bert.configuration_bert import BertConfig
    from mytransformers.models.bert.modeling_bert import BertForSequenceClassification

    cfg = BertConfig()
    model = BertForSequenceClassification(cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_crypten_model(model: nn.Module, dummy_ids, dummy_mask, dummy_token):
    wrapper = BertWrapper(model)
    crypten_model = ct.nn.from_pytorch(wrapper, (dummy_ids, dummy_mask, dummy_token))
    crypten_model.encrypt().eval()
    return crypten_model


def demo_run(crypten_model, batch_size, seq_len, vocab_size, device, rank):
    input_ids = torch.randint(
        0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
    )
    attn_mask = torch.ones_like(input_ids)
    token_type = torch.zeros_like(input_ids)

    x_ids = ct.cryptensor(input_ids).to(device)
    x_mask = ct.cryptensor(attn_mask).to(device)
    x_token = ct.cryptensor(token_type).to(device)

    logits_enc = crypten_model(x_ids, x_mask, x_token)
    if rank == 0:
        logits = logits_enc.get_plain_text()
        print(f"[demo] logits shape: {tuple(logits.shape)}")


def run_worker(args):
    rank = comm.get().get_rank()
    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    if rank == 0:
        print(
            f"[info] device={device}, batch_size={args.batch_size}, seq_len={args.seq_len}"
        )

    st_root = ensure_st_on_path(args.st_root)
    default_weights = os.path.join(st_root, "imdb-bert", "encrypted_ori_model.bin")
    weights_path = args.weights or default_weights
    default_keys = os.path.join(st_root, "ShowCase", "ViT", "keys")
    key_dir = args.key_dir or default_keys
    ensure_keys(key_dir, dst_dir=args.keys_out)

    torch_model = build_torch_model(weights_path, device=device)

    dummy_ids = torch.zeros(args.batch_size, args.seq_len, dtype=torch.long, device=device)
    dummy_mask = torch.ones_like(dummy_ids)
    dummy_token = torch.zeros_like(dummy_ids)
    crypten_model = build_crypten_model(torch_model, dummy_ids, dummy_mask, dummy_token)

    if args.save_crypten and rank == 0:
        ct.save(crypten_model, args.save_crypten)
        print(f"[info] saved CrypTen model to {args.save_crypten}")

    if args.run_demo:
        demo_run(
            crypten_model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=torch_model.config.vocab_size,
            device=device,
            rank=rank,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAFT example: run ST's encrypted BERT with CrypTen"
    )
    default_st = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "ST-main")
    )
    parser.add_argument(
        "--st-root",
        default=default_st,
        help="Path to ST-main repository (for mytransformers and weights)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to encrypted BERT weights (default: <st-root>/imdb-bert/encrypted_ori_model.bin)",
    )
    parser.add_argument(
        "--key-dir",
        default=None,
        help="Directory containing key_m.pt and unkey_m.pt (default: <st-root>/ShowCase/ViT/keys)",
    )
    parser.add_argument(
        "--keys-out",
        default="keys",
        help="Where to copy keys for runtime (model expects ./keys)",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--save-crypten",
        default=None,
        help="Optional path to save converted CrypTen model",
    )
    parser.add_argument("--world-size", type=int, default=2, help="Number of parties")
    parser.add_argument(
        "--run-demo",
        action="store_true",
        help="Run a toy forward pass after building the CrypTen model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.world_size > 1:
        launcher = MultiProcessLauncher(args.world_size, run_worker, args)
        launcher.start()
        launcher.join()
        launcher.terminate()
    else:
        run_worker(args)
