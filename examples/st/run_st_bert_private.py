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
from typing import Optional


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

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits


class HFFeatureShuffleBertForSequenceClassification(nn.Module):
    """HF BERT + ST-style shuffle around the encoder to preserve semantics."""

    def __init__(self, model: nn.Module, pc: torch.Tensor, ipc: torch.Tensor):
        super().__init__()
        self.model = model
        self.register_buffer("pc", pc)
        self.register_buffer("ipc", ipc)

    def forward(self, input_ids, attention_mask):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.float32)

        input_shape = input_ids.size()
        device = input_ids.device

        embedding_output = self.model.bert.embeddings(
            input_ids=input_ids,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=None,
            past_key_values_length=0,
        )
        embedding_output = torch.matmul(
            embedding_output, self.ipc.to(device=device, dtype=embedding_output.dtype)
        )

        extended_attention_mask = self.model.bert.get_extended_attention_mask(
            attention_mask, input_shape, device
        )
        head_mask = self.model.bert.get_head_mask(
            None, self.model.bert.config.num_hidden_layers
        )

        encoder_outputs = self.model.bert.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = torch.matmul(
            sequence_output, self.pc.to(device=device, dtype=sequence_output.dtype)
        )

        pooled_output = (
            self.model.bert.pooler(sequence_output)
            if self.model.bert.pooler is not None
            else None
        )
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return logits


def build_st_torch_model(weights_path: str, st_root: str, device: str):
    from mytransformers.models.bert.configuration_bert import BertConfig
    from mytransformers.models.bert.modeling_bert import BertForSequenceClassification

    cfg = BertConfig()
    model = BertForSequenceClassification(cfg)
    state = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def build_hf_torch_model(
    weights_path: str,
    key_dir: str,
    c_idx: int,
    device: str,
    num_labels: Optional[int],
):
    try:
        from transformers import BertConfig, BertForSequenceClassification
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it, e.g. `pip install transformers`."
        ) from exc

    state = torch.load(weights_path, map_location="cpu")
    if num_labels is None:
        classifier_weight = state.get("classifier.weight")
        num_labels = int(classifier_weight.shape[0]) if classifier_weight is not None else 2

    config = BertConfig(num_labels=num_labels)
    model = BertForSequenceClassification(config)
    model.load_state_dict(state, strict=True)
    model.eval()

    key_path = os.path.join(key_dir, "key_m.pt")
    unkey_path = os.path.join(key_dir, "unkey_m.pt")
    if not os.path.exists(key_path) or not os.path.exists(unkey_path):
        raise FileNotFoundError(
            f"Expected key files not found: {key_path} / {unkey_path}\n"
            "Generate them with:\n"
            "  python examples/st/generate_st_keys.py --out-dir keys"
        )

    pc = torch.load(key_path)[c_idx]
    ipc = torch.load(unkey_path)[c_idx]

    wrapped = HFFeatureShuffleBertForSequenceClassification(model, pc, ipc)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def build_crypten_model(model: nn.Module, dummy_ids, dummy_mask):
    device = dummy_ids.device
    crypten_model = ct.nn.from_pytorch(model, (dummy_ids, dummy_mask)).encrypt().to(device)
    crypten_model.eval()
    return crypten_model


def demo_run(crypten_model, batch_size, seq_len, vocab_size, device, rank, encrypt_inputs: bool):
    if encrypt_inputs:
        input_ids = torch.randint(
            0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long
        )
        attn_mask = torch.ones_like(input_ids, dtype=torch.float32)

        # Secret-share inputs (src=0 by default): only rank0 provides the plaintext.
        x_ids = ct.cryptensor(input_ids).to(device)
        x_mask = ct.cryptensor(attn_mask).to(device)
        logits_enc = crypten_model(x_ids, x_mask)
    else:
        # Public inputs must be identical across all ranks.
        if rank == 0:
            input_ids = torch.randint(
                0, vocab_size, (batch_size, seq_len), device="cpu", dtype=torch.long
            )
            attn_mask = torch.ones_like(input_ids, dtype=torch.float32)
        else:
            input_ids = torch.empty((batch_size, seq_len), device="cpu", dtype=torch.long)
            attn_mask = torch.empty((batch_size, seq_len), device="cpu", dtype=torch.float32)

        comm.get().broadcast(input_ids, 0)
        comm.get().broadcast(attn_mask, 0)
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)
        logits_enc = crypten_model(input_ids, attn_mask)

    # Reveal is a collective: all ranks must participate.
    logits = logits_enc.get_plain_text(dst=0)
    if rank == 0:
        print(f"[demo] logits shape: {tuple(logits.shape)}")


def run_worker(args):
    rank = comm.get().get_rank()
    if args.device == "cuda" and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            torch.cuda.set_device(rank % torch.cuda.device_count())
            device = f"cuda:{torch.cuda.current_device()}"
        else:
            device = "cuda"
    else:
        device = "cpu"
    print(
        f"[info] rank={rank}, device={device}, batch_size={args.batch_size}, "
        f"seq_len={args.seq_len}, encrypt_inputs={bool(args.encrypt_inputs)}"
    )

    weights_path = args.weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "Encrypted weights not found at:\n"
            f"  {weights_path}\n\n"
            "Generate it first (from SHAFT root):\n"
            "  python examples/st/encrypt_st_bert_weights.py --key-dir keys "
            "--out imdb-bert/encrypted_ori_model.bin"
        )
    key_dir = args.key_dir

    if args.mode == "st":
        if args.st_root is None:
            raise ValueError("--st-root is required when --mode st")
        st_root = ensure_st_on_path(args.st_root)
        if not os.path.isdir(os.path.join(st_root, "mytransformers")):
            raise FileNotFoundError(
                "Invalid `--st-root`: expected to find `mytransformers/` under "
                f"{st_root}. Point this to your ST-main folder."
            )
        if rank == 0:
            ensure_keys(key_dir, dst_dir=args.keys_out)
        comm.get().barrier()
        torch_model = build_st_torch_model(weights_path, st_root=st_root, device=device)
        vocab_size = int(torch_model.config.vocab_size)
        torch_model = BertWrapper(torch_model).to(device).eval()
    else:
        torch_model = build_hf_torch_model(
            weights_path,
            key_dir=key_dir,
            c_idx=args.c_idx,
            device=device,
            num_labels=args.num_labels,
        )
        vocab_size = int(torch_model.model.config.vocab_size)

    dummy_ids = torch.zeros(
        args.batch_size, args.seq_len, dtype=torch.long, device=device
    )
    dummy_mask = torch.ones_like(dummy_ids, dtype=torch.float32)
    crypten_model = build_crypten_model(torch_model, dummy_ids, dummy_mask)

    if args.save_crypten and rank == 0:
        ct.save(crypten_model, args.save_crypten)
        print(f"[info] saved CrypTen model to {args.save_crypten}")

    if args.run_demo:
        demo_run(
            crypten_model,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            vocab_size=vocab_size,
            device=device,
            rank=rank,
            encrypt_inputs=args.encrypt_inputs,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="SHAFT example: run ST's encrypted BERT with CrypTen"
    )
    parser.add_argument(
        "--mode",
        choices=["hf", "st"],
        default="hf",
        help="Run mode: `hf` (no ST repo required) or `st` (use ST-main/mytransformers).",
    )
    parser.add_argument(
        "--st-root",
        default=None,
        help="Path to ST-main repository (for mytransformers and weights)",
    )
    parser.add_argument(
        "--weights",
        default=os.path.join("imdb-bert", "encrypted_ori_model.bin"),
        help="Path to encrypted BERT weights (default: imdb-bert/encrypted_ori_model.bin)",
    )
    parser.add_argument(
        "--key-dir",
        default="keys",
        help="Directory containing key_m.pt and unkey_m.pt (default: keys)",
    )
    parser.add_argument(
        "--keys-out",
        default="keys",
        help="Where to copy keys for runtime (model expects ./keys)",
    )
    parser.add_argument(
        "--c-idx",
        type=int,
        default=1,
        help="Key index to use (must match encrypt_st_bert_weights.py --c-idx)",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=None,
        help="HF mode only: override num_labels (default: infer from weights)",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--encrypt-inputs",
        action="store_true",
        help=(
            "Encrypt input_ids / attention_mask (private tokens). "
            "Warning: this triggers a very expensive MPC embedding lookup."
        ),
    )
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
