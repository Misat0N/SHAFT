#!/usr/bin/env python3
"""
Encrypt HuggingFace BERT encoder weights using ST's column-permutation scheme.

This reproduces ST-main/ShowCase/Bert_GPT2/encrypt_Bert.py, but lives inside
SHAFT so you can generate encrypted weights next to the CrypTen demo.

Notes:
- This is "weight encryption" by permutation (obfuscation), not MPC encryption.
- To preserve functionality (equivariance), the corresponding model forward
  must apply hidden-dimension shuffles around the encoder (see ST's modeling).
"""

import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Encrypt BERT weights with column permutation key (ST scheme)"
    )
    parser.add_argument(
        "--base-model",
        default="bert-base-uncased",
        help="HF model name or local path to start from",
    )
    parser.add_argument(
        "--num-labels",
        type=int,
        default=2,
        help="Number of labels for BertForSequenceClassification head",
    )
    parser.add_argument(
        "--c-idx",
        type=int,
        default=1,
        help="Which permutation key to use (index into key_m.pt/unkey_m.pt)",
    )
    parser.add_argument(
        "--key-dir",
        default="keys",
        help="Folder containing key_m.pt and unkey_m.pt",
    )
    parser.add_argument(
        "--out",
        default=os.path.join("imdb-bert", "encrypted_ori_model.bin"),
        help="Output path for encrypted state_dict (.bin via torch.save)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from transformers import BertForSequenceClassification
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it in your env, e.g. "
            "`pip install transformers`."
        ) from exc

    model = BertForSequenceClassification.from_pretrained(
        args.base_model, num_labels=args.num_labels
    )
    state_dict = model.state_dict()

    key_path = os.path.join(args.key_dir, "key_m.pt")
    unkey_path = os.path.join(args.key_dir, "unkey_m.pt")
    if not os.path.exists(key_path) or not os.path.exists(unkey_path):
        raise FileNotFoundError(
            f"Expected key files not found: {key_path} / {unkey_path}"
        )

    pc_all = torch.load(key_path)
    ipc_all = torch.load(unkey_path)
    pc, ipc = pc_all[args.c_idx], ipc_all[args.c_idx]

    prefix = "bert.encoder.layer."
    for name, value in list(state_dict.items()):
        if not name.startswith(prefix):
            continue
        short = name[len(prefix) :]

        if "attention" in short:
            enc = torch.matmul(value, ipc)  # B P_C^-1  or  W P_C^-1
            if "weight" in short and "LayerNorm" not in short:
                enc = torch.matmul(pc, enc)  # P_C W P_C^-1
            state_dict[name] = enc
        elif "intermediate" in short:
            if "weight" in short:
                state_dict[name] = torch.matmul(value, ipc)  # W P_C^-1
            elif "bias" in short:
                # keep bias unchanged
                continue
            else:
                raise ValueError(f"Unexpected param name: {name}")
        elif "output" in short:
            if "weight" in short:
                state_dict[name] = torch.matmul(pc, value)  # P_C W
            else:
                state_dict[name] = torch.matmul(value, ipc)  # B P_C^-1
        else:
            raise ValueError(f"Unexpected param name: {name}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(state_dict, args.out)
    print(f"[ok] saved encrypted weights to {args.out}")


if __name__ == "__main__":
    main()

