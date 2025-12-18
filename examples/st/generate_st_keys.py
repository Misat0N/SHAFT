#!/usr/bin/env python3
"""
Generate permutation keys (pc, ipc) for ST-style feature shuffle.

Produces two files:
  - key_m.pt   : pc  (P^{-1} = P^T for permutation matrices)
  - unkey_m.pt : ipc (P)

These are compatible with the scripts in `examples/st/`:
  - encrypt_st_bert_weights.py
  - run_st_bert_private.py (HF mode)
"""

import argparse
import os
from typing import Tuple

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Generate ST permutation keys")
    parser.add_argument(
        "--out-dir",
        default="keys",
        help="Output directory for key_m.pt / unkey_m.pt (default: ./keys)",
    )
    parser.add_argument(
        "--num-keys",
        type=int,
        default=10,
        help="Number of keys to generate (default: 10)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Hidden dimension to permute (default: 768 for BERT-base)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=12,
        help="Attention heads (default: 12 for BERT-base)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    return parser.parse_args()


def _build_multihead_perm(dim: int, num_heads: int, generator: torch.Generator) -> torch.Tensor:
    if dim % num_heads != 0:
        raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
    cols_per_head = dim // num_heads

    # Indices laid out as [head, cols_per_head]
    indices = torch.arange(dim, dtype=torch.long).reshape(num_heads, cols_per_head)

    # ST's key generator effectively applies one random permutation to the per-head columns,
    # then shuffles the head order.
    perm_cols = torch.randperm(cols_per_head, generator=generator)
    perm_heads = torch.randperm(num_heads, generator=generator)

    indices = indices[:, perm_cols]
    indices = indices[perm_heads, :]
    return indices.reshape(-1)


def _perm_to_matrix(perm: torch.Tensor, dim: int) -> torch.Tensor:
    eye = torch.eye(dim, dtype=torch.float32)
    return eye[:, perm]


def make_key_pair(dim: int, num_heads: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    perm = _build_multihead_perm(dim, num_heads, generator)
    p = _perm_to_matrix(perm, dim)
    pc = p.t().contiguous()   # P^{-1}
    ipc = p.contiguous()      # P
    return pc, ipc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    g = torch.Generator()
    g.manual_seed(args.seed)

    pcs = torch.empty((args.num_keys, args.dim, args.dim), dtype=torch.float32)
    ipcs = torch.empty((args.num_keys, args.dim, args.dim), dtype=torch.float32)

    pcs[0] = torch.eye(args.dim, dtype=torch.float32)
    ipcs[0] = torch.eye(args.dim, dtype=torch.float32)
    for i in range(1, args.num_keys):
        pc, ipc = make_key_pair(args.dim, args.num_heads, g)
        pcs[i] = pc
        ipcs[i] = ipc

    key_path = os.path.join(args.out_dir, "key_m.pt")
    unkey_path = os.path.join(args.out_dir, "unkey_m.pt")
    torch.save(pcs, key_path)
    torch.save(ipcs, unkey_path)

    print(f"[ok] wrote {key_path}")
    print(f"[ok] wrote {unkey_path}")


if __name__ == "__main__":
    main()

