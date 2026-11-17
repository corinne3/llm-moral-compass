"""Shared command-line options and file helpers for the numbered scripts."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re

from .models import load_model


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Hugging Face model id, or 'mock' to test the pipeline")
    p.add_argument("--results", default="results", help="results root folder")
    p.add_argument("--n-per-dim", type=int, default=50,
                   help="dilemmas per character dimension (6 dimensions)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="auto",
                   help="cpu, cuda, or auto (= cuda if a GPU is available)")
    p.add_argument("--quant", default=None, choices=["4bit"],
                   help="4bit: load the weights in 4 bits (GPU + bitsandbytes), for 7B models")
    p.add_argument("--threads", type=int, default=None,
                   help="CPU threads for PyTorch (default: PyTorch's choice)")
    p.add_argument("--cpu-watts", type=float, default=None,
                   help="your CPU power in watts, for a rough energy estimate")
    return p


def slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", model_name.split("/")[-1])


def out_dir(args) -> str:
    d = os.path.join(args.results, slug(args.model))
    os.makedirs(d, exist_ok=True)
    return d


def get_model(args):
    device = args.device
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    print(f"Loading {args.model} on {device} ...", flush=True)
    return load_model(args.model, seed=args.seed, dtype=args.dtype, batch_size=args.batch_size,
                      threads=args.threads, device=device, quant=args.quant)


def save_json(obj, path: str):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
