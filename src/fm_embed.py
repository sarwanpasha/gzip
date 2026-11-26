"""Frozen foundation-model embeddings (no fine-tuning): final-layer hidden states, mean-pooled
over residue/nucleotide tokens (special tokens excluded). Sequences longer than the model context
are split into consecutive windows; window means are averaged with weights equal to window length.

  --device cuda           embeds every sequence -> data/emb/<ds>/<model>.npy (+ timing json)
  --device cpu --limit N  timing only, on N randomly chosen sequences (same CPU type as the NCD jobs)
"""
import argparse
import os
import re
import time

import numpy as np
import pandas as pd
import torch

from common import ensure_dir, load_config, log, p, peak_rss_mb, write_json

MODELS = {
    "esm2_650m": {"hf": "facebook/esm2_t33_650M_UR50D", "kind": "esm", "window": 1022},
    "prott5_xl_u50": {"hf": "Rostlab/prot_t5_xl_half_uniref50-enc", "kind": "t5", "window": 1000},
    "nt_v2_500m": {"hf": "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species", "kind": "nt", "window": 5994},
}


def clean(seq, kind):
    s = re.sub(r"[\s\-\*\.]", "", seq.upper())
    if kind == "nt":
        return re.sub(r"[^ACGTN]", "N", s)
    return re.sub(r"[UZOB]", "X", re.sub(r"[^A-Z]", "X", s))


def use_half(device):
    # fp16 only on GPUs with good half-precision support (compute capability >= 7.0);
    # older cards such as the Tesla K80 (3.7) run in fp32
    return device == "cuda" and torch.cuda.get_device_capability(0)[0] >= 7


def load(name, device):
    from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer, T5EncoderModel, T5Tokenizer
    m = MODELS[name]
    if m["kind"] == "esm":
        tok = AutoTokenizer.from_pretrained(m["hf"])
        model = AutoModel.from_pretrained(m["hf"])
    elif m["kind"] == "t5":
        tok = T5Tokenizer.from_pretrained(m["hf"], do_lower_case=False)
        model = T5EncoderModel.from_pretrained(m["hf"], torch_dtype=torch.float16 if use_half(device) else torch.float32)
    else:
        tok = AutoTokenizer.from_pretrained(m["hf"], trust_remote_code=True)
        model = AutoModelForMaskedLM.from_pretrained(m["hf"], trust_remote_code=True)
    model = model.to(device).eval()
    if use_half(device) and m["kind"] != "t5":
        model = model.half()
    return tok, model


@torch.no_grad()
def embed_window(text, name, tok, model, device):
    kind = MODELS[name]["kind"]
    if kind == "t5":
        enc = tok(" ".join(list(text)), return_tensors="pt", add_special_tokens=True).to(device)
        h = model(**enc).last_hidden_state[0]
        h = h[:-1]  # drop </s>
    elif kind == "esm":
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        h = model(**enc).last_hidden_state[0][1:-1]  # drop <cls>, <eos>
    else:
        enc = tok(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1][0][1:]  # drop <cls>
    return h.float().mean(0).cpu().numpy()


def embed_sequence(seq, name, tok, model, device):
    w = MODELS[name]["window"]
    s = clean(seq, MODELS[name]["kind"])
    if not s:
        return None
    parts = [s[i:i + w] for i in range(0, len(s), w)]
    vecs = [embed_window(x, name, tok, model, device) for x in parts]
    wts = np.array([len(x) for x in parts], dtype=float)
    return (np.stack(vecs) * wts[:, None]).sum(0) / wts.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    cfg = load_config(a.config)
    df = pd.read_csv(p(cfg, "data", "processed", a.ds, "sequences.csv"))
    seqs = df["sequence"].astype(str).tolist()
    od = ensure_dir(p(cfg, "data", "emb", a.ds))
    mark = od / (a.model + ".running")
    if a.device == "cuda" and ((od / (a.model + ".npy")).exists() or mark.exists()):
        log("skip", a.ds, a.model); return
    if a.device == "cuda": mark.touch()
    if a.device == "cuda" and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    if a.device == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())))
    t = time.perf_counter()
    tok, model = load(a.model, a.device)
    load_s = time.perf_counter() - t
    idx = np.arange(len(seqs))
    if a.limit:
        idx = np.random.default_rng(0).choice(len(seqs), size=min(a.limit, len(seqs)), replace=False)
    vecs, empty = [], []
    t = time.perf_counter()
    for j, i in enumerate(idx):
        v = embed_sequence(seqs[i], a.model, tok, model, a.device)
        if v is None:
            empty.append(int(i))
        vecs.append(v)
        if j % 500 == 0:
            log(a.ds, a.model, j, "/", len(idx))
    total = time.perf_counter() - t
    timing = {"model": a.model, "hf_id": MODELS[a.model]["hf"], "device": a.device,
              "gpu": torch.cuda.get_device_name(0) if a.device == "cuda" else None,
              "fp16": use_half(a.device),
              "cpu_threads": torch.get_num_threads(), "host": os.uname().nodename,
              "n_sequences": int(len(idx)), "model_load_s": load_s, "embed_wall_s": total,
              "embed_s_per_seq": total / max(1, len(idx)), "peak_rss_mb": peak_rss_mb(),
              "n_empty_after_cleaning": len(empty)}
    if a.device == "cuda":
        timing["gpu_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        dim = next(v for v in vecs if v is not None).shape[0]
        E = np.stack([v if v is not None else np.zeros(dim) for v in vecs]).astype(np.float32)
        np.save(od / f"{a.model}.npy", E)
        write_json(timing, od / f"{a.model}_timing_gpu.json")
    else:
        write_json(timing, od / f"{a.model}_timing_cpu.json")
    log("done", timing)


if __name__ == "__main__":
    main()
