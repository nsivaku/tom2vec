from __future__ import annotations

import argparse
import json
import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW

from tom2vec.data import load_pairs
from tom2vec.loss import tom_loss
from tom2vec.model import ToM2Vec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True, help="JSONL of self-play pairs (see tom2vec/data.py)")
    p.add_argument("--out", default="/nas-hdd/[anonymized]/matom/checkpoints/tom2vec")
    p.add_argument("--model-id", default="Qwen/Qwen3-8B")
    p.add_argument("--k-latents", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--max-partner-tokens", type=int, default=4000)
    p.add_argument("--max-context-tokens", type=int, default=4096)
    p.add_argument("--no-thinking", action="store_true", help="disable Qwen3 thinking mode")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--save-every", type=int, default=200)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    examples = load_pairs(args.data)
    print(f"[tom2vec] {len(examples)} training pairs from {args.data}", flush=True)

    model = ToM2Vec(
        model_id=args.model_id, k_latents=args.k_latents, dtype=args.dtype, device=args.device,
        enable_thinking=not args.no_thinking, max_context_tokens=args.max_context_tokens,
    )
    model.train()
    optim = AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    log_path = os.path.join(args.out, "train_log.jsonl")
    t0 = time.time()

    for step in range(1, args.steps + 1):
        batch = random.choices(examples, k=args.batch_size)

        readouts = []
        for ex in batch:
            if ex.target is None:
                per_partner = []
                for partner in ex.partners:
                    z, info = model.extract_targets(
                        model.chat_prompt(partner), max_partner_tokens=args.max_partner_tokens
                    )
                    per_partner.append(z)
                    readouts.append(info)
                ex.target = torch.stack(per_partner).mean(dim=0)

        targets = torch.stack([ex.target for ex in batch]).to(model.device)
        preds = model.emit_latents([model.chat_prompt(ex.main) for ex in batch]).latents
        loss = tom_loss(preds, targets)

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        if step == 1 or step % args.log_every == 0:
            rec = {"step": step, "loss": round(loss.item(), 6), "elapsed_s": round(time.time() - t0, 1)}
            if readouts:
                rec["new_traces"] = len(readouts)
                rec["mean_chunks"] = round(float(np.mean([r["n_chunks"] for r in readouts])), 2)
            print(f"[tom2vec] {json.dumps(rec)}", flush=True)
            with open(log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

        if args.save_every and step % args.save_every == 0:
            model.save_pretrained(os.path.join(args.out, f"step_{step:06d}"))

    model.save_pretrained(os.path.join(args.out, "final"))
    print(f"[tom2vec] saved {os.path.join(args.out, 'final')}", flush=True)


if __name__ == "__main__":
    main()
