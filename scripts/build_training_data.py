import argparse
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from envs import common
from tom2vec.data import write_pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="env", required=True)

    def add(name, n_default, val_default=0.0):
        sp = sub.add_parser(name)
        sp.add_argument("--n", type=int, default=n_default, help="number of samples (Decrypto: keyword sets, two pairs each)")
        sp.add_argument("--out", default="/nas-hdd/[anonymized]/matom/data/")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--val-frac", type=float, default=val_default)
        return sp

    sp = add("overcooked", 4000)
    sp.add_argument("--layout", required=True,
                    help="Overcooked-AI layout name, e.g. cramped_room, asymmetric_advantages, coordination_ring")
    sp.add_argument("--horizon", type=int, default=400)
    add("hanabi", 4000)
    add("collab_escape", 4000)
    add("collab_capture", 4000)
    sp = add("decrypto", 1000)
    sp.add_argument("--hint-url", required=True, help="OpenAI-compatible endpoint serving the base model")
    sp.add_argument("--hint-model", default="Qwen/Qwen3-8B")
    sp.add_argument("--seed-offset", type=int, default=100_000,
                    help="keyword-set seeds start here (keep clear of evaluation seeds)")
    sp = add("negotiation", 0, val_default=0.2)
    sp.add_argument("--manifest", required=True,
                    help="TSV of `game_dir<TAB>history.json`, one played game per line")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.env == "overcooked":
        from envs import overcooked
        pairs = common.collect(overcooked.rollout_fn(args.layout, args.horizon), args.n, args.seed)
        name = f"overcooked_{args.layout}"
    elif args.env in ("hanabi", "collab_escape", "collab_capture"):
        import importlib
        module = importlib.import_module(f"envs.{args.env}")
        pairs = common.collect(module.rollout_fn(), args.n, args.seed)
        name = args.env
    elif args.env == "decrypto":
        from envs import decrypto
        pairs = decrypto.build(args.n, args.hint_url, args.hint_model, seed_offset=args.seed_offset)
        name = "decrypto"
    else:
        from envs import negotiation
        pairs = negotiation.build(negotiation.read_manifest(args.manifest))
        random.Random(args.seed).shuffle(pairs)
        if args.n:
            pairs = pairs[: args.n]
        name = "negotiation"

    out = args.out or os.path.join("data", f"{name}.jsonl")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    n_val = int(round(len(pairs) * args.val_frac))
    if n_val:
        val_out = out[: -len(".jsonl")] + ".val.jsonl" if out.endswith(".jsonl") else out + ".val"
        write_pairs(val_out, pairs[:n_val])
        print(f"wrote {n_val} validation pairs to {val_out}")
    n = write_pairs(out, pairs[n_val:])
    print(f"wrote {n} training pairs to {out}")


if __name__ == "__main__":
    main()
