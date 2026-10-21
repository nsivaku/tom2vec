import argparse
import json
import os
import sys

import numpy as np
from openai import OpenAI
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.overcooked import chat_messages, parse_action

LAYOUTS = ["cramped_room", "asymmetric_advantages", "coordination_ring"]


class ChatAgent:
    def __init__(self, url: str, model: str, max_tokens: int, temperature: float, api_key: str):
        self.client = OpenAI(base_url=url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def act(self, state, index, mdp, seed):
        reply = self.client.chat.completions.create(
            model=self.model,
            messages=chat_messages(state, index, mdp),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            seed=seed,
        )
        return parse_action(reply.choices[0].message.content or "")


def run_episode(layout, horizon, main, partner, main_index, seed):
    mdp = OvercookedGridworld.from_layout_name(layout)
    env = OvercookedEnv.from_mdp(mdp, horizon=horizon)
    env.reset()
    state, done, score, turn = env.state, False, 0, 0
    agents = {main_index: main, 1 - main_index: partner}
    while not done:
        # Per-turn seed keeps episodes reproducible and paired across conditions.
        turn_seed = (seed * 1_000_003 + turn) % (2**31 - 1)
        joint = [agents[i].act(state, i, mdp, turn_seed) for i in range(2)]
        state, reward, done, _ = env.step(joint)
        score += reward
        turn += 1
    return score


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--main-url", required=True)
    p.add_argument("--main-model", default="tom2vec")
    p.add_argument("--partner-url", required=True)
    p.add_argument("--partner-model", default="Qwen/Qwen3-8B")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    p.add_argument("--layouts", nargs="+", default=LAYOUTS)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--main-index", type=int, default=0, choices=[0, 1])
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--out", default="/nas-hdd/[anonymized]/matom/results/overcooked/results.json", help="optional JSON file for per-seed scores")
    args = p.parse_args()

    main_agent = ChatAgent(args.main_url, args.main_model, args.max_tokens, args.temperature, args.api_key)
    partner = ChatAgent(args.partner_url, args.partner_model, args.max_tokens, args.temperature, args.api_key)

    results = {}
    for layout in args.layouts:
        scores = []
        for seed in range(args.seeds):
            scores.append(run_episode(layout, args.horizon, main_agent, partner, args.main_index, seed))
            print(f"{layout} seed={seed} score={scores[-1]}", flush=True)
        sem = np.std(scores, ddof=1) / np.sqrt(len(scores)) if len(scores) > 1 else 0.0
        results[layout] = {"scores": scores, "mean": float(np.mean(scores)), "sem": float(sem)}
        print(f"{layout}: {results[layout]['mean']:.1f} ± {sem:.2f}", flush=True)

    print(f"average: {np.mean([r['mean'] for r in results.values()]):.1f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
