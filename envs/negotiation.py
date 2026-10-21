import json
import os
import sys

from envs import common

from llm_deliberation.initial_prompts import InitialPrompt
from llm_deliberation.rounds import RoundPrompts
from llm_deliberation.save_utils import extract_answer, extract_plan
from llm_deliberation.setup_utils import load_setup

GAMES_DIR = os.path.join(common.STORED_GAMES, "negotiation_games")


class _Party:
    def __init__(self, game_dir, name, spec, roles, initial_deal, agents_num, issues_num,
                 rounds_num, window_size):
        self.name = name
        self.initial = InitialPrompt(
            game_dir, name, spec["file_name"], roles["p1"], roles["p2"],
            num_issues=issues_num, num_agents=agents_num, incentive=spec["incentive"],
        ).return_initial_prompt()
        self.rounds = RoundPrompts(
            name, roles["p1"], initial_deal, incentive=spec["incentive"],
            window_size=window_size, target_agent=roles.get("target", ""),
            rounds_num=rounds_num, agents_num=agents_num,
        )

    def messages(self, history, round_idx):
        # Same structure as Agent.prompt for API models: the initial prompt and
        # the turn prompt as two user messages.
        return [
            {"role": "user", "content": self.initial},
            {"role": "user", "content": self.rounds.build_slot_prompt(history, round_idx)},
        ]


def pairs_from_history(game_dir, history_path, agents_num=6, issues_num=5, rounds_num=24,
                       window_size=6):
    with open(history_path) as f:
        log = json.load(f)
    agents, initial_deal, roles = load_setup(game_dir, agents_num)
    parties = {
        name: _Party(game_dir, name, spec, roles, initial_deal, agents_num, issues_num,
                     rounds_num, window_size)
        for name, spec in agents.items()
    }

    turns = log["rounds"]
    history = {"rounds": [], "plan": {}}
    pairs = []
    for k, turn in enumerate(turns[:-1]):
        round_idx = max(k - 1, 0)
        speaker = turn["agent"]
        main = parties[speaker].messages(history, round_idx)
        partners = [p.messages(history, round_idx) for n, p in parties.items() if n != speaker]
        pairs.append(common.pair(main, partners))

        full = turn["full_answer"]
        history["rounds"].append({"agent": speaker, "public_answer": extract_answer(full)})
        plan = extract_plan(full)
        if plan:
            history["plan"].setdefault(speaker, []).append(plan)
    return pairs


def build(runs, **kwargs):
    """`runs`: iterable of (game_dir, history_json) pairs."""
    pairs = []
    for game_dir, history_path in runs:
        pairs.extend(pairs_from_history(game_dir, history_path, **kwargs))
    return pairs


def read_manifest(path):
    runs = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            game_dir, history_path = line.rstrip("\n").split("\t")
            if not os.path.isdir(game_dir):
                game_dir = os.path.join(GAMES_DIR, game_dir)
            runs.append((game_dir, history_path))
    return runs
