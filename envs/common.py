import contextlib
import io
import os
import random
import sys
import tempfile


@contextlib.contextmanager
def quiet_sandbox():
    prev = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        os.chdir(tmp)
        try:
            yield
        finally:
            os.chdir(prev)


def pair(main_messages, partner_messages_list):
    return {"main": main_messages, "partners": partner_messages_list}


def both_directions(views):
    """Each agent in turn is the main agent; the others are its partners."""
    return [
        pair(views[i], [views[j] for j in range(len(views)) if j != i])
        for i in range(len(views))
    ]


def collect(rollout, n, seed, max_rollouts=100_000):
    rng = random.Random(seed)
    pool = []
    for _ in range(max_rollouts):
        if len(pool) >= n:
            break
        pool.extend(rollout(rng))
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} pairs after {max_rollouts} rollouts")
    rng.shuffle(pool)
    return pool[:n]
