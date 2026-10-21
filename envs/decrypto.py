import json
import re

from envs import common
from decrypto_env.decrypto import Decrypto
from decrypto_env.prompts import (
    get_decoder_prompt,
    get_encoder_prompt,
    get_interceptor_prompt,
    get_system_prompt,
)

MODE = 0  # Decrypto's default system/role prompt mode


def _messages(role, user_prompt):
    return [
        {"role": "system", "content": get_system_prompt(role, MODE)},
        {"role": "user", "content": user_prompt},
    ]


def parse_hints(text):
    """Three string hints from an Encoder reply, or None (Decrypto's format)."""
    text = text.rsplit("</think>", 1)[-1]
    m = re.search(r"ANSWER:\s*(\{.*\})", text, re.DOTALL)
    if not m:
        return None
    try:
        hints = json.loads(m.group(1)).get("hints")
    except (json.JSONDecodeError, AttributeError):
        return None
    if isinstance(hints, list) and len(hints) == 3 and all(isinstance(h, str) for h in hints):
        return hints
    return None


def build(n_keyword_sets, hint_url, hint_model, seed_offset=100_000, max_tokens=4096,
          retries=3, api_key="EMPTY"):
    from openai import OpenAI

    client = OpenAI(base_url=hint_url, api_key=api_key)
    pairs, skipped = [], 0
    for i in range(n_keyword_sets):
        env = Decrypto()
        code, keywords, info = env.reset(seed=seed_offset + i)
        encoder = _messages("encoder", get_encoder_prompt(info, keywords, code, MODE))

        hints = None
        for attempt in range(retries):
            reply = client.chat.completions.create(
                model=hint_model, messages=encoder, max_tokens=max_tokens,
                temperature=0.0 if attempt == 0 else 0.6, seed=seed_offset + i,
            )
            hints = parse_hints(reply.choices[0].message.content or "")
            if hints:
                break
        if hints is None:
            skipped += 1
            continue

        decoder = _messages("decoder", get_decoder_prompt(keywords, info, hints, MODE))
        interceptor = _messages("interceptor", get_interceptor_prompt(info, hints, MODE))
        pairs.append(common.pair(decoder, [encoder]))
        pairs.append(common.pair(interceptor, [encoder]))
    if skipped:
        print(f"[Warning] skipped")
    return pairs
