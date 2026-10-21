from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

import torch
from torch import nn

from tom2vec.chunking import (
    DEFAULT_K_LATENTS,
    DEFAULT_SENTENCE_DELIMITERS,
    plan_chunk_readout,
)

Messages = List[Dict[str, str]]

_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass
class LatentEmission:
    latents: torch.Tensor  # (B, K, d_model)
    past_key_values: object
    attention_mask: torch.Tensor
    final_logits: Optional[torch.Tensor] = None


class ToM2Vec(nn.Module):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-8B",
        k_latents: int = DEFAULT_K_LATENTS,
        dtype: str = "bfloat16",
        device: str = "cuda",
        enable_thinking: bool = True,
        max_context_tokens: int = 4096,
        delimiters: Sequence[str] = DEFAULT_SENTENCE_DELIMITERS,
    ):
        super().__init__()
        
        self.k_latents = k_latents
        self.enable_thinking = enable_thinking
        self.max_context_tokens = max_context_tokens
        self.delimiters = tuple(delimiters)

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = AutoModelForCausalLM.from_pretrained(model_id, dtype=_DTYPES[dtype])
        self.lm.to(device)
        self.d_model = int(self.lm.config.hidden_size)

        stop = self.lm.generation_config.eos_token_id
        stop = stop if isinstance(stop, list) else [stop]
        self.stop_ids = {int(t) for t in stop + [self.tokenizer.eos_token_id] if t is not None}

    @property
    def device(self) -> torch.device:
        return next(self.lm.parameters()).device

    @property
    def param_dtype(self) -> torch.dtype:
        return next(self.lm.parameters()).dtype


    def chat_prompt(self, messages: Messages) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

    def _tokenize(self, prompts: Sequence[str]) -> Dict[str, torch.Tensor]:
        """Left-pad (so every example's last token is at the right edge, where
        the latent chain starts) and left-truncate (keep the latest state)."""
        tok = self.tokenizer
        prior = tok.padding_side, tok.truncation_side
        tok.padding_side = tok.truncation_side = "left"
        try:
            enc = tok(
                list(prompts), padding=True, truncation=True,
                max_length=self.max_context_tokens, return_tensors="pt",
                add_special_tokens=False,
            )
        finally:
            tok.padding_side, tok.truncation_side = prior
        return {k: v.to(self.device) for k, v in enc.items()}


    def emit_latents(
        self, prompts: Sequence[str], compute_final_logits: bool = False
    ) -> LatentEmission:
        """Emit K latents per prompt. Runs under autograd unless the caller
        wraps it in `torch.no_grad()`."""
        enc = self._tokenize(prompts)
        mask = enc["attention_mask"]
        out = self.lm(
            input_ids=enc["input_ids"], attention_mask=mask,
            use_cache=True, output_hidden_states=True,
        )
        past = out.past_key_values
        z = out.hidden_states[-1][:, -1, :]
        latents = [z]

        n_feeds = self.k_latents if compute_final_logits else self.k_latents - 1
        for step in range(n_feeds):
            mask = torch.cat([mask, mask.new_ones(mask.shape[0], 1)], dim=1)
            out = self.lm(
                inputs_embeds=z.unsqueeze(1).to(self.param_dtype),
                attention_mask=mask, past_key_values=past,
                use_cache=True, output_hidden_states=True,
            )
            past = out.past_key_values
            if step < self.k_latents - 1:
                z = out.hidden_states[-1][:, -1, :]
                latents.append(z)

        return LatentEmission(
            latents=torch.stack(latents, dim=1),
            past_key_values=past,
            attention_mask=mask,
            final_logits=out.logits[:, -1, :] if compute_final_logits else None,
        )


    @torch.no_grad()
    def extract_targets(
        self, partner_prompt: str, max_partner_tokens: int = 4000
    ) -> tuple[torch.Tensor, dict]:

        enc = self._tokenize([partner_prompt])
        ctx_len = enc["input_ids"].shape[1]
        seq = self.lm.generate(
            **enc, max_new_tokens=max_partner_tokens, do_sample=False,
            eos_token_id=sorted(self.stop_ids),
            pad_token_id=self.tokenizer.pad_token_id,
        )
        gen = seq[0, ctx_len:].tolist()
        hit_stop = bool(gen) and gen[-1] in self.stop_ids
        trace = gen[:-1] if hit_stop else gen

        full = enc["input_ids"][0].tolist() + trace
        if hit_stop:
            end_tok = gen[-1]
        else:
            logits = self.lm(input_ids=torch.tensor([full], device=self.device)).logits
            end_tok = int(logits[0, -1].argmax())
        full.append(end_tok)

        hidden = self.lm.get_decoder()(
            input_ids=torch.tensor([full], device=self.device)
        ).last_hidden_state[0, ctx_len:]  # (len(trace) + 1, d)

        if not trace:
            idx = [0] * self.k_latents
            info = {"n_trace_tokens": 0, "n_chunks": 0, "n_filled": 0, "degenerate": True}
        else:
            pieces = self.tokenizer.batch_decode([[t] for t in trace])
            readout = plan_chunk_readout(
                pieces, n_chunk_latents=self.k_latents - 2, delimiters=self.delimiters
            )
            idx = [0] + readout.positions + [len(trace)]
            info = {
                "n_trace_tokens": len(trace), "n_chunks": readout.n_chunks,
                "n_filled": readout.n_filled, "degenerate": readout.degenerate,
            }
        return hidden[idx].float().cpu(), info


    @torch.no_grad()
    def generate(
        self,
        messages: Messages,
        max_new_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 20,
        presence_penalty: float = 1.5,
        seed: Optional[int] = None,
    ) -> str:
        gen_rng = None
        if seed is not None:
            gen_rng = torch.Generator(device=self.device).manual_seed(int(seed))

        emit = self.emit_latents([self.chat_prompt(messages)], compute_final_logits=True)
        past, mask, logits = emit.past_key_values, emit.attention_mask, emit.final_logits

        out: List[int] = []
        for _ in range(max_new_tokens):
            logits = logits[0].float()
            if presence_penalty and out:
                logits[torch.tensor(sorted(set(out)), device=logits.device)] -= presence_penalty
            if temperature <= 0:
                tok = int(logits.argmax())
            else:
                tok = _sample(logits / temperature, top_p, top_k, gen_rng)
            if tok in self.stop_ids:
                break
            out.append(tok)
            mask = torch.cat([mask, mask.new_ones(1, 1)], dim=1)
            step = self.lm(
                input_ids=torch.tensor([[tok]], device=self.device),
                attention_mask=mask, past_key_values=past, use_cache=True,
            )
            past, logits = step.past_key_values, step.logits[:, -1, :]
        return self.tokenizer.decode(out, skip_special_tokens=True)

    def save_pretrained(self, path: str) -> None:
        self.lm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)


def _sample(logits: torch.Tensor, top_p: float, top_k: int, rng) -> int:
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    probs = torch.softmax(logits, dim=-1)
    if top_p and top_p < 1.0:
        sorted_p, order = probs.sort(descending=True)
        drop = sorted_p.cumsum(0) - sorted_p > top_p
        sorted_p[drop] = 0.0
        probs = torch.zeros_like(probs).scatter(0, order, sorted_p)
    return int(torch.multinomial(probs / probs.sum(), 1, generator=rng))
