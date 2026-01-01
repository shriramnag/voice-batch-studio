# VoiceBatch Studio Pro - T3 Model (Token-To-Token)
# Copyright (c) 2025 Resemble AI | MIT License

import logging
from typing import Union, Optional, List
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import LlamaModel, LlamaConfig, GPT2Config, GPT2Model
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
    MinPLogitsWarper,
)

# Internal Module Imports
from .modules.learned_pos_emb import LearnedPositionEmbeddings
from .modules.cond_enc import T3CondEnc, T3Cond
from .modules.t3_config import T3Config
from .llama_configs import LLAMA_CONFIGS
from .inference.t3_hf_backend import T3HuggingfaceBackend
from .inference.alignment_stream_analyzer import AlignmentStreamAnalyzer
from ..utils import AttrDict

logger = logging.getLogger(__name__)

def _ensure_BOT_EOT(text_tokens: Tensor, hp):
    B = text_tokens.size(0)
    assert (text_tokens == hp.start_text_token).int().sum() >= B, "missing start_text_token"
    assert (text_tokens == hp.stop_text_token).int().sum() >= B, "missing stop_text_token"

class T3(nn.Module):
    """
    T3 (Token-To-Token) TTS model: यह टेक्स्ट टोकन्स को स्पीच टोकन्स में बदलता है।
    """
    def __init__(self, hp=None):
        if hp is None:
            hp = T3Config.english_only()
        super().__init__()
        self.hp = hp

        config_dict = LLAMA_CONFIGS[hp.llama_config_name]
        self.is_gpt = config_dict.get("model_type") == "gpt2"

        if self.is_gpt:
            self.cfg = GPT2Config(**config_dict)
            self.tfmr = GPT2Model(self.cfg)
        else:
            self.cfg = LlamaConfig(**config_dict)
            self.tfmr = LlamaModel(self.cfg)

        self.dim = self.cfg.hidden_size
        self.cond_enc = T3CondEnc(hp)
        self.text_emb = nn.Embedding(hp.text_tokens_dict_size, self.dim)
        self.speech_emb = nn.Embedding(hp.speech_tokens_dict_size, self.dim)

        # Position Embeddings Setup
        self.text_pos_emb = None
        self.speech_pos_emb = None
        if hp.input_pos_emb == "learned":
            self.text_pos_emb = LearnedPositionEmbeddings(hp.max_text_tokens + 2, self.dim)
            self.speech_pos_emb = LearnedPositionEmbeddings(hp.max_speech_tokens + 4, self.dim)

        self.text_head = nn.Linear(self.cfg.hidden_size, hp.text_tokens_dict_size, bias=False)
        self.speech_head = nn.Linear(self.cfg.hidden_size, hp.speech_tokens_dict_size, bias=self.is_gpt)
        self.compiled = False

    @property
    def device(self):
        return self.speech_head.weight.device

    def prepare_conditioning(self, t3_cond: T3Cond):
        if t3_cond.cond_prompt_speech_tokens is not None and t3_cond.cond_prompt_speech_emb is None:
            t3_cond.cond_prompt_speech_emb = self.speech_emb(t3_cond.cond_prompt_speech_tokens)
            if not self.is_gpt:
                t3_cond.cond_prompt_speech_emb += self.speech_pos_emb(t3_cond.cond_prompt_speech_tokens)
        return self.cond_enc(t3_cond)

    def prepare_input_embeds(self, *, t3_cond, text_tokens, speech_tokens, cfg_weight=0.0):
        cond_emb = self.prepare_conditioning(t3_cond)
        text_emb = self.text_emb(text_tokens)
        if cfg_weight > 0.0 and not self.is_gpt:
            text_emb[1].zero_()

        speech_emb = self.speech_emb(speech_tokens)
        if self.hp.input_pos_emb == "learned":
            text_emb = text_emb + self.text_pos_emb(text_tokens)
            speech_emb = speech_emb + self.speech_pos_emb(speech_tokens)
        
        len_cond = cond_emb.size(1)
        if cond_emb.size(0) != text_emb.size(0):
             cond_emb = cond_emb.expand(text_emb.size(0), -1, -1)

        embeds = torch.stack([torch.cat((ce, te, se)) for ce, te, se in zip(cond_emb, text_emb, speech_emb)])
        return embeds, len_cond

    def forward(self, *, t3_cond, text_tokens, text_token_lens, speech_tokens, speech_token_lens, training=False):
        _ensure_BOT_EOT(text_tokens, self.hp)
        embeds, len_cond = self.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=speech_tokens)

        tfmr_out = self.tfmr.forward(input_ids=None, inputs_embeds=embeds, output_hidden_states=True, return_dict=True, use_cache=(not training))
        hidden_states = tfmr_out.hidden_states[-1]

        len_text, len_speech = text_tokens.size(1), speech_tokens.size(1)
        B, _, dim = hidden_states.shape
        text_latents = torch.zeros(B, len_text, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        speech_latents = torch.zeros(B, len_speech, dim, dtype=hidden_states.dtype, device=hidden_states.device)
        
        for i in range(B):
            text_end = len_cond + text_token_lens[i].item()
            speech_start = len_cond + text_tokens.size(1)
            speech_end = speech_start + speech_token_lens[i].item()
            text_latents[i, :text_token_lens[i]] = hidden_states[i, len_cond:text_end]
            speech_latents[i, :speech_token_lens[i]] = hidden_states[i, speech_start:speech_end]

        return AttrDict(
            text_logits=self.text_head(text_latents),
            speech_logits=self.speech_head(speech_latents),
            hidden_states=hidden_states
        )

    @torch.inference_mode()
    def inference(self, *, t3_cond, text_tokens, max_new_tokens=None, temperature=0.8, top_p=0.95, min_p=0.05, repetition_penalty=1.2, cfg_weight=0.5, **kwargs):
        text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=self.device)
        initial_speech_tokens = self.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])

        embeds, len_cond = self.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight)

        if not self.compiled:
            alignment_stream_analyzer = None
            if self.hp.is_multilingual:
                alignment_stream_analyzer = AlignmentStreamAnalyzer(self.tfmr, None, text_tokens_slice=(len_cond, len_cond + text_tokens.size(-1)), alignment_layer_idx=9, eos_idx=self.hp.stop_speech_token)

            self.patched_model = T3HuggingfaceBackend(config=self.cfg, llama=self.tfmr, speech_enc=self.speech_emb, speech_head=self.speech_head, alignment_stream_analyzer=alignment_stream_analyzer)
            self.compiled = True

        device = embeds.device
        bos_token = torch.tensor([[self.hp.start_speech_token]], dtype=torch.long, device=device)
        bos_embed = self.speech_emb(bos_token)
        if self.speech_pos_emb: bos_embed += self.speech_pos_emb.get_fixed_embedding(0)
        bos_embed = torch.cat([bos_embed, bos_embed])

        inputs_embeds = torch.cat([embeds, bos_embed], dim=1)
        generated_ids = bos_token.clone()
        predicted = []

        repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
        top_p_warper = TopPLogitsWarper(top_p=top_p)
        min_p_warper = MinPLogitsWarper(min_p=min_p)

        output = self.patched_model(inputs_embeds=inputs_embeds, use_cache=True)
        past = output.past_key_values

        for i in tqdm(range(max_new_tokens or self.hp.max_speech_tokens), desc="Sampling"):
            logits_step = output.logits[:, -1, :]
            cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
            logits = cond + cfg_weight * (cond - uncond)
            
            if self.patched_model.alignment_stream_analyzer:
                logits = self.patched_model.alignment_stream_analyzer.step(logits, next_token=generated_ids[0, -1].item())

            logits = repetition_penalty_processor(generated_ids[:1, :], logits)
            if temperature != 1.0: logits /= temperature
            logits = min_p_warper(generated_ids[:1, :], logits)
            logits = top_p_warper(generated_ids[:1, :], logits)

            next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
            predicted.append(next_token)
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

            if next_token.item() == self.hp.stop_speech_token: break

            next_token_embed = self.speech_emb(next_token)
            if self.speech_pos_emb: next_token_embed += self.speech_pos_emb.get_fixed_embedding(i + 1)
            next_token_embed = torch.cat([next_token_embed, next_token_embed])

            output = self.patched_model(inputs_embeds=next_token_embed, past_key_values=past, use_cache=True)
            past = output.past_key_values

        return torch.cat(predicted, dim=1)
