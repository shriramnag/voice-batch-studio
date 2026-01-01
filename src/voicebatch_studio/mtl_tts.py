# VoiceBatch Studio Pro - Multilingual TTS Engine (Supports Hindi & 23+ Languages)
from dataclasses import dataclass
from pathlib import Path
import os

import librosa
import torch
import perth
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors
from huggingface_hub import snapshot_download

# आपके VoiceBatch फोल्डर स्ट्रक्चर से इम्पोर्ट
from .models.t3 import T3
from .models.t3.modules.t3_config import T3Config
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import MTLTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond

REPO_ID = "ResembleAI/chatterbox"

# सपोर्टेड भाषाएँ (हिंदी शामिल है)
SUPPORTED_LANGUAGES = {
  "ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek",
  "en": "English", "es": "Spanish", "fi": "Finnish", "fr": "French",
  "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese",
  "ko": "Korean", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
  "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
  "sw": "Swahili", "tr": "Turkish", "zh": "Chinese",
}

def punc_norm(text: str) -> str:
    """विदेशी और भारतीय विराम चिन्हों को सुधारने के लिए"""
    if len(text) == 0:
        return "You need to add some text."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    punc_to_replace = [
        ("...", ", "), ("…", ", "), (":", ","), ("—", "-"), ("–", "-"),
    ]
    for old, new in punc_to_replace:
        text = text.replace(old, new)
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ",", "।", "॥"} # हिंदी विराम चिन्ह जोड़े गए
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."
    return text

@dataclass
class Conditionals:
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])

class VoiceBatchMultilingualTTS: # नाम बदलकर VoiceBatch कर दिया गया है
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(self, t3: T3, s3gen: S3Gen, ve: VoiceEncoder, tokenizer: MTLTokenizer, device: str, conds: Conditionals = None):
        self.sr = S3GEN_SR
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'VoiceBatchMultilingualTTS':
        ckpt_dir = Path(ckpt_dir)
        ve = VoiceEncoder()
        ve.load_state_dict(torch.load(ckpt_dir / "ve.pt", weights_only=True))
        ve.to(device).eval()

        t3 = T3(T3Config.multilingual())
        t3_state = load_safetensors(ckpt_dir / "t3_mtl23ls_v2.safetensors")
        t3.load_state_dict(t3_state if "model" not in t3_state else t3_state["model"][0])
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(torch.load(ckpt_dir / "s3gen.pt", weights_only=True))
        s3gen.to(device).eval()

        tokenizer = MTLTokenizer(str(ckpt_dir / "grapheme_mtl_merged_expanded_v1.json"))
        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice).to(device)
        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    def generate(self, text, language_id, audio_prompt_path=None, exaggeration=0.5, temperature=0.8):
        if language_id.lower() not in SUPPORTED_LANGUAGES:
            raise ValueError(f"Language {language_id} is not supported.")
        
        if audio_prompt_path:
            # वॉयस क्लोनिंग के लिए रेफरेंस लोड करना
            s3gen_ref_wav, _ = librosa.load(audio_prompt_path, sr=S3GEN_SR)
            ref_16k = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)
            s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav[:self.DEC_COND_LEN], S3GEN_SR, device=self.device)
            ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k], sample_rate=S3_SR)).mean(axis=0, keepdim=True).to(self.device)
            
            t3_cond = T3Cond(speaker_emb=ve_embed, cond_prompt_speech_tokens=None, emotion_adv=exaggeration * torch.ones(1, 1, 1)).to(device=self.device)
            self.conds = Conditionals(t3_cond, s3gen_ref_dict)

        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text, language_id=language_id.lower()).to(self.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)

        with torch.inference_mode():
            speech_tokens = self.t3.inference(t3_cond=self.conds.t3, text_tokens=text_tokens, temperature=temperature)
            speech_tokens = drop_invalid_tokens(speech_tokens[0]).to(self.device)
            wav, _ = self.s3gen.inference(speech_tokens=speech_tokens, ref_dict=self.conds.gen)
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
