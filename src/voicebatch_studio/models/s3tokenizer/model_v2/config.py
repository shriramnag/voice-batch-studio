from dataclasses import dataclass

@dataclass
class ModelConfig:
    """S3TokenizerV2 के लिए कॉन्फ़िगरेशन"""
    encoder_embed_dim: int = 512
    n_mels: int = 80
    n_fft: int = 400
    hop_size: int = 160
    win_size: int = 400
